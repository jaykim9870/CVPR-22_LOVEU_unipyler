
import torch, os, json
from torch import nn
from torch.functional import F
from pytorch_lightning import LightningModule
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR

class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.act1 = nn.PReLU() #self.act = nn.ReLU()

        self.fc2 = nn.Linear(in_features, in_features)
        self.act2 = nn.PReLU()

        self.fc3 = nn.Linear(in_features,out_features)

        self.in_features = in_features
        self.out_features = out_features
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)

        return x

class Q2A(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.mlp_v = MLP(cfg.INPUT.DIM, cfg.INPUT.DIM)
        self.mlp_t = MLP(cfg.INPUT.DIM, cfg.INPUT.DIM)
        self.mlp_pre = MLP(cfg.INPUT.DIM*(3+cfg.INPUT.NUM_MASKS), cfg.MODEL.DIM_STATE)

        self.s2v = nn.MultiheadAttention(cfg.INPUT.DIM, cfg.MODEL.NUM_HEADS)
        self.v2s = nn.MultiheadAttention(cfg.INPUT.DIM, cfg.MODEL.NUM_HEADS)
        self.qa2s = nn.MultiheadAttention(cfg.INPUT.DIM, cfg.MODEL.NUM_HEADS)
        self.ab2at = nn.MultiheadAttention(cfg.INPUT.DIM, cfg.MODEL.NUM_HEADS)
        self.q2a = nn.MultiheadAttention(cfg.INPUT.DIM, cfg.MODEL.NUM_HEADS)
        self.v2b = nn.MultiheadAttention(cfg.INPUT.DIM, cfg.MODEL.NUM_HEADS)        
        
        self.state = torch.randn(cfg.MODEL.DIM_STATE)
        if cfg.MODEL.HISTORY.ARCH == "mlp":
            self.proj = MLP(cfg.MODEL.DIM_STATE*2, 1)
        elif cfg.MODEL.HISTORY.ARCH == "gru":
            self.gru = nn.GRUCell(cfg.MODEL.DIM_STATE, cfg.MODEL.DIM_STATE)
            self.proj = MLP(cfg.MODEL.DIM_STATE, 1)
        else:
            assert False, "unknown arch"
        
        self.history_train = cfg.MODEL.HISTORY.TRAIN
        self.history_val = cfg.MODEL.HISTORY.VAL
        self.cfg = cfg

    def forward(self, batch):
        loss, count = 0, 0
        results = []
        for video, script, question, actions, label, meta in batch:
            video = self.mlp_v(video) 
            script = self.mlp_t(script)
            video = self.s2v(script.unsqueeze(1), video.unsqueeze(1), video.unsqueeze(1))[0].squeeze_()
            question = self.mlp_t(question)

            scores = []
            for i, actions_per_step in enumerate(actions):
                a_texts, a_buttons = zip(*[(action['text'], action['button']) for action in actions_per_step])
                a_texts = self.mlp_t(torch.cat(a_texts))
                A = len(a_buttons)
                a_buttons = self.mlp_v(
                    torch.stack(a_buttons).view(A, -1, a_texts.shape[1])
                ).view(A, -1) 
                answer_f, _= self.ab2at(a_buttons,a_texts, a_texts)
                q2answer, _ = self.q2a(answer_f, question, question)
                qa_script, qa_script_mask = self.qa2s(
                    q2answer.unsqueeze(1), script.unsqueeze(1), script.unsqueeze(1)
                )
                qa_video = qa_script_mask @ video 
                
                inputs = torch.cat(
                    [qa_video[0], q2answer, a_buttons, a_texts],
                    dim=1
                )
                inputs = self.mlp_pre(inputs)
                state = self.state.to(inputs.device)
                if hasattr(self, "gru"):
                    states = self.gru(inputs, state.expand_as(inputs))
                else:
                    states = torch.cat([inputs, state.expand_as(inputs)], dim=1)
                logits = self.proj(states)
                if self.training:
                    loss += F.cross_entropy(logits.view(1, -1), label[i].view(-1))
                    count += 1
                else:
                    scores.append(logits.view(-1).tolist())
                if self.history_train == "gt" and self.training:
                    state = inputs[label[i]]
                if (self.history_train == "max" and self.training) \
                    or (self.history_val == "max" and not self.training):
                    state = inputs[logits.argmax()]
            if not self.training:
                meta["scores"] = scores
                results.append(meta)
        if self.training:
            return loss / count
        else:
            return results

models = {"q2a": Q2A}

class ModelModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = models[cfg.MODEL.ARCH](cfg)
        self.cfg = cfg
    
    def training_step(self, batch, idx):
        loss = self.model(batch)
        dataset = self.trainer.datamodule.__class__.__name__
        self.log(f"{dataset} loss", loss, rank_zero_only=True)
        return loss
    
    def configure_optimizers(self):
        cfg = self.cfg
        model = self.model
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.SOLVER.LR,
            momentum=0.9, weight_decay=0.0005) 
        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, 
            warmup_epochs=cfg.SOLVER.WARMUP_EPOCHS, max_epochs=cfg.SOLVER.MAX_EPOCHS, 
            warmup_start_lr=cfg.SOLVER.LR*0.1)
        return [optimizer], [lr_scheduler]
    
    def validation_step(self, batch, idx):
        batched_results = self.model(batch)
        return batched_results
            
    def validation_epoch_end(self, outputs) -> None:
        self.shared_epoch_end(outputs, "val")
    
    def test_step(self, batch, idx):
        batched_results = self.model(batch)
        return batched_results

    def test_epoch_end(self, outputs) -> None:
        self.shared_epoch_end(outputs, "test")
    
    def shared_epoch_end(self, outputs, mode):
        from eval_for_loveu_cvpr2022 import evaluate
        results = sum(outputs, [])
        all_preds = {}
        for result in results:
            pred = dict(
                question=result['question'], 
                scores=result['scores']
            )
            folder = result['folder']
            if folder not in all_preds:
                all_preds[folder] = []
            all_preds[folder].append(pred)

        if mode=='val':         #self.cfg.DATASET.GT:
            with open(self.cfg.DATASET.GT) as f:
                all_annos = json.load(f)
            r1, r3, mr, mrr = evaluate(all_preds, all_annos)
            dataset = self.trainer.datamodule.__class__.__name__
            # for tensorboard
            self.log(f"{dataset} recall@1", r1, rank_zero_only=True)
            self.log(f"{dataset} recall@3", r3, rank_zero_only=True)
            self.log(f"{dataset} mean_rank", mr, rank_zero_only=True)
            self.log(f"{dataset} mrr", mrr)
        else:
            OUTPUT_DIR = "test_results"
            json_name = f"submit_test_{self.current_epoch}.json"
            json_file = os.path.join(OUTPUT_DIR, json_name)
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            print("\n No ground-truth labels for test \n")
            print(f"Generating json file at {json_file}. You can zip and submit it to CodaLab ;)")
            with open(json_file, 'w') as f:
                json.dump(all_preds, f)

def build_model(cfg):
    return ModelModule(cfg)