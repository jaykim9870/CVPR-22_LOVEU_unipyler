from transformers import EarlyStoppingCallback
from data import build_data
from model import build_model
from configs import build_config
import datetime as dt
import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

x = dt.datetime.now()
display_name = 'stage1_epoch50//' + x.strftime("%Y-%m-%d %H:%M")
wandb_logger = WandbLogger()

if __name__ == "__main__":
    seed_everything(0, workers=True)
    cfg = build_config()
    dataset = build_data(cfg)
    model = build_model(cfg)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=30,
        verbose=True,
        monitor= "EncodedAssistQADataModule mean_rank", #training loss
        mode="min",
        save_last=True,
        dirpath='./outputs/khy_mlp_og_stage2'
    )

    trainer = Trainer(
        accelerator="gpu",
        gpus=cfg['NUM_GPUS'],
        strategy=DDPPlugin(find_unused_parameters=True),
        callbacks=[
            LearningRateMonitor(logging_interval='step'), 
            checkpoint_callback
        ],
        benchmark=False, 
        deterministic=True,
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        logger= wandb_logger,
        default_root_dir=cfg.OUTPUT_DIR,
        check_val_every_n_epoch=cfg.CHECK_VAL_EVERY_N_EPOCH,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=dataset, 
        ckpt_path=cfg.CKPT if hasattr(cfg, "CKPT") else None)
    