# 2nd winning solution for CVPR'22 LOVEU challenge:Track 3

This repo provides a code and the checkpoint that won the 2nd place for the track3 of CVPR'22 LOVEU challenge.

[[Page]](https://showlab.github.io/assistq/)  [[Paper]](https://arxiv.org/abs/2203.04203)   [[LOVEU@CVPR'22 Challenge]](https://sites.google.com/view/loveucvpr22/track-3?authuser=0) [[CodaLab Leaderboard]](https://codalab.lisn.upsaclay.fr/competitions/4642#results)

Click to know the task:

[![Click to see the demo](https://img.youtube.com/vi/3v8ceel9Mos/0.jpg)](https://www.youtube.com/watch?v=3v8ceel9Mos)

Model Architecture (see [[Paper]](https://arxiv.org/abs/2203.04203) for details):

![arch](https://user-images.githubusercontent.com/107248364/173805860-fc6a9660-05ce-4937-afcf-c0fbcb7f8004.png)


## Install
(1) PyTorch. See https://pytorch.org/ for instruction. For example,
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
(2) PyTorch Lightning. See https://www.pytorchlightning.ai/ for instruction. For example,
```
pip install pytorch-lightning
```

## Data

Download training set and testing set (without ground-truth labels) by filling in the [[AssistQ Downloading Agreement]](https://forms.gle/h9A8GxHksWJfPByf7).

Then carefully set your data path in the config file ;)

## Encoding

Before starting, you should encode the instructional videos, scripts, QAs. See [encoder.md](https://github.com/showlab/Q2A/blob/master/encoder/README.md).

## Training & Evaluation

Select the config file and simply train, e.g.,

```
CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/q2a_gru+fps1+maskx-1_vit_b16+bert_b.yaml
```

Our best model can be founded in [best model](https://drive.google.com/file/d/15WawspoHIzTXgs6lP5LNoEVZbVlCX1Va/view?usp=sharing).

To inference a model, e.g.,

```
CUDA_VISIBLE_DEVICES=0 python inference.py --cfg configs/q2a_gru+fps1+maskx-1_vit_b16+bert_b.yaml CKPT "best_model_path"
```


The evaluation will be performed after each epoch. You can use Tensorboard, or just terminal outputs to record evaluation results.

## Baseline Performance for LOVEU@CVPR2022 Challenge: 80 videos' QA samples for training, 20 videos' QA samples for testing

|  Model   | Recall@1 ↑ | Recall@3 ↑ | MR (Mean Rank) ↓ | MRR (Mean Reciprocal Rank) ↑ |
|  ----  |  ----  |  ----  |  ----  |  ----  |
| Q2A ([configs/q2a_gru+fps1+maskx-1_vit_b16+bert_b.yaml](configs/q2a_gru+fps1+maskx-1_vit_b16+bert_b.yaml)) | 0.38 (2) | 0.75 (1) | 2.69 (1) | 3.11 (3) |

![image](https://user-images.githubusercontent.com/20626415/166685483-7c39c6e3-8d3d-43a2-a431-58bdcf67cc16.png)

## Contact

Feel free to contact us if you have any problems: khy0501@unist.ac.kr, or leave an issue in this repo.


## Thank you!