# SegFormer-based Embedding-Patch Knowledge Distillation for High-Performance Crop Segmentation

## Introduction

we proposed a knowledge distillation method based on features and embedding patches. This method can effectively utilize high-level feature information and the texture infor-mation of embedding patches to achieve effective distillation at different levels. In addition, we utilized the attention mechanism to enhance the effect of the distilled features, which further improves the distillation performance.

<div align=center>
	<img src="https://github.com/xl-alt/SEKD/blob/main/The%20block%20framework%20of%20the%20SEKD.PNG?raw=true" width="600">
</div>

Our contributions are as follows:
- We proposed an embedding knowledge distillation method, which improves the distillation effect by effectively utilizing the texture information in the embedding patches.
- We introduce both feature relation distillation and self-supervised feature recon-struction distillation into the knowledge distillation framework, effectively im-proving the distillation efficiency.


## requirements

It can be easily imported in the following ways：

```
pip install -r requirements.txt
```

Or introduce the following packages separately：

```
torch~=2.0.1+cu117
timm~=1.0.8
mmcv~=2.2.0
numpy~=1.26.2
ipython~=8.15.0
torchvision~=0.15.2+cu117
thop~=0.1.1-2209072238
pillow~=9.3.0
argparse~=1.4.0
visdom~=0.2.4
ptflops~=0.7.4
opencv-python~=4.8.0.76
scipy~=1.11.3
utils~=1.0.2
```

## Dataset

the VOC2012 dataset at [http://host.robots.ox.ac.uk/pascal/VOC/voc2012/](VOC2012 Dataset).
the Cityscapes dataset at [https://www.cityscapes-dataset.com/](Cityscapes Dataset). 
The Sweet Pepper Dataset is currently not available for download, please contact the author for download if you need it.
The Sugar Dataset is available at [https://www.ipb.uni-bonn.de/data/sugarbeets2016/](Sugar Dataset).

## weight

Pre-training weights：

[Segformer weights Download](#)

The distillation weights we have trained can be used to reproduce the visualization in the paper:

[Our weights Download](#)


## Train

You need to download the dataset in advance, process the dataset according to the VOC format, and divide the proportion of the dataset in advance.The Cityscapes data format remains unchanged, and all other datasets are treated in VOC format



Train model:

```
python train_sekd.py --cuda --model SegformerB0 --dataset VOC --datadir ./VOC2012 --num-epochs 500 --num-workers 4 --batch-size 4 --steps-loss 50 --steps-plot 50 --epochs-save 0 --savedate --iouVal --device cuda --knowledge-distillation-loss MultiScaleContextAlignmentDistillationLoss --kd-tau 1.0 --kd-alpha 0.5 --review-kd-loss-weight 1.0 --teacher-val True --temperature 1.0 --divergence --lr 5e-5 --height 512 --width 512

python train_student.py --cuda --model SegformerB0 --dataset VOC --datadir ./VOC2012 --height 512 --num-epochs 500 --num-workers 4 --batch-size 4 --steps-loss 50 --steps-plot 50 --epochs-save 0 --savedir ckpt --savedate --iouVal --device cuda

python train_teacher.py --cuda --model SegformerB3 --dataset VOC --datadir ./VOC2012 --height 512 --num-epochs 500 --num-workers 4 --batch-size 4 --steps-loss 50 --steps-plot 50 --epochs-save 0 --savedir ckpt --savedate --iouVal --device cuda

```


Evaluation:

```
python eval/eval.py --state /path/to/model.pth --subset val --datadir /VOC2012 --distillation-type sekd --num-workers 4 --batch-size 1
```

If you plan to run on a CPU, you can add the --cpu parameter:

```
python eval/eval.py --state /path/to/model.pth --subset val --datadir /VOC2012 --distillation-type sekd --num-workers 4 --batch-size 1 --cpu 
```


## Reproduction Results

**Cityscapes Dataset:**

| Model Name  | Params (M) | FLOPs (G) | mIoU (%)         |
|-------------|------------|-----------|------------------|
| SegFormerB0 | 3.23       | 13.67     | 53.40            |
| SegFormerB2 | 27.36      | 113.84    | 76.37            |
| SEKD        | 5.21       | 14.77     | 62.93 (+9.53)    |
| SEKD-SU     | 5.21       | 14.77     | **63.96 (+10.56)** |
| SEKD-CT     | 5.21       | 14.77     | 63.15 (+9.75)    |


**Sweet Pepper Dataset:**

| Model Name  | Params (M) | FLOPs (G) | mIoU (%)         |
|-------------|------------|-----------|------------------|
| SegFormerB0 | 3.72       | 13.59     | 37.83            |
| SegFormerB3 | 47.23      | 142.87    | 67.54            |
| SEKD        | 4.59       | 16.41     | 40.56 (+2.73)    |
| SEKD-SU     | 4.59       | 16.41     | **45.42 (+7.59)** |
| SEKD-CT     | 4.59       | 16.41     | 44.40 (+6.57)    |

**Sugar Dataset:**

| Model Name  | Params (M) | FLOPs (G) | mIoU (%)          |
|-------------|------------|-----------|-------------------|
| SegFormerB0 | 3.71       | 13.55     | 74.44             |
| SegFormerB3 | 47.22      | 142.75    | 87.38             |
| SEKD        | 4.58       | 16.37     | 76.73 (+2.29)     |
| SEKD-SU     | 4.58       | 16.37     | 79.77 (+5.33)     |
| SEKD-CT     | 4.58       | 16.37     | **81.23 (+6.79)** |




Due to some random operations in the training stage, reproduced results (run once) are slightly different from the reported in paper.

## Acknowledgement

Our code is an improvement based on [Distilling Knowledge via Knowledge Review] (https://arxiv.org/abs/2104.09044) and [TransKD] (https://arxiv.org/abs/2202.13393), thanks to them for their great work!



























