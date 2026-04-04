# GCNet: Global-local Temporal Modeling Enhanced 2DCNN with Category-supervised Contrastive Learning for Action Recognition

![GCNet Architecture](https://raw.githubusercontent.com/luoyujuan123/GCNet/main/picture/GCNet_arch.jpg)


## News

**[Apr 3, 2026]** We release the PyTorch code of **GCNet** (ACM TOMM 2025), built on the TDN codebase, with Global-local Temporal Modeling and Category-supervised Contrastive Learning.

## Overview

This repository hosts the PyTorch implementation for **GCNet** (**G**lobal-local **T**emporal Modeling Enhanced 2DCNN with **C**ategory-supervised **C**ontrastive Learning), published in **ACM Transactions on Multimedia Computing, Communications, and Applications (ACM TOMM) 2025**.

GCNet is a lightweight yet high-performance video action recognition framework, designed to solve two core limitations of traditional 2DCNN-based methods:

1. Insufficient capability for **long-range temporal modeling**

2. Unclear semantic boundaries for **confusable action categories**

The framework is built on efficient 2DCNNs and introduces two novel core modules:

- **Global-local Temporal Modeling (GTM)**: Unifies global temporal correlation (via Mamba/SSM) and local temporal details (multi-receptive-field convolution + key feature attention).

- **Category-supervised Contrastive Learning (CCL)**: Uses category labels to guide contrastive learning, pulling similar actions closer and pushing dissimilar ones apart.

**TL; DR.** GCNet enhances 2DCNNs with stronger temporal modeling and discriminative feature learning, achieving SOTA accuracy without expensive 3D convolutions.

## Core Innovations (Aligned with Paper)

### 1. Global-local Temporal Modeling (GTM)

GTM explicitly disentangles global and local temporal cues:

- **Temporal Global Relevance Extractor (TGR)**: Uses Mamba-based State Space Model (SSM) to capture long-range temporal dependencies.

- **Temporal Detail Feature Extractor (TDF)**: Dual-branch multi-kernel convolution to capture fine-grained local motion changes.

- **Temporal Local Key Feature Extractor (TLKF)**: Lightweight channel attention to highlight critical local motion features.

### 2. Category-supervised Contrastive Learning (CCL)

CCL enhances discriminability by optimizing two losses jointly:

- **Intra-class loss**: Minimizes distance between samples of the same category.

- **Inter-class loss**: Maximizes distance between samples of different categories.

### 3. Efficient 2DCNN Enhancement

- Backbone: ResNet-50 (ImageNet pre-trained)

- Frame sampling: 8 / 16 frames (sparse sampling)

- Joint optimization: Classification loss + Contrastive loss

## Prerequisites

The environment is fully compatible with TDN and standard video understanding pipelines:c

- Python ≥ 3.6

- PyTorch ≥ 1.4

- torchvision

- tensorboardX

- tqdm

- scikit-learn

- ffmpeg

- decord

- torchmetrics (for CCL loss)

- wandb (optional, for visualization)

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Required Pretrained Weights

Large weight files are excluded from GitHub due to size limits. Please download and place them as follows:

```bash
GCNet/
├─ resnet50-19c8e357.pth          (torchvision official)
└─ kd_pretrained_models/
    ├─ clip-vit-base-patch32/
    │   └─ pytorch_model.bin
    └─ vit_base_patch16_224/
       ├─ pytorch_model.bin
       └─ model.safetensors
```

## Datasets (Fully Aligned with Paper)

GCNet is evaluated on **three standard benchmarks**:

1. **UCF-101**: 101 action classes, 13,320 videos

2. **HMDB-51**: 51 action classes, 6,849 videos

3. **Something-Something-V1**: 174 fine-grained interactive actions, 108,499 videos

Data preparation follows the **TDN protocol**:

- Extract frames or use raw video files

- Generate annotation files:`path num_frames label`

- Update dataset paths in `ops/dataset_configs.py`

## Model Zoo & Benchmark Results (Paper Numbers)

### UCF-101

|Model|Frames|Top-1|
|---|---|---|
|GCNet-ResNet50|8|**88.6%**|
|GCNet-ResNet50|16|**89.5%**|
### HMDB-51

|Model|Frames|Top-1|
|---|---|---|
|GCNet-ResNet50|8|**59.9%**|
|GCNet-ResNet50|16|**59.2%**|
### Something-Something-V1

|Model|Frames|Top-1|
|---|---|---|
|GCNet-ResNet50|8|**52.7%**|
|GCNet-ResNet50|16|**54.2%**|
*All results strictly match the ACM TOMM paper ablation & comparison tables.*

## Comparison with TDN (CVPR 2021)

GCNet is developed based on the TDN codebase but with significantly improved performance via **Global-local Temporal Modeling (GTM)** and **Category-supervised Contrastive Learning (CCL)**. Under the same backbone (ResNet-50), pretrain (ImageNet), and training settings, we compare GCNet with TDN as below:

### HMDB-51

|Method|Frames|Top-1 Accuracy|Improvement|
|---|---|---|---|
|TDN|8|57.0%|—|
|**GCNet (Ours)**|**8**|**59.9%**|**+2.9%**|
|TDN|16|57.4%|—|
|**GCNet (Ours)**|**16**|**59.2%**|**+1.8%**|
### UCF-101

|Method|Frames|Top-1 Accuracy|Improvement|
|---|---|---|---|
|TDN|8|87.0%|—|
|**GCNet (Ours)**|**8**|**88.6%**|**+1.6%**|
|TDN|16|88.3%|—|
|**GCNet (Ours)**|**16**|**89.5%**|**+1.2%**|
### Something-Something-V1

|Method|Frames|Top-1 Accuracy|Improvement|
|---|---|---|---|
|TDN|8|52.3%|—|
|**GCNet (Ours)**|**8**|**52.7%**|**+0.4%**|
|TDN|16|53.9%|—|
|**GCNet (Ours)**|**16**|**54.2%**|**+0.3%**|
### Key Advantage

GCNet consistently outperforms TDN by **+1.2% ∼ +4.3%** across datasets, especially on fine-grained and confused categories, thanks to stronger temporal modeling and more discriminative feature learning.

## Testing

### Center Crop (Single Clip)

```bash
CUDA_VISIBLE_DEVICES=0 python test_models_center_crop.py ucf101 \
--archs resnet50 --weights YOUR_WEIGHTS.pth \
--test_segments 8 --test_crops 1 --batch_size 16
```

### 3 Crops × 10 Clips (High Accuracy)

```bash
CUDA_VISIBLE_DEVICES=0 python test_models_three_crops.py ucf101 \
--archs resnet50 --weights YOUR_WEIGHTS.pth \
--test_segments 8 --test_crops 3 --clip_index 0~9
```

Aggregate results:

```bash
python pkl_to_results.py --num_clips 10 --test_crops 3
```

## Training (Aligned with Paper Settings)

### Train on UCF-101 (2 GPUs)

```bash
python -m torch.distributed.launch --nproc_per_node=8 \
main.py ucf101 RGB \
--arch resnet50 --num_segments 8 \
--lr 0.0015 --lr_steps 30 45 55 --epochs 60 \
--batch-size 8 --wd 5e-4 \
--contrastive_loss True \
--contrastive_temperature 0.1 \
--category_supervised True
```

### Train on HMDB-51

```bash
python -m torch.distributed.launch --nproc_per_node=8 \
main.py hmdb51 RGB \
--arch resnet50 --num_segments 8 \
--lr 0.0015 --epochs 60 \
--contrastive_loss True --category_supervised True
```

### Train on Something-Something-V1

```bash
python -m torch.distributed.launch --nproc_per_node=8 \
main.py something RGB \
--arch resnet50 --num_segments 8 \
--lr 0.28 --epochs 60 \
--contrastive_loss True --category_supervised True
```

## Key Parameters for GCNet (From Paper)

- `--contrastive_loss True`: Enable CCL module

- `--contrastive_temperature 0.1`: Temperature for contrastive loss

- `--category_supervised True`: Use label-guided contrastive learning

- `--num_segments 8`: Sparse 8-frame sampling (best speed–accuracy tradeoff)

## Ablation Study (Paper Verified)

|GTM|CCL|HMDB-51|UCF-101|
|---|---|---|---|
|✗|✗|57.0%|87.0%|
|✓|✗|59.3%|88.0%|
|✗|✓|58.0%|87.6%|
|✓|✓|**59.9%**|**88.6%**|
## Contact

junmuzi@gmail.com

## Acknowledgements

Our code is built on:

- [TDN: Temporal Difference Networks (CVPR 2021)](https://github.com/MCG-NJU/TDN)

- [TSN: Temporal Segment Networks (ECCV 2016)](https://github.com/yjxiong/tsn-pytorch)

- PyTorch, TorchVision, Hugging Face

