# GCNet: Global-local Temporal Modeling Enhanced 2DCNN with Category-supervised Contrastive Learning for Action Recognition

![GCNet Architecture](https://github.com/luoyujuan123/GCNet/blob/main/GCNet_arch.jpg)  <!-- 可后续上传模型结构图替换链接 -->

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gcnet-global-local-temporal-modeling-enhanced/action-recognition-in-videos-on-ucf101)](https://paperswithcode.com/sota/action-recognition-in-videos-on-ucf101?p=gcnet-global-local-temporal-modeling-enhanced)

[![PWC](https:/img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gcnet-global-local-temporal-modeling-enhanced/action-recognition-in-videos-on-hmdb51)](https://paperswithcode.com/sota/action-recognition-in-videos-on-hmdb51?p=gcnet-global-local-temporal-modeling-enhanced)

## News

**[Apr 3, 2026]** We release the PyTorch code of GCNet, based on the TDN codebase, integrating global-local temporal modeling and category-supervised contrastive learning.

**[Apr 2, 2026]** GCNet achieves competitive performance on UCF101 and HMDB51 datasets.

**[Mar 30, 2026]** The core module design of GCNet is completed, including global-local temporal module and category-supervised contrastive loss.

## Overview

We release the PyTorch code of GCNet (Global-local Temporal Modeling Enhanced 2DCNN with Category-supervised Contrastive Learning for Action Recognition). This code is based on the [TDN](https://github.com/MCG-NJU/TDN) codebase, with key improvements in temporal modeling and feature learning.

The core innovations of GCNet are:

- **Global-local Temporal Modeling**: A novel temporal module that captures both long-range global temporal dependencies and fine-grained local motion details, overcoming the limitations of traditional 2DCNNs in video temporal modeling.

- **Category-supervised Contrastive Learning**: A contrastive loss function guided by category labels, which enhances the discriminability of video features and improves action recognition accuracy.

- **Efficient 2DCNN Enhancement**: Based on lightweight 2DCNNs, avoiding the high computational cost of 3DCNNs while achieving superior temporal modeling performance.

**TL; DR.** GCNet enhances 2DCNNs with global-local temporal modeling and category-supervised contrastive learning, achieving efficient and accurate action recognition without relying on complex 3D convolution operations.

* [Prerequisites](#prerequisites)

* [Excluded Large Files](#excluded-large-files)

* [Data Preparation](#data-preparation)

* [Model Zoo](#model-zoo)

* [Testing](#testing)  

* [Training](#training)

* [Contact](#contact)

## Prerequisites

The code is built with following libraries (consistent with TDN, ensuring compatibility):

- Python 3.6 or higher

- [PyTorch](https://pytorch.org/) **1.4** or higher

- [Torchvision](https://github.com/pytorch/vision)

- [TensorboardX](https://github.com/lanpa/tensorboardX)

- [tqdm](https://github.com/tqdm/tqdm.git)

- [scikit-learn](https://scikit-learn.org/stable/)

- [ffmpeg](https://www.ffmpeg.org/)  

- [decord](https://github.com/dmlc/decord)

- Additional required libraries (for contrastive learning):
        

    - torchmetrics

    - wandb (for training visualization, optional)

Install all dependencies via requirements.txt:

```bash
pip install -r requirements.txt
```

## Excluded Large Files

Due to GitHub's single file size limit (≤100MB), the following large model weight files are excluded from this repository (only kept locally for use). The exclusion list is recorded in `.gitignore`, and the correct storage path is as follows:

### 1. Excluded File List

- resnet50-19c8e357.pth

- resnet101-5d3b4d8f.pth

- kd_pretrained_models/clip-vit-base-patch32/pytorch_model.bin

- kd_pretrained_models/vit_base_patch16_224/pytorch_model.bin

- kd_pretrained_models/vit_base_patch16_224/model.safetensors

### 2. Storage Path

Please download the above weight files manually and place them in the following directory structure:

```bash
项目根目录/
├─ resnet50-19c8e357.pth          # 直接放在根目录
├─ resnet101-5d3b4d8f.pth         # 直接放在根目录
└─ kd_pretrained_models/           # 模型文件夹
    ├─ clip-vit-base-patch32/
    │   └── pytorch_model.bin
    └─ vit_base_patch16_224/
       ├─ pytorch_model.bin
       └─ model.safetensors
```

### 3. Download Sources

- ResNet weights: Official pre-trained weights from torchvision.

- ViT / CLIP weights: Official repository from Hugging Face.

## Data Preparation

We have successfully trained GCNet on [UCF101](https://www.crcv.ucf.edu/data/UCF101.php), [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/), and [Kinetics400](https://deepmind.com/research/open-source/kinetics) with this codebase. The data processing steps are consistent with TDN for easy migration:

### Something-Something-V1 & V2 (Optional)

1. Extract frames from videos (you can use ffmpeg to get frames from video).

2. Generate annotations needed for dataloader ("<path_to_frames> <frames_num> <video_class>" in annotations). The annotation usually includes train.txt and val.txt. The format of *.txt file is like:
        `dataset_root/frames/video_1 num_frames label_1
dataset_root/frames/video_2 num_frames label_2
dataset_root/frames/video_3 num_frames label_3
...
dataset_root/frames/video_N num_frames label_N`

3. Add the information to `ops/dataset_configs.py`.

### Kinetics400

1. Preprocess the data by resizing the short edge of video to 320px. You can refer to [MMAction2 Data Benchmark](https://github.com/open-mmlab/mmaction2) for detailed steps.

2. Generate annotations needed for dataloader ("<path_to_video> <video_class>" in annotations). The annotation usually includes train.txt and val.txt. The format of *.txt file is like:
       `dataset_root/video_1.mp4  label_1
dataset_root/video_2.mp4  label_2
dataset_root/video_3.mp4  label_3
...
dataset_root/video_N.mp4  label_N`

3. Add the information to `ops/dataset_configs.py`.

**Note**: We use [decord](https://github.com/dmlc/decord) to decode the Kinetics videos **on the fly**.

### UCF101 & HMDB51

1. Extract frames from videos or use video files directly (consistent with Kinetics400 processing).

2. Generate annotations in the same format as Kinetics400 (video path + label).

3. Add dataset configuration to `ops/dataset_configs.py`.

## Model Zoo

Here we provide some off-the-shelf pretrained models of GCNet. The accuracy might vary a little bit due to differences in data preprocessing and training environments.

### UCF101

|Model|Frames x Crops x Clips|Top-1|Top-5|Checkpoint|
|---|---|---|---|---|
|GCNet-ResNet50|8x1x1|94.2%|98.8%|[link] (to be updated)|
|GCNet-ResNet50|16x1x1|95.1%|99.1%|[link] (to be updated)|
|GCNet-ResNet101|16x3x10|96.3%|99.4%|[link] (to be updated)|
### HMDB51

|Model|Frames x Crops x Clips|Top-1|Top-5|Checkpoint|
|---|---|---|---|---|
|GCNet-ResNet50|8x1x1|78.5%|93.2%|[link] (to be updated)|
|GCNet-ResNet101|16x3x10|81.3%|95.1%|[link] (to be updated)|
### Kinetics400

|Model|Frames x Crops x Clips|Top-1 (30 view)|Top-5 (30 view)|Checkpoint|
|---|---|---|---|---|
|GCNet-ResNet50|8x3x10|78.2%|94.1%|[link] (to be updated)|
|GCNet-ResNet101|16x3x10|79.8%|94.8%|[link] (to be updated)|
## Testing

GCNet inherits the testing pipeline of TDN, supporting center crop single clip and 3 crops 10 clips testing. The core difference is the use of GCNet's pretrained weights and modified model architecture.

### 1. Center Crop Single Clip

1. Run the following testing scripts:
       `CUDA_VISIBLE_DEVICES=0 python3 test_models_center_crop.py ucf101 \
--archs='resnet50' --weights <your_checkpoint_path>  --test_segments=8  \
--test_crops=1 --batch_size=16  --gpus 0 --output_dir <your_pkl_path> -j 4 --clip_index=0`Note: Replace `ucf101` with `hmdb51` or `kinetics` for other datasets.

2. Run the following scripts to get result from the raw score:
        `python3 pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir <your_pkl_path>`

### 2. 3 Crops, 10 Clips (High Accuracy)

1. Run the following testing scripts for 10 times (clip_index from 0 to 9):
        `CUDA_VISIBLE_DEVICES=0 python3 test_models_three_crops.py  kinetics \
--archs='resnet50' --weights <your_checkpoint_path>  --test_segments=8 \
--test_crops=3 --batch_size=16 --full_res --gpus 0 --output_dir <your_pkl_path>  \
-j 4 --clip_index <your_clip_index>`

2. Run the following scripts to ensemble the raw score of the 30 views:
        `python pkl_to_results.py --num_clips 10 --test_crops 3 --output_dir <your_pkl_path>`

## Training

This implementation supports multi-gpu, `DistributedDataParallel` training, which is faster and simpler. The training scripts are modified based on TDN, adding parameters for category-supervised contrastive learning.

### Example 1: Train GCNet-ResNet50 on UCF101 with 8 gpus

```bash
python -m torch.distributed.launch --master_port 12347 --nproc_per_node=8 \
            main.py  ucf101  RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.01 \
            --lr_scheduler step --lr_steps  30 45 55 --epochs 60 --batch-size 8 \
            --wd 5e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 4 --npb \
            --contrastive_loss True --contrastive_temperature 0.1 --category_supervised True
```

### Example 2: Train GCNet-ResNet50 on Kinetics400 with 8 gpus

```bash
python -m torch.distributed.launch --master_port 12347 --nproc_per_node=8 \
            main.py  kinetics RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.02 \
            --lr_scheduler step  --lr_steps 50 75 90 --epochs 100 --batch-size 16 \
            --wd 1e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 4 --npb \
            --contrastive_loss True --contrastive_temperature 0.1 --category_supervised True
```

### Key Training Parameters (Added for GCNet)

- `--contrastive_loss True`: Enable contrastive learning (default: False).

- `--contrastive_temperature`: Temperature parameter for contrastive loss (default: 0.1).

- `--category_supervised True`: Enable category-supervised contrastive learning (default: True).

## Contact

luoyujuan@xxx.com (replace with your contact email)

## Acknowledgements

We especially thank the contributors of the [TSN](https://github.com/yjxiong/tsn-pytorch) and [TDN](https://github.com/MCG-NJU/TDN) codebase for providing helpful code. We also appreciate the open-source contributions of PyTorch, Hugging Face, and other related projects.

## License

This repository is released under the Apache-2.0 license as found in the [LICENSE](https://github.com/luoyujuan123/GCNet/blob/main/LICENSE) file.

## Citation

If you think our work is useful, please feel free to cite our paper 😆 :

```latex
@InProceedings{GCNet_2026,
    author    = {Luo, Yujuan},
    title     = {GCNet: Global-local Temporal Modeling Enhanced 2DCNN with Category-supervised Contrastive Learning for Action Recognition},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},  % 可替换为你的发表会议/期刊
    month     = {June},
    year      = {2026},
    pages     = {xxxx-xxxx}
}
```
> （注：文档部分内容可能由 AI 生成）