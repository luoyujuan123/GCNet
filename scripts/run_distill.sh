# scripts/run_distill.sh
#!/bin/bash

# 设置参数
DATASET=hmdb51
MODALITY=RGB
ARCH=resnet50
NUM_SEGMENTS=8
EPOCHS=50
BATCH_SIZE=4
LR=0.0015
PRETRAIN=yuantdn_distill

# 教师模型权重
CLIP_MODEL=openai/clip-vit-base-patch32
VIVIT_MODEL=vit_base_patch16_224

# 蒸馏参数
TEMPERATURE=3.0
FEATURE_WEIGHT=0.3
TEMPORAL_WEIGHT=0.3

# 冻结设置
FREEZE_BACKBONE=true
FREEZE_UNTIL=layer4  # 冻结直到layer4
TRAIN_FC=true

# # 运行命令
# python -m torch.distributed.launch \
#     --master_port 12349 \
#     --nproc_per_node=1 \
#     main_distill.py \
#     $DATASET \
#     $MODALITY \
#     --arch $ARCH \
#     --num_segments $NUM_SEGMENTS \
#     --gd 20 \
#     --lr $LR \
#     --lr_steps 25 35 \
#     --epochs $EPOCHS \
#     --batch-size $BATCH_SIZE \
#     --dropout 0.8 \
#     --consensus_type=avg \
#     --eval-freq=1 \
#     -j 4 \
#     --npb \
#     --pretrain $PRETRAIN \
#     --temperature $TEMPERATURE \
#     --feature_weight $FEATURE_WEIGHT \
#     --temporal_weight $TEMPORAL_WEIGHT \
#     --freeze_backbone $FREEZE_BACKBONE \
#     --freeze_until $FREEZE_UNTIL \
#     --train_fc $TRAIN_FC
python -m torch.distributed.launch \
    --master_port 12349 \
    --nproc_per_node=1 \
    main_distill.py \
    something \
    RGB \
    --arch resnet50 \
    --num_segments 8 \
    --gd 20 \
    --lr 0.001 \
    --lr_steps 25 35 \
    --epochs  50\
    --batch-size 4 \
    --dropout 0.8 \
    --consensus_type=avg \
    --eval-freq=1 \
    -j 4 \
    --npb \
    --pretrain yuantdn_distill \
    --distill \
    --temperature 3.5 \
    --feature_weight 0.3 \
    --temporal_weight 0 \
    --freeze_backbone \
    --freeze_until layer4 \
    --train_fc
