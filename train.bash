#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --master_port 12347 --nproc_per_node=1 \
#            main.py  ucf101 RGB --arch resnet50 --num_segments 1 --gd 20 --lr 0.02 \
#            --lr_scheduler step  --lr_steps 50 75 90 --epochs 10 --batch-size 4 \
#            --wd 1e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 4 --npb \

# python -m torch.distributed.launch --master_port 12348 --nproc_per_node=1  main.py  ucf101  RGB --arch resnet50 --num_segments 1 --gd 20 --lr 0.02 --lr_scheduler step --lr_steps  50 75 90 --epochs 30 --batch-size 16  --wd 5e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 4 --npb --local_rank=1
# python -m torch.distributed.launch --master_port 12348 --nproc_per_node=1  main.py  kinetics  RGB --arch resnet50 --num_segments 1 --gd 20 --lr 0.02 --lr_scheduler step --lr_steps  50 75 90 --epochs 100 --batch-size 16  --wd 5e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 4 --npb --local_rank=1
# python -m torch.distributed.launch --master_port 12348 --nproc_per_node=1 main.py hmdb51 RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.00015 --lr_steps 25 35 --epochs 50 --batch-size 4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 4 --npb --pretrain none1

# python -m torch.distributed.launch --master_port 12347 --nproc_per_node=1 \
#             main.py  kinetics RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.02 \
#             --lr_scheduler step  --lr_steps 50 75 90 --epochs 100 --batch-size 16 \
#             --wd 1e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 4 --npb

#python -m torch.distributed.launch --master_port 12349 --nproc_per_node=1 main.py ucf101 RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.0015 --lr_steps 25 35 --epochs 50 --batch-size 4 --dropout 0.8 --consensus_type=avg --eval-freq=1 -j 4 --npb --pretrain yuantdn1

# python -m torch.distributed.launch --master_port 12349 --nproc_per_node=2 main.py ucf101 RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.0015 --lr_steps 25 35 --epochs 50 --batch-size 4 --dropout 0.8 --consensus_type=avg --eval-freq=1 -j 4 --npb --pretrain double_se_con6

#python -m torch.distributed.launch --master_port 12348 --nproc_per_node=2 main.py something RGB --arch resnet50 --num_segments 16 --gd 20 --lr 0.004 --lr_steps 30.0 45.0 55.0 --epochs 60 --batch-size 6 --dropout 0.8 --consensus_type=avg --eval-freq=1 -j 4 --npb --pretrain mamba_con_se7

#python -m torch.distributed.launch --master_port 12349 --nproc_per_node=2 main.py ucf101 RGB --arch resnet50 --num_segments 16 --gd 20 --lr 0.0015 --lr_steps 25 35 --epochs 50 --batch-size 6 --dropout 0.8 --consensus_type=avg --eval-freq=1 -j 6 --npb --pretrain GTM1

#python -m torch.distributed.launch --master_port 12348 --nproc_per_node=2 main.py kinetics RGB --arch resnet50 --num_segments 16 --gd 20 --lr 0.02 --lr_steps 30.0 50.0 65.0 --epochs 80 --batch-size 4 --dropout 0.8 --consensus_type=avg --eval-freq=1 -j 4 --npb --pretrain GCNet2


#python -m torch.distributed.launch --master_port 12349 --nproc_per_node=2 main.py hmdb51 RGB --arch resnet50 --num_segments 16 --gd 20 --lr 0.003 --lr_steps 15 35 --epochs 50 --batch-size 6 --dropout 0.8 --consensus_type=avg --eval-freq=1 -j 4 --npb --pretrain GCNet1 --seed 42


#ssv1 8
#python -m torch.distributed.launch --master_port 12349 --nproc_per_node=2 main.py something RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.0025 --lr_steps 30.0 45.0 55.0 --epochs 60 --batch-size 8 --dropout 0.8 --consensus_type=avg --eval-freq=1 -j 8 --npb --pretrain GCNet_1024 --seed 1024

#ssv1 16
python -m torch.distributed.launch --master_port 12349 --nproc_per_node=3 main.py something RGB --arch resnet50 --num_segments 16 --gd 20 --lr 0.0042 --lr_steps 30.0 45.0 55.0 --epochs 60 --batch-size 6 --dropout 0.8 --consensus_type=avg --eval-freq=1 -j 16 --npb --pretrain GCNet_16_1024 --seed 1024


#8zhen CCL+Mamba screen -r mamba_ccl
#python -m torch.distributed.launch --master_port 12348 --nproc_per_node=2 main.py hmdb51 RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.003 --lr_steps 25 35 --epochs 50 --batch-size 4 --dropout 0.8 --consensus_type=avg --eval-freq=1 -j 4 --npb --pretrain CCL_Mamba_8_42 --seed 42

