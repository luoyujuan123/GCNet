#CUDA_VISIBLE_DEVICES=0 python test_models_center_crop.py something \
#--archs='resnet50' --weights '/raid5/liujiayu/TDN-main/checkpoints/best.pth.tar'  --test_segments=8  \
#--test_crops=1 --batch_size=16  --gpus 0 --output_dir '/raid5/liujiayu/TDN-main/output' -j 4 --clip_index=0
#python pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir '/raid5/liujiayu/TDN-main/output'

# CUDA_VISIBLE_DEVICES=0 python test_models_three_crops.py   ucf101 \
# --archs='resnet50' --weights '/raid5/liujiayu/TDN-main/checkpoint/TDN__ucf101_RGB_resnet50_avg_segment1_e5_Kinetics/best.pth.tar'  --test_segments=1 \
# --test_crops=3 --batch_size=16 --full_res --gpus 0 --output_dir '/raid5/liujiayu/TDN-main/output'  \
# -j 4 --clip_index 3

# python test_models_center_crop.py ucf101 --archs='resnet50' --weights /media/sdc/liujiayu/TDN-main/checkpoint/TDN__ucf101_RGB_resnet50_avg_segment8_e50_yuantdn/best.pth.tar --test_segments=8  --test_crops=1 --batch_size=8  --gpus 0 --output_dir /media/sdc/liujiayu/TDN-main/log/TDN__ucf101_RGB_resnet50_avg_segment8_e50_yuantdn -j 4 --clip_index=0

# python pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir /media/sdc/liujiayu/TDN-main/log/TDN__ucf101_RGB_resnet50_avg_segment8_e50_yuantdn

#hmdb51 16
#CUDA_VISIBLE_DEVICES=0,1 python test_models_center_crop.py hmdb51 --archs 'resnet50' --weights /home/lj/wanaihua/GCNet-all/GCNet/testData/tdn_hmdb51/TDN__hmdb51_RGB_resnet50_avg_segment16_e50_mamba_con5_59.2/TDN__hmdb51_RGB_resnet50_avg_segment16_e50_mamba_con5/best.pth.tar  --test_segments 16  --test_crops 1 --batch_size 6  --gpus 0 --output_dir /home/lj/wanaihua/GCNet-all/GCNet/testData/tdn_hmdb51/TDN__hmdb51_RGB_resnet50_avg_segment16_e50_mamba_con5_59.2/TDN__hmdb51_RGB_resnet50_avg_segment16_e50_mamba_con5 -j 0 --clip_index 0

#CUDA_VISIBLE_DEVICES=0,1 python pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir /home/lj/wanaihua/GCNet-all/GCNet/testData/tdn_hmdb51/TDN__hmdb51_RGB_resnet50_avg_segment16_e50_mamba_con5_59.2/TDN__hmdb51_RGB_resnet50_avg_segment16_e50_mamba_con5

#hmdb51 8
#CUDA_VISIBLE_DEVICES=0,1 python test_models_center_crop.py hmdb51 --archs 'resnet50' --weights /home/lj/wanaihua/GCNet-all/GCNet/TDNCode/TDN-main/checkpoint/TDN__hmdb51_RGB_resnet50_avg_segment8_e50_CCL_Mamba_8_42/best.pth.tar  --test_segments 8  --test_crops 1 --batch_size 4  --gpus 0 --output_dir /home/lj/wanaihua/GCNet-all/GCNet/TDNCode/TDN-main/checkpoint/TDN__hmdb51_RGB_resnet50_avg_segment8_e50_CCL_Mamba_8_42 -j 0 --clip_index 0

#CUDA_VISIBLE_DEVICES=0,1 python pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir /home/lj/wanaihua/GCNet-all/GCNet/TDNCode/TDN-main/checkpoint/TDN__hmdb51_RGB_resnet50_avg_segment8_e50_CCL_Mamba_8_42


# CUDA_VISIBLE_DEVICES=1 python test_models_center_crop.py ucf101 --archs 'resnet50' --weights /media/hd0/liujiayu/code/TDN-main/checkpoint/TDN__ucf101_RGB_resnet50_avg_segment8_e50_GCNet1/best.pth.tar  --test_segments 16  --test_crops 1 --batch_size 8  --gpus 0 --output_dir /media/hd0/liujiayu/code/TDN-main/checkpoint/TDN__ucf101_RGB_resnet50_avg_segment8_e50_GCNet1 -j 4 --clip_index 0

# CUDA_VISIBLE_DEVICES=1 python pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir /media/hd0/liujiayu/code/TDN-main/checkpoint/TDN__something_RGB_resnet50_avg_segment16_e80_mamba_con_se2

#ssv1 16
CUDA_VISIBLE_DEVICES=0,1 python test_models_center_crop.py something --archs 'resnet50' --weights /root/TDN-main/testData/TDN__something_RGB_resnet50_avg_segment16_e60_GCNet1m_54.2/best.pth.tar  --test_segments 16  --test_crops 1 --batch_size 6  --gpus 0 --output_dir /root/TDN-main/testData/TDN__something_RGB_resnet50_avg_segment16_e60_GCNet1m_54.2 -j 4 --clip_index 0

CUDA_VISIBLE_DEVICES=0,1 python pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir /root/TDN-main/testData/TDN__something_RGB_resnet50_avg_segment16_e60_GCNet1m_54.2
