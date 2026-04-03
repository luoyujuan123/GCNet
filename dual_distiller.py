
# dual_distiller.py
"""
双教师知识蒸馏器：ViViT-FE（时序教师） + CLIP（特征教师） → TDN（学生）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from timm.models.vision_transformer import vit_base_patch16_224
import torchvision.transforms as transforms
import os
import glob


class FeatureDistiller:
    """特征蒸馏器（CLIP → TDN）"""
    
    def __init__(self, student, clip_teacher, temperature=3.0, alpha=0.3):
        self.student = student
        self.clip_teacher = clip_teacher
        self.temperature = temperature
        self.alpha = alpha
        
        # 冻结CLIP教师
        for param in self.clip_teacher.parameters():
            param.requires_grad = False
        
        # CLIP预处理
        self.clip_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                               (0.26862954, 0.26130258, 0.27577711))
        ])
    
    def extract_clip_features(self, frames):
        """提取单帧CLIP特征"""
        # frames: [B*T, C, H, W] (TDN只使用中间帧)
        frames_resized = self.clip_transform(frames)
        with torch.no_grad():
            clip_features = self.clip_teacher.get_image_features(frames_resized)
        return clip_features
    
    def compute_feature_loss(self, student_features, clip_features):
        """计算特征蒸馏损失"""
        # student_features: [B*T, 2048]
        # clip_features: [B*T, 768]
        
        # 对齐维度：将学生特征从2048降到768
        if student_features.size(1) != clip_features.size(1):
            # 使用线性变换降维
            student_features = F.adaptive_avg_pool1d(
                student_features.unsqueeze(1), 
                clip_features.size(1)
            ).squeeze(1)
        
        # 归一化
        student_norm = F.normalize(student_features, p=2, dim=1)
        clip_norm = F.normalize(clip_features, p=2, dim=1)
        
        # 余弦相似度损失
        feature_loss = 1 - F.cosine_similarity(student_norm, clip_norm).mean()
        
        return feature_loss * self.alpha


class TemporalDistiller:
    """时序蒸馏器（ViViT-FE → TDN）"""
    
    def __init__(self, student, vivit_teacher, temperature=3.0, beta=0.3):
        self.student = student
        self.vivit_teacher = vivit_teacher
        self.temperature = temperature
        self.beta = beta
        
        # 冻结ViViT教师
        for param in self.vivit_teacher.parameters():
            param.requires_grad = False
        
        # ViViT预处理
        self.vivit_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def extract_vivit_features(self, video_clips):
        """提取视频片段的ViViT时序特征"""
        # video_clips: [B, T, C, H, W] (每个片段取中间帧)
        B, T, C, H, W = video_clips.shape
        video_clips_resized = self.vivit_transform(
            video_clips.view(-1, C, H, W)
        ).view(B, T, C, 224, 224)
        
        with torch.no_grad():
            # 使用ViViT提取时序特征
            vivit_features = []
            for t in range(T):
                frame_features = self.vivit_teacher.forward_features(
                    video_clips_resized[:, t]
                )
                # 提取[CLS] token或全局特征
                if frame_features.dim() == 3:  # [B, N, D]
                    # 取[CLS] token (索引0)
                    frame_features = frame_features[:, 0]  # [B, D]
                elif frame_features.dim() == 2:  # [B, D]
                    # 已经是全局特征
                    pass
                vivit_features.append(frame_features)
            vivit_features = torch.stack(vivit_features, dim=1)  # [B, T, D]
        
        return vivit_features
    
    def compute_temporal_loss(self, student_features, vivit_features):
        """计算时序蒸馏损失"""
        # student_features: [B, T, 2048] (从TDN获取)
        # vivit_features: [B, T, 768] (从ViViT获取)
        
        # print(f"Debug compute_temporal_loss - student_features shape: {student_features.shape}")
        # print(f"Debug compute_temporal_loss - vivit_features shape: {vivit_features.shape}")
        
        # 对齐维度
        if student_features.size(2) != vivit_features.size(2):
            #print(f"Aligning dimensions: {student_features.size(2)} -> {vivit_features.size(2)}")
            # 方法1: 使用自适应平均池化对齐维度
            # 首先将特征重塑为 [B*T, C] 格式
            B, T, C_student = student_features.shape
            student_features_flat = student_features.view(-1, C_student)  # [B*T, 2048]
            
            # 使用自适应池化降维
            student_features_flat = student_features_flat.unsqueeze(1)  # [B*T, 1, 2048]
            student_features_flat = F.adaptive_avg_pool1d(student_features_flat, vivit_features.size(2))  # [B*T, 1, 768]
            student_features_flat = student_features_flat.squeeze(1)  # [B*T, 768]
            
            # 重塑回 [B, T, 768]
            student_features = student_features_flat.view(B, T, -1)
        
        #print(f"Debug compute_temporal_loss - student_features aligned shape: {student_features.shape}")
        
        # 计算时序关系蒸馏损失
        # 对时间维度取平均，得到 [B, D]
        student_aggregated = student_features.mean(dim=1)  # [B, 768]
        vivit_aggregated = vivit_features.mean(dim=1)      # [B, 768]
        
        # print(f"Debug compute_temporal_loss - student_aggregated shape: {student_aggregated.shape}")
        # print(f"Debug compute_temporal_loss - vivit_aggregated shape: {vivit_aggregated.shape}")
        
        # 使用MSE损失
        temporal_loss = F.mse_loss(student_aggregated, vivit_aggregated)
        
        #print(f"Debug compute_temporal_loss - loss: {temporal_loss.item()}")
        
        return temporal_loss * self.beta


class TSNFeatureExtractor(nn.Module):
    """TDN/TSN特征提取器包装类"""
    
    def __init__(self, tsn_model):
        super().__init__()
        self.tsn_model = tsn_model
        self.num_segments = tsn_model.num_segments
        self.new_length = tsn_model.new_length
        self.modality = tsn_model.modality
        
        # 注册钩子来提取中间特征
        self.frame_features = None
        
        # 在ResNet的layer4之后注册钩子
        self._register_hooks()
    
    def _register_hooks(self):
        """注册钩子来提取特征"""
        # 清空之前的钩子
        self.frame_features = None
        
        # 在layer4_bak之后注册钩子提取帧级特征
        if hasattr(self.tsn_model.base_model, 'layer4_bak'):
            def frame_feature_hook(module, input, output):
                # output: [B*T, 2048, 7, 7] 
                # 对于TDN，每个片段只处理一个帧（中间帧），所以是 B*T
                self.frame_features = output
            
            self.tsn_model.base_model.layer4_bak.register_forward_hook(frame_feature_hook)
    
    def forward_with_features(self, x):
        """前向传播并返回特征"""
        # 清空之前保存的特征
        self.frame_features = None
        
        # 标准前向传播
        sample_len = 3 * self.new_length
        base_out = self.tsn_model.base_model(
            x.view((-1, sample_len * 5) + x.size()[-2:])
        )
        
        if self.tsn_model.dropout > 0:
            base_out = self.tsn_model.new_fc(base_out)
        
        # 重塑输出
        base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
        output = self.tsn_model.consensus(base_out)
        output = output.squeeze(1)
        
        # 准备特征字典
        features = {}
        
        # 帧级特征
        if self.frame_features is not None:
            # [B*T, 2048, 7, 7] -> [B*T, 2048]
            # 使用全局平均池化获取帧级特征
            frame_feat = F.adaptive_avg_pool2d(self.frame_features, 1)
            frame_feat = frame_feat.view(frame_feat.size(0), -1)
            features['frame_features'] = frame_feat
            
            # 计算时序特征（从帧级特征重塑）
            # 当前 frame_feat 是 [B*T, 2048]
            B = frame_feat.size(0) // self.num_segments
            temporal_feat = frame_feat.view(B, self.num_segments, -1)
            features['temporal_features'] = temporal_feat
            
            # 打印调试信息
            # print(f"Debug TSNFeatureExtractor - frame_features shape: {self.frame_features.shape}")
            # print(f"Debug TSNFeatureExtractor - frame_feat shape after pooling: {frame_feat.shape}")
            # print(f"Debug TSNFeatureExtractor - temporal_feat shape: {temporal_feat.shape}")
            # print(f"Debug TSNFeatureExtractor - B: {B}, T: {self.num_segments}")
        
        return output, features
    
    def forward(self, x):
        """普通前向传播"""
        return self.tsn_model(x)
    
    # 添加属性访问以兼容原始TSN模型
    @property
    def base_model(self):
        """提供对原始base_model的访问"""
        return self.tsn_model.base_model
    
    @property
    def new_fc(self):
        """提供对原始new_fc的访问"""
        return self.tsn_model.new_fc
    
    @property
    def consensus(self):
        """提供对原始consensus的访问"""
        return self.tsn_model.consensus
    
    @property
    def dropout(self):
        """提供对原始dropout的访问"""
        return self.tsn_model.dropout
    
    # 添加方法访问以兼容原始TSN模型
    def get_optim_policies(self):
        """提供对原始get_optim_policies方法的访问"""
        return self.tsn_model.get_optim_policies()
    
    def partialBN(self, mode):
        """提供对原始partialBN方法的访问"""
        return self.tsn_model.partialBN(mode)
    
    def get_augmentation(self, flip=True):
        """提供对原始get_augmentation方法的访问"""
        return self.tsn_model.get_augmentation(flip)
    
    @property
    def crop_size(self):
        """提供对crop_size的访问"""
        return self.tsn_model.crop_size
    
    @property
    def scale_size(self):
        """提供对scale_size的访问"""
        return self.tsn_model.scale_size
    
    @property
    def input_mean(self):
        """提供对input_mean的访问"""
        return self.tsn_model.input_mean
    
    @property
    def input_std(self):
        """提供对input_std的访问"""
        return self.tsn_model.input_std
    
    @property
    def _enable_pbn(self):
        """提供对_enable_pbn的访问"""
        return self.tsn_model._enable_pbn
    
    @_enable_pbn.setter
    def _enable_pbn(self, value):
        """设置_enable_pbn"""
        self.tsn_model._enable_pbn = value


class DualTeacherDistiller(nn.Module):
    """双教师蒸馏器主类"""
    
    def __init__(self, student_model, num_classes=51, 
                 clip_model_path='/home/lj/wanaihua/GCNet-all/GCNet/TDNCode/TDN-main/kd_pretrained_models/clip-vit-base-patch32',
                 vivit_model_path='/home/lj/wanaihua/GCNet-all/GCNet/TDNCode/TDN-main/kd_pretrained_models/vit_base_patch16_224',
                 temperature=3.0, alpha=0.3, beta=0.3):
        super().__init__()
        
        # 包装学生模型以提取特征
        self.student = TSNFeatureExtractor(student_model)
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        
        # 检查模型路径是否存在
        if not os.path.exists(clip_model_path):
            raise FileNotFoundError(f"CLIP模型路径不存在: {clip_model_path}")
        if not os.path.exists(vivit_model_path):
            raise FileNotFoundError(f"ViViT模型路径不存在: {vivit_model_path}")
        
        print(f"从本地加载CLIP模型: {clip_model_path}")
        print(f"从本地加载ViViT模型: {vivit_model_path}")
        
        # 从本地路径加载CLIP模型
        self.clip_teacher = CLIPModel.from_pretrained(clip_model_path)
        
        # 从本地路径加载ViT模型
        # 首先检查是文件还是目录
        if os.path.isfile(vivit_model_path):
            # 如果是文件，直接加载
            self._load_vit_from_file(vivit_model_path)
        elif os.path.isdir(vivit_model_path):
            # 如果是目录，寻找模型文件
            self._load_vit_from_directory(vivit_model_path)
        else:
            raise ValueError(f"ViViT模型路径既不是文件也不是目录: {vivit_model_path}")
        
        # 设置教师模型为评估模式
        self.clip_teacher.eval()
        self.vivit_teacher.eval()
        
        # 初始化蒸馏器
        self.feature_distiller = FeatureDistiller(
            self.student.tsn_model, self.clip_teacher, temperature, alpha
        )
        self.temporal_distiller = TemporalDistiller(
            self.student.tsn_model, self.vivit_teacher, temperature, beta
        )
        
    def _load_vit_from_file(self, file_path):
        """从文件加载ViT模型"""
        print(f"从文件加载ViT模型: {file_path}")
        self.vivit_teacher = vit_base_patch16_224(pretrained=False)
        checkpoint = torch.load(file_path, map_location='cpu')
        
        # 根据checkpoint格式调整
        state_dict = checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        
        # 加载权重
        missing_keys, unexpected_keys = self.vivit_teacher.load_state_dict(state_dict, strict=False)
        print(f"ViT模型加载 - 缺失的键: {len(missing_keys)}, 意外的键: {len(unexpected_keys)}")
        
        if len(missing_keys) > 0:
            print("缺失的键:", missing_keys[:5])
        
    def _load_vit_from_directory(self, dir_path):
        """从目录加载ViT模型"""
        print(f"从目录加载ViT模型: {dir_path}")
        
        # 查找可能的模型文件
        possible_files = [
            os.path.join(dir_path, 'pytorch_model.bin'),
            os.path.join(dir_path, 'model.pth'),
            os.path.join(dir_path, 'checkpoint.pth'),
            os.path.join(dir_path, 'best.pth'),
            os.path.join(dir_path, 'last.pth'),
        ]
        
        # 查找所有.pth文件
        pth_files = glob.glob(os.path.join(dir_path, '*.pth'))
        tar_files = glob.glob(os.path.join(dir_path, '*.pth.tar'))
        bin_files = glob.glob(os.path.join(dir_path, '*.bin'))
        
        all_files = possible_files + pth_files + tar_files + bin_files
        
        # 找到存在的文件
        existing_files = [f for f in all_files if os.path.exists(f)]
        
        if not existing_files:
            # 如果没有找到文件，尝试使用timm预训练模型
            print("未找到模型文件，尝试使用timm预训练模型...")
            try:
                self.vivit_teacher = vit_base_patch16_224(pretrained=True)
                print("成功加载timm预训练的ViT模型")
                return
            except:
                raise FileNotFoundError(f"在目录 {dir_path} 中找不到模型文件")
        
        # 使用第一个找到的文件
        model_file = existing_files[0]
        print(f"找到模型文件: {model_file}")
        
        # 尝试从文件加载
        try:
            self._load_vit_from_file(model_file)
        except Exception as e:
            print(f"从文件加载失败: {e}")
            # 如果失败，使用timm预训练模型
            print("尝试使用timm预训练模型...")
            self.vivit_teacher = vit_base_patch16_224(pretrained=True)
    
    def forward(self, x, target=None, extract_features=False):
        """
        Args:
            x: 输入视频 [B, T*5*3, H, W]
            target: 真实标签
            extract_features: 是否提取中间特征
        """
        B = x.size(0)
        T = self.student.num_segments
        
        if extract_features:
            # 使用包装器提取特征
            student_output, student_features = self.student.forward_with_features(x)
        else:
            # 普通前向传播
            student_output = self.student(x)
        
        if target is not None:
            # 计算硬标签损失
            hard_loss = F.cross_entropy(student_output, target)
            
            if extract_features:
                try:
                    # 对于TDN，每个视频片段只使用中间帧
                    # 输入x的形状是 [B, T*5*3, H, W]
                    # 我们需要提取每个片段的中间帧（第3帧）
                    
                    # 重塑为 [B, T, 5, 3, H, W]
                    video_frames = x.view(B, T, 5, 3, x.size(-2), x.size(-1))
                    
                    # 提取每个片段的中间帧（索引2，因为0-based）
                    middle_frames = video_frames[:, :, 2, :, :, :]  # [B, T, 3, H, W]
                    
                    # 重塑为 [B*T, 3, H, W] 用于CLIP
                    frames_for_clip = middle_frames.contiguous().view(B * T, 3, x.size(-2), x.size(-1))
                    
                    # 提取CLIP特征
                    clip_features = self.feature_distiller.extract_clip_features(frames_for_clip)
                    
                    # 提取ViViT特征
                    # 将中间帧重塑为 [B, T, 3, H, W]
                    frames_for_vivit = middle_frames
                    vivit_features = self.temporal_distiller.extract_vivit_features(frames_for_vivit)
                    
                    # 打印特征维度用于调试
                    # print(f"Debug DualTeacherDistiller - clip_features shape: {clip_features.shape}")
                    # print(f"Debug DualTeacherDistiller - vivit_features shape: {vivit_features.shape}")
                    
                    # 计算蒸馏损失
                    feature_loss = 0
                    temporal_loss = 0
                    
                    if 'frame_features' in student_features and student_features['frame_features'] is not None:
                        #print(f"Debug DualTeacherDistiller - student frame_features shape: {student_features['frame_features'].shape}")
                        feature_loss = self.feature_distiller.compute_feature_loss(
                            student_features['frame_features'], clip_features
                        )
                    
                    if 'temporal_features' in student_features and student_features['temporal_features'] is not None:
                        #print(f"Debug DualTeacherDistiller - student temporal_features shape: {student_features['temporal_features'].shape}")
                        temporal_loss = self.temporal_distiller.compute_temporal_loss(
                            student_features['temporal_features'], vivit_features
                        )
                    
                    # 总损失
                    total_loss = hard_loss + feature_loss + temporal_loss
                    #total_loss = hard_loss + feature_loss
                    #total_loss = hard_loss + temporal_loss
                    
                    #print(f"Debug DualTeacherDistiller - Losses: hard={hard_loss.item():.4f}, feature={feature_loss.item():.4f}, temporal={temporal_loss.item():.4f}, total={total_loss.item():.4f}")
                    
                    losses = {
                        'total': total_loss,
                        'hard': hard_loss,
                        'feature': feature_loss,
                        'temporal': temporal_loss
                    }
                    return student_output, losses
                    
                except Exception as e:
                    print(f"Error in distillation forward: {e}")
                    import traceback
                    traceback.print_exc()
                    # 如果出错，只返回硬标签损失
                    return student_output, {'total': hard_loss, 'hard': hard_loss}
            else:
                return student_output, {'total': hard_loss, 'hard': hard_loss}
        
        return student_output
    
    # 添加属性访问以兼容原始模型接口
    @property
    def base_model(self):
        """提供对原始base_model的访问"""
        return self.student.base_model
    
    @property
    def new_fc(self):
        """提供对原始new_fc的访问"""
        return self.student.new_fc
    
    @property
    def consensus(self):
        """提供对原始consensus的访问"""
        return self.student.consensus
    
    @property
    def dropout(self):
        """提供对原始dropout的访问"""
        return self.student.dropout
    
    @property
    def num_segments(self):
        """提供对num_segments的访问"""
        return self.student.num_segments
    
    @property
    def modality(self):
        """提供对modality的访问"""
        return self.student.modality
    
    # 添加方法访问以兼容原始模型接口
    def get_optim_policies(self):
        """提供对原始get_optim_policies方法的访问"""
        return self.student.get_optim_policies()
    
    def partialBN(self, mode):
        """提供对原始partialBN方法的访问"""
        return self.student.partialBN(mode)
    
    def get_augmentation(self, flip=True):
        """提供对原始get_augmentation方法的访问"""
        return self.student.get_augmentation(flip)
    
    @property
    def crop_size(self):
        """提供对crop_size的访问"""
        return self.student.crop_size
    
    @property
    def scale_size(self):
        """提供对scale_size的访问"""
        return self.student.scale_size
    
    @property
    def input_mean(self):
        """提供对input_mean的访问"""
        return self.student.input_mean
    
    @property
    def input_std(self):
        """提供对input_std的访问"""
        return self.student.input_std