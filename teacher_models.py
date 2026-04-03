# teacher_models.py
"""
教师模型定义和适配器
"""

import torch
import torch.nn as nn
from transformers import CLIPModel
from timm.models.vision_transformer import vit_base_patch16_224


class CLIPFeatureExtractor(nn.Module):
    """CLIP特征提取器适配器"""
    
    def __init__(self, clip_model_name='openai/clip-vit-base-patch32'):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        
        # 冻结所有参数
        for param in self.clip.parameters():
            param.requires_grad = False
        
    def forward(self, images):
        """提取图像特征"""
        # images: [B, C, H, W]
        return self.clip.get_image_features(pixel_values=images)


class ViViTFeatureExtractor(nn.Module):
    """ViViT时序特征提取器"""
    
    def __init__(self, pretrained=True):
        super().__init__()
        self.vivit = vit_base_patch16_224(pretrained=pretrained)
        
        # 冻结所有参数
        for param in self.vivit.parameters():
            param.requires_grad = False
        
        # 获取特征维度
        self.feature_dim = self.vivit.embed_dim
        
    def forward(self, frames):
        """提取视频帧特征"""
        # frames: [B, T, C, H, W]
        B, T, C, H, W = frames.shape
        
        # 逐帧处理
        frame_features = []
        for t in range(T):
            features = self.vivit.forward_features(frames[:, t])
            frame_features.append(features)
        
        # [B, T, D]
        return torch.stack(frame_features, dim=1)
    
    def extract_temporal_relations(self, frame_features):
        """提取时序关系"""
        # frame_features: [B, T, D]
        
        # 计算帧间差异作为时序关系
        temporal_relations = []
        for t in range(frame_features.size(1) - 1):
            diff = frame_features[:, t+1] - frame_features[:, t]
            temporal_relations.append(diff)
        
        # [B, T-1, D]
        return torch.stack(temporal_relations, dim=1)