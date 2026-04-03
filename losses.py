# losses.py
"""
蒸馏损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """知识蒸馏损失"""
    
    def __init__(self, temperature=3.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_logits, teacher_logits, labels=None):
        """
        Args:
            student_logits: 学生模型输出
            teacher_logits: 教师模型输出
            labels: 真实标签（可选）
        """
        # 软化教师输出
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # 计算蒸馏损失
        distill_loss = self.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            teacher_probs
        ) * (self.temperature ** 2)
        
        if labels is not None:
            # 计算硬标签损失
            hard_loss = F.cross_entropy(student_logits, labels)
            
            # 组合损失
            total_loss = (1 - self.alpha) * hard_loss + self.alpha * distill_loss
            return total_loss, {'hard': hard_loss, 'distill': distill_loss}
        
        return distill_loss


class FeatureDistillationLoss(nn.Module):
    """特征蒸馏损失"""
    
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, student_features, teacher_features, mode='mse'):
        """
        Args:
            student_features: 学生特征
            teacher_features: 教师特征
            mode: 损失类型 'mse' 或 'cosine'
        """
        if mode == 'mse':
            # MSE损失
            loss = F.mse_loss(student_features, teacher_features)
        elif mode == 'cosine':
            # 余弦相似度损失
            student_norm = F.normalize(student_features, p=2, dim=1)
            teacher_norm = F.normalize(teacher_features, p=2, dim=1)
            loss = 1 - F.cosine_similarity(student_norm, teacher_norm).mean()
        else:
            raise ValueError(f"Unsupported loss mode: {mode}")
        
        return loss


class TemporalConsistencyLoss(nn.Module):
    """时序一致性损失"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, features):
        """
        鼓励相邻帧特征相似
        Args:
            features: [B, T, D] 时序特征
        """
        B, T, D = features.shape
        if T < 2:
            return torch.tensor(0.0, device=features.device)
        
        # 计算相邻帧特征差异
        loss = 0
        for t in range(T - 1):
            diff = F.mse_loss(features[:, t], features[:, t+1])
            loss += diff
        
        return loss / (T - 1)