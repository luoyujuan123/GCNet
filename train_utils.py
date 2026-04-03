# train_utils.py
"""
训练工具函数
"""

import torch
import torch.nn as nn


def freeze_layers(model, freeze_until='layer4'):
    """
    冻结指定层之前的层
    
    Args:
        model: TDN模型
        freeze_until: 冻结直到哪一层 ('conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'none')
    """
    if freeze_until == 'none':
        return
    
    # 根据模型结构冻结
    layers_to_freeze = []
    
    if freeze_until == 'conv1':
        layers_to_freeze = ['conv1', 'bn1', 'relu']
    elif freeze_until == 'layer1':
        layers_to_freeze = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1_bak']
    elif freeze_until == 'layer2':
        layers_to_freeze = ['conv1', 'bn1', 'relu', 'maxpool', 
                           'layer1_bak', 'layer2_bak']
    elif freeze_until == 'layer3':
        layers_to_freeze = ['conv1', 'bn1', 'relu', 'maxpool',
                           'layer1_bak', 'layer2_bak', 'layer3_bak']
    elif freeze_until == 'layer4':
        layers_to_freeze = ['conv1', 'bn1', 'relu', 'maxpool',
                           'layer1_bak', 'layer2_bak', 'layer3_bak', 'layer4_bak']
    
    # 冻结层
    for name, param in model.named_parameters():
        for layer_name in layers_to_freeze:
            if layer_name in name:
                param.requires_grad = False
                break


def setup_distillation(student_model, freeze_backbone=True, train_fc=True):
    """
    设置蒸馏训练
    
    Args:
        student_model: 学生模型
        freeze_backbone: 是否冻结backbone
        train_fc: 是否训练全连接层
    """
    # 设置参数是否需要梯度
    for name, param in student_model.named_parameters():
        if 'new_fc' in name or 'fc' in name:
            param.requires_grad = train_fc
        elif freeze_backbone and 'base_model' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in student_model.parameters())
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    return student_model


def get_optimizer_params(model, lr, weight_decay, finetune_fc_lr=5.0):
    """
    获取优化器参数（不同层不同学习率）
    """
    # 默认策略
    params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 全连接层使用更高的学习率
        if 'new_fc' in name or 'fc' in name:
            params.append({
                'params': param,
                'lr': lr * finetune_fc_lr,
                'weight_decay': weight_decay
            })
        else:
            params.append({
                'params': param,
                'lr': lr,
                'weight_decay': weight_decay
            })
    
    return params


def calculate_model_size(model):
    """计算模型大小"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb