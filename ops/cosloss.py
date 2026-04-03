import torch
import torch.nn.functional as F
from torch import nn
# class ContrastiveLoss(nn.Module):
#     def __init__(self, batch_size, device='cuda', temperature=0.5):
#         super().__init__()
#         self.batch_size = batch_size
#         self.register_buffer("temperature", torch.tensor(temperature).to(device))			# 超参数 温度
#         self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())		# 主对角线为0，其余位置全为1的mask矩阵
        
#     def forward(self, emb_i, emb_j):		# emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
#         z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
#         z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

#         representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
#         similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
#         sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
#         sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
#         positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
#         nominator = torch.exp(positives / self.temperature)             # 2*bs
#         denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
#         loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
#         loss = torch.sum(loss_partial) / (2 * self.batch_size)
#         return loss
# closs=ContrastiveLoss(32)
# x=torch.randn(32,2048).cuda()
# y=torch.randn(32,2048).cuda()
# loss=closs(x,x)
# print(loss)

# import torch
# import torch.nn as nn
# from torch.nn import CosineEmbeddingLoss
# import numpy as np


# def cosine_similarity(x, y):
#     num = x.dot(y.T)
#     denom = np.linalg.norm(x) * np.linalg.norm(y)
#     return num / denom


# def cal_score(score, target):
#     if target == 1:
#         return 1 - score
#     else:
#         return max(0, score)


# def criterion_my(x1, x2, target, reduction='mean'):
#     batch_size, hidden_size = x1.size()
#     scores = torch.cosine_similarity(x1, x2)
#     for i in range(batch_size):
#         scores[i] = cal_score(scores[i], target[i].item())
#     if reduction == 'mean':
#         return scores.mean()
#     elif reduction == 'sum':
#         return scores.sum()


# def criterion_my2(x1, x2, target, reduction='mean'):
#     batch_size, hidden_size = x1.size()
#     scores = torch.zeros(batch_size)
#     for i in range(batch_size):
#         score = cosine_similarity(x1[i], x2[i])
#         scores[i] = cal_score(score, target[i].item())
#     if reduction == 'mean':
#         return scores.mean()
#     elif reduction == 'sum':
#         return scores.sum()


# if __name__ == '__main__':
#     A = torch.tensor([[1.0617, 1.3397, -0.2303],
#                       [0.3459, -0.9821, 1.2511]])

#     B = torch.tensor([[-1.3730, 0.0183, -1.2268],
#                       [0.4486, -0.6504, 1.5173]])
#     Tar = torch.tensor([1, -1])

#     criterion = nn.CosineEmbeddingLoss()
#     score = criterion(A, B, Tar)

#     score_my = criterion_my(A, B, Tar)
#     score_my2 = criterion_my2(A, B, Tar)

#     print(score)
#     print(score_my)
#     print(score_my2)  # tensor(1.1646)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

class DSCLLoss(nn.Module):
    def __init__(self, temperature=1.0, base_temperature=None, K=128, weighted_beta=8.0):
        super(DSCLLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.weighted_beta = weighted_beta

    def forward(self, features, labels=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        ss = features.shape[0]
        batch_size = (features.shape[0] - self.K) // 2

        labels = labels.contiguous().view(-1, 1) # 2BKx1 ) 

        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        # logits = anchor_dot_contrast
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask

        class_weighted = torch.ones_like(mask) / (mask.sum(dim=1, keepdim=True) - 1.0 + 1e-12) * self.weighted_beta
        class_weighted = class_weighted.scatter(1, torch.arange(batch_size).view(-1, 1).to(device) + batch_size, 1.0)

        # compute mean of log-likelihood over positive
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (mask * log_prob * class_weighted).sum(1) / (mask * class_weighted).sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * (mean_log_prob_pos)
        loss_contrastive = loss.mean()

        return loss_contrastive

# import torch
# import pdb
# def min_max_normalize(tensor):
#     min_val = torch.min(tensor)
#     max_val = torch.max(tensor)
#     pdb.set_trace()
#     normalized_tensor = (tensor - min_val) / (max_val - min_val)
#     return normalized_tensor

# # 示例用法
# tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# normalized_tensor = min_max_normalize(tensor)
# print(normalized_tensor)

#三元组对比学习
# class TripletLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super(TripletLoss, self).__init__()
#         self.margin = margin

#     def forward(self, anchor, positive, negative):
#         pos_dist = nn.functional.pairwise_distance(anchor, positive)
#         neg_dist = nn.functional.pairwise_distance(anchor, negative)
#         loss = torch.mean(torch.relu(pos_dist - neg_dist + self.margin))
#         return loss

# anchor = torch.randn(10, 128)
# positive = torch.randn(10, 128)
# negative = torch.randn(10, 128)

# criterion = TripletLoss()
# loss = criterion(anchor, positive, negative)
# print(loss)