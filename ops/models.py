# Code for "TDN: Temporal Difference Networks for Efficient Action Recognition"
# arXiv: 2012.10071
# Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
# tongzhan@smail.nju.edu.cn

from torch import nn

from ops.basic_ops import ConsensusModule
from ops.transforms import *
from torch.nn.init import normal_, constant_
from ops.tdn_net_mamba import tdn_net
from ops.con_se import conblock
from mamba_ssm import Mamba
import torch.nn.functional as F

class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8, img_feature_dim=256,crop_num=1,
                 partial_bn=True, print_spec=True, pretrain='imagenet',fc_lr5=False):
        super(TSN, self).__init__()

        self.bash_model=Mamba(d_model=6272, # Model dimension d_model
        d_state=16,  # SSM state expansion factor
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
        )
        self.cb=conblock(hidden_dim=128)


        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        self.pretrain = pretrain
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5  # fine_tuning for UCF/HMDB
        self.target_transforms = {86:87,87:86,93:94,94:93,166:167,167:166}

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        if print_spec:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout, self.img_feature_dim)))

        self._prepare_base_model(self.cb,self.bash_model,base_model, self.num_segments)#
        #self._prepare_base_model(self.bash_model,base_model, self.num_segments)#
        feature_dim = self._prepare_tsn(num_class)
        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)
        
        return feature_dim

    def _prepare_base_model(self, cb,bash_model,base_model, num_segments):#
    #def _prepare_base_model(self,bash_model,base_model, num_segments):#
        print(('=> base model: {}'.format(base_model)))
        if 'resnet' in base_model :
            self.base_model = tdn_net(base_model, num_segments,bash_model,cb)#
            #self.base_model = tdn_net(base_model, num_segments,bash_model)#
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
        else :
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []
        inorm = []
        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            # elif len(m._modules) == 0:
            #     if len(list(m.parameters())) > 0:
            #         raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        if self.fc_lr5: # fine_tuning for UCF/HMDB
            return [
                {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
                'name': "first_conv_weight"},
                {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
                'name': "first_conv_bias"},
                {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
                'name': "normal_weight"},
                {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
                'name': "normal_bias"},
                {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
                'name': "BN scale/shift"},
                {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
                'name': "custom_ops"},
                {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
                'name': "lr5_weight"},
                {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
                'name': "lr10_bias"},
            ]
        else : # default 
            return [
                {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
                'name': "first_conv_weight"},
                {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
                'name': "first_conv_bias"},
                {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
                'name': "normal_weight"},
                {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
                'name': "normal_bias"},
                {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
                'name': "BN scale/shift"},
                {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
                'name': "custom_ops"},
            ]
    def comprehensive_contrastive_loss(self,out1, target1, out2, target2, out_f1, out_f2, t1, t2, temperature=0.07):
        """
        综合对比损失函数
        
        Args:
            out1, target1: 正样本组1及其标签
            out2, target2: 正样本组2及其标签
            out_f1: 浅层次负样本（batch维度打乱）
            out_f2: 深层次负样本（segment交换）
            t1, t2: 标签（与out_f1, out_f2对应的原始标签）
            temperature: 温度参数
        """
        # 对特征进行平均池化
        def pool_and_norm(x):
            return F.normalize(x.mean(dim=1), p=2, dim=1)
        
        out1_norm = pool_and_norm(out1)
        out2_norm = pool_and_norm(out2)
        out_f1_norm = pool_and_norm(out_f1)
        out_f2_norm = pool_and_norm(out_f2)
        
        batch_size = out1_norm.size(0)
        
        # 1. out1 vs out2: 正样本对比
        # 创建标签掩码
        target1_exp = target1.unsqueeze(1).expand(-1, batch_size)
        target2_exp = target2.unsqueeze(0).expand(batch_size, -1)
        
        same_class_mask = (target1_exp == target2_exp).float()
        diff_class_mask = 1.0 - same_class_mask
        
        # 计算相似度矩阵
        sim_out1_out2 = torch.mm(out1_norm, out2_norm.t()) / temperature
        
        # 相同类别：拉近（最小化距离）-> 计算相似损失
        same_class_loss = (same_class_mask * (1 - sim_out1_out2)).sum() / (same_class_mask.sum() + 1e-8)
        
        # 不同类别：拉远（最大化距离）-> 计算不相似损失
        diff_class_loss = (diff_class_mask * torch.relu(sim_out1_out2)).sum() / (diff_class_mask.sum() + 1e-8)
        
        loss_out1_out2 = same_class_loss + diff_class_loss
        
        # 2. out1 vs out_f1: 正样本对（out1有效，out_f1无效，应该拉远）
        sim_out1_f1 = torch.mm(out1_norm, out_f1_norm.t()) / temperature
        # 应该拉远，所以最小化相似度
        loss_out1_f1 = torch.mean(torch.relu(sim_out1_f1))
        
        # 3. out1 vs out_f2: 计算不相同损失（非同一批样本变化）
        sim_out1_f2 = torch.mm(out1_norm, out_f2_norm.t()) / temperature
        # 非同一批样本，应该不相似
        loss_out1_f2 = torch.mean(torch.relu(sim_out1_f2))
        
        # 4. out2 vs out_f1: 拉近不相似损失（非同一批样本变化）
        sim_out2_f1 = torch.mm(out2_norm, out_f1_norm.t()) / temperature
        # 非同一批样本，应该不相似
        loss_out2_f1 = torch.mean(torch.relu(sim_out2_f1))
        
        # 5. out2 vs out_f2: 拉近不相同损失（最重要的语义边界划分证明组）
        sim_out2_f2 = torch.mm(out2_norm, out_f2_norm.t()) / temperature
        # out_f2是out2的轻微变化，应该有一定相似性但又不完全相同
        # 这里使用一个温和的损失，既不完全拉近也不完全拉远
        target_similarity = 0.3  # 期望的相似度阈值
        loss_out2_f2 = torch.mean(torch.abs(sim_out2_f2 - target_similarity))
        
        # 6. out_f1 vs out_f2: 无效样本对比
        sim_f1_f2 = torch.mm(out_f1_norm, out_f2_norm.t()) / temperature
        # 都是无效样本，如果被判定为相同，增大惩罚力度
        # 使用动态温度调整
        avg_sim = sim_f1_f2.mean()
        dynamic_temp = temperature * (1 + avg_sim)  # 相似度越高，温度越高，惩罚越大
        loss_f1_f2 = torch.mean(torch.relu(sim_f1_f2)) * dynamic_temp
        
        # 总损失
        total_loss = torch.mean(
            0.1*loss_out1_out2 +
            0.2*loss_out1_f1 +
            0.2*loss_out1_f2 +
            0.2*loss_out2_f1 +
            0.2*loss_out2_f2 +
            0.1*loss_f1_f2
        )
        
        return total_loss/100
    # def contrastive_classification_loss(self,f1, f2, l1, l2, margin=1.0):
    #     """
    #     计算对比学习的损失函数，结合分类任务。
        
    #     :param features1: 第一个特征向量组, 形状为 [batch_size, 8, 2048]
    #     :param features2: 第二个特征向量组, 形状为 [batch_size, 8, 2048]
    #     :param labels1: 第一个特征向量组的标签, 形状为 [batch_size]
    #     :param labels2: 第二个特征向量组的标签, 形状为 [batch_size]
    #     :param margin: 不相似样本对的最小距离
    #     :return: 损失值
    #     """
    #     # 计算批次大小
    #     #pdb.set_trace()
    #     batch_size = f1.size()[0]
        
    #     # 对特征向量进行归一化
    #     f1 = F.normalize(f1, p=2, dim=2)
    #     f2 = F.normalize(f2, p=2, dim=2)
        
    #     # 计算余弦距离
    #     cosine_similarity = torch.sum(f1 * f2, dim=2)  # [batch_size, 8]
    #     distances = 1 - cosine_similarity  # 余弦距离
        
    #     # 标签相似性
    #     labels_similarity = (l1.unsqueeze(1) == l2.unsqueeze(1)).float()
        
    #     # 相似样本对的损失
    #     loss_similar = labels_similarity * torch.pow(distances, 2)
        
    #     # 不相似样本对的损失
    #     loss_dissimilar = (1 - labels_similarity) * torch.pow(torch.clamp(margin - distances, min=0.0), 2)
        
    #     # 总损失
    #     loss = torch.mean(loss_similar + loss_dissimilar)
        
    #     return loss


    def forward(self, input, target=None, no_reshape=False):
    #def forward(self, input,no_reshape=False):
        if not no_reshape:


            sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
            # import pdb
            # pdb.set_trace()
            base_out = self.base_model(input.view((-1, sample_len*5) + input.size()[-2:]))

            #2026/2/28

            if target is not None:
                base_out_c=base_out.view((target.size()[0],-1)+base_out.size()[1:])#[batch_size,segment,2048]
                #pdb.set_trace()
                out=torch.chunk(base_out_c,chunks=2,dim=0)#[batch_size/2,segment,2048]
                targets=torch.chunk(target,chunks=2,dim=0)#[batch_size/2]

                # 复制一份用于打乱标签
                out_f = base_out_c.clone()  # [batch_size, segment, 2048]
                
                # 打乱out_f的标签（在batch维度打乱）
                batch_size = out_f.size(0)
                label_shuffle_idx = torch.randperm(batch_size).to(out_f.device)
                out_f = out_f[label_shuffle_idx, :, :]  # 标签打乱后的特征
                
                # 将原始特征和打乱标签后的特征都分成两个子集
                out = torch.chunk(base_out_c, chunks=2, dim=0)  # [batch_size/2, segment, 2048]
                out_f = torch.chunk(out_f, chunks=2, dim=0)     # [batch_size/2, segment, 2048]
                targets = torch.chunk(target, chunks=2, dim=0)  # [batch_size/2]
                
                out1 = out[0]  # 正样本组1 [batch_size/2, segment, 2048]
                out2 = out[1]  # 正样本组2 [batch_size/2, segment, 2048]
                out_f1 = out_f[0]  # 浅层次负样本组 [batch_size/2, segment, 2048]
                out_f2 = out_f[1]  # 深层次负样本组 [batch_size/2, segment, 2048]
                target1 = targets[0]  # out1的标签
                target2 = targets[1]  # out2的标签
                
                # 对out_f1进行batch维度打乱（浅层次负样本）
                batch_size_half = out_f1.size(0)
                batch_shuffle_idx = torch.randperm(batch_size_half).to(out_f1.device)
                out_f1_shuffled = out_f1[batch_shuffle_idx, :, :]  # batch维度打乱
                
                # 对out_f2进行segment维度任意交换一对（深层次负样本）
                seq_len = out_f2.size(1)
                # 随机选择两个不同的segment位置
                if seq_len > 1:
                    idx1 = torch.randint(0, seq_len, (1,)).item()
                    idx2 = torch.randint(0, seq_len, (1,)).item()
                    while idx2 == idx1:
                        idx2 = torch.randint(0, seq_len, (1,)).item()
                    
                    # 交换这两个segment的特征
                    out_f2_swapped = out_f2.clone()
                    out_f2_swapped[:, idx1, :] = out_f2[:, idx2, :]
                    out_f2_swapped[:, idx2, :] = out_f2[:, idx1, :]
                else:
                    out_f2_swapped = out_f2  # 如果只有一个segment，则不交换
                
                # 计算对比损失
                loss = self.comprehensive_contrastive_loss(
                    out1, target1,           # 正样本组1
                    out2, target2,           # 正样本组2
                    out_f1_shuffled,         # 浅层次负样本（batch维度打乱）
                    out_f2_swapped,          # 深层次负样本（segment交换）
                    target1, target2,        # 标签
                    temperature=0.08
                )
           

            #2026/2/28

        else:
            base_out = self.base_model(input)

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)

        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            output = self.consensus(base_out)

        if target is None:
            return output.squeeze(1)
        return loss,output.squeeze(1)


    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if flip:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip(is_flow=False)])
        else:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                            GroupRandomHorizontalFlip_sth(self.target_transforms)])

