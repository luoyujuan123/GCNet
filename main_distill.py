# main_distill.py
"""
双教师知识蒸馏主训练脚本
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel

from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops.logger import setup_logger
from ops.lr_scheduler import get_scheduler
from ops.utils import reduce_tensor, AverageMeter, accuracy
from opts import parser
from ops import dataset_config
from tensorboardX import SummaryWriter

from dual_distiller import DualTeacherDistiller
from train_utils import freeze_layers, setup_distillation

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    
    # # 添加蒸馏相关参数
    # args.distill = True
    # args.temperature = 3.0
    # args.feature_weight = 0.3
    # args.temporal_weight = 0.3
    
    # 分布式初始化
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    
    # 数据集配置
    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(
        args.dataset, args.modality
    )
    
    # 存储名称
    full_arch_name = args.arch
    args.store_name = '_'.join([
        'TDN_Distill', args.dataset, args.modality, full_arch_name,
        args.consensus_type, f'segment{args.num_segments}',
        f'e{args.epochs}', f'temp{args.temperature}',
        f'f{args.feature_weight}_t{args.temporal_weight}'
    ])
    if args.pretrain != 'imagenet':
        args.store_name += f'_{args.pretrain}'
    
    if dist.get_rank() == 0:
        check_rootfolders()
    
    logger = setup_logger(
        output=os.path.join(args.root_log, args.store_name),
        distributed_rank=dist.get_rank(),
        name='TDN_Distill'
    )
    logger.info('storing name: ' + args.store_name)
    
    # 1. 创建学生模型（TDN）
    model = TSN(
        num_class,
        args.num_segments,
        args.modality,
        base_model=args.arch,
        consensus_type=args.consensus_type,
        dropout=args.dropout,
        img_feature_dim=args.img_feature_dim,
        partial_bn=not args.no_partialbn,
        pretrain=args.pretrain,
        fc_lr5=(args.tune_from and args.dataset in args.tune_from)
    )
    
    # 2. 加载预训练权重（修复版本）
    pretrain_path = '/media/sdc/liujiayu/TDN-main/checkpoint/TDN__ucf101_RGB_resnet50_avg_segment8_e50_yuantdn/best.pth.tar'
    if os.path.exists(pretrain_path):
        logger.info(f"=> loading pretrained model from '{pretrain_path}'")
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        
        # 方法1：处理 state_dict 中的键名
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 移除 'module.' 前缀（如果存在）
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                # 移除 'module.' 前缀
                new_key = k[7:]
            else:
                new_key = k
            new_state_dict[new_key] = v
        
        # 加载权重
        load_result = model.load_state_dict(new_state_dict, strict=False)
    
    # 3. 创建双教师蒸馏器
    distiller = DualTeacherDistiller(
        student_model=model,
        num_classes=num_class,
        temperature=args.temperature,
        alpha=args.feature_weight,
        beta=args.temporal_weight
    ).cuda()
    
    # 4. 冻结指定层（根据需求调整）
    if args.freeze_backbone:
        freeze_layers(distiller.student.base_model, freeze_until=args.freeze_until)
        logger.info("=> frozen backbone layers")
    
    # 5. 配置优化策略
    policies = distiller.student.get_optim_policies()
    
    # 6. 数据预处理
    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    
    train_augmentation = model.get_augmentation(
        flip=False if 'something' in args.dataset else True
    )
    
    cudnn.benchmark = True
    
    # 7. 数据加载
    normalize = GroupNormalize(input_mean, input_std)
    
    train_dataset = TSNDataSet(
        args.dataset,
        args.root_path,
        args.train_list,
        num_segments=args.num_segments,
        modality=args.modality,
        image_tmpl=prefix,
        transform=torchvision.transforms.Compose([
            train_augmentation,
            Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
            normalize,
        ]),
        dense_sample=args.dense_sample
    )
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    
    val_dataset = TSNDataSet(
        args.dataset,
        args.root_path,
        args.val_list,
        num_segments=args.num_segments,
        modality=args.modality,
        image_tmpl=prefix,
        random_shift=False,
        transform=torchvision.transforms.Compose([
            GroupScale(int(scale_size)),
            GroupCenterCrop(crop_size),
            Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
            normalize,
        ]),
        dense_sample=args.dense_sample
    )
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=True
    )
    
    # 8. 损失函数和优化器
    criterion = nn.CrossEntropyLoss().cuda()
    
    optimizer = torch.optim.SGD(
        policies,
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    scheduler = get_scheduler(optimizer, len(train_loader), args)
    
    # 9. 分布式训练
    model = DistributedDataParallel(
        distiller.cuda(),
        device_ids=[args.local_rank],
        broadcast_buffers=True,
        find_unused_parameters=True
    )
    
    # 10. 训练循环
    if args.evaluate:
        validate(val_loader, model, criterion, logger)
        return
    
    for epoch in range(args.start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        
        # 训练一个epoch
        train_losses, train_top1, train_top5 = train(
            train_loader, model, criterion, optimizer, 
            epoch=epoch, logger=logger, scheduler=scheduler
        )
        
        # 记录训练指标
        if dist.get_rank() == 0:
            tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
            tf_writer.add_scalar('loss/train_total', train_losses['total'], epoch)
            tf_writer.add_scalar('loss/train_hard', train_losses['hard'], epoch)
            tf_writer.add_scalar('loss/train_feature', train_losses.get('feature', 0), epoch)
            tf_writer.add_scalar('loss/train_temporal', train_losses.get('temporal', 0), epoch)
            tf_writer.add_scalar('acc/train_top1', train_top1, epoch)
            tf_writer.add_scalar('acc/train_top5', train_top5, epoch)
            tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)
        
        # 验证
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            val_loader.sampler.set_epoch(epoch)
            prec1, prec5, val_loss = validate(val_loader, model, criterion, logger)
            
            if dist.get_rank() == 0:
                tf_writer.add_scalar('loss/test', val_loss, epoch)
                tf_writer.add_scalar('acc/test_top1', prec1, epoch)
                tf_writer.add_scalar('acc/test_top5', prec5, epoch)
                
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                
                # 保存checkpoint
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.module.student.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'prec1': prec1,
                        'best_prec1': best_prec1,
                    },
                    epoch + 1,
                    is_best
                )


def train(train_loader, model, criterion, optimizer, epoch, logger=None, scheduler=None):
    """训练函数"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = {'total': AverageMeter(), 'hard': AverageMeter(), 
              'feature': AverageMeter(), 'temporal': AverageMeter()}
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # 设置部分BN
    if args.no_partialbn:
        model.module.student.partialBN(False)
    else:
        model.module.student.partialBN(True)
    
    model.train()
    
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        target = target.cuda()
        input_var = input.cuda()
        
        # 前向传播（带蒸馏）
        output, loss_dict = model(input_var, target=target, extract_features=True)
        
        # 计算准确率
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        
        # 更新损失记录
        losses['total'].update(loss_dict['total'].item(), input.size(0))
        losses['hard'].update(loss_dict['hard'].item(), input.size(0))
        if 'feature' in loss_dict:
            losses['feature'].update(loss_dict['feature'].item(), input.size(0))
        if 'temporal' in loss_dict:
            losses['temporal'].update(loss_dict['temporal'].item(), input.size(0))
        
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        
        # 反向传播
        optimizer.zero_grad()
        loss_dict['total'].backward()
        
        if args.clip_gradient is not None:
            clip_grad_norm_(model.parameters(), args.clip_gradient)
        
        optimizer.step()
        scheduler.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        # 打印日志
        if i % args.print_freq == 0:
            logger.info(
                f'Epoch: [{epoch}][{i}/{len(train_loader)}], '
                f'lr: {optimizer.param_groups[-1]["lr"]:.5f}, '
                f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                f'Data: {data_time.val:.3f} ({data_time.avg:.3f}), '
                f'Loss: {losses["total"].val:.4f} ({losses["total"].avg:.4f}), '
                f'Hard: {losses["hard"].val:.4f}, '
                f'Feature: {losses.get("feature", AverageMeter()).val:.4f}, '
                f'Temporal: {losses.get("temporal", AverageMeter()).val:.4f}, '
                f'Prec@1: {top1.val:.3f} ({top1.avg:.3f}), '
                f'Prec@5: {top5.val:.3f} ({top5.avg:.3f})'
            )
    
    return {k: v.avg for k, v in losses.items()}, top1.avg, top5.avg


def validate(val_loader, model, criterion, logger=None):
    """验证函数"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    model.eval()
    
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            
            # 只使用学生模型进行验证
            output = model.module.student(input_var)
            loss = criterion(output, target)
            
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            
            # 分布式平均
            loss = reduce_tensor(loss)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
            
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % args.print_freq == 0:
                logger.info(
                    f'Test: [{i}/{len(val_loader)}], '
                    f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                    f'Loss: {losses.val:.4f} ({losses.avg:.4f}), '
                    f'Prec@1: {top1.val:.3f} ({top1.avg:.3f}), '
                    f'Prec@5: {top5.val:.3f} ({top5.avg:.3f})'
                )
    
    logger.info(
        f'Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {losses.avg:.5f}'
    )
    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, epoch, is_best):
    """保存checkpoint"""
    filename = f'{args.root_model}/{args.store_name}/{epoch}_epoch_ckpt.pth.tar'
    torch.save(state, filename)
    if is_best:
        best_filename = f'{args.root_model}/{args.store_name}/best.pth.tar'
        torch.save(state, best_filename)


def check_rootfolders():
    """创建日志和模型文件夹"""
    folders_util = [
        args.root_log, args.root_model,
        os.path.join(args.root_log, args.store_name),
        os.path.join(args.root_model, args.store_name)
    ]
    for folder in folders_util:
        if not os.path.exists(folder):
            print(f'creating folder {folder}')
            os.makedirs(folder)


if __name__ == '__main__':
    main()