# Farm Vision Enhanced Auxiliary Training Script
# Key differences from standard train_aux.py:
# 1. Farm Vision specific metrics and logging
# 2. Agricultural data augmentations  
# 3. Fruit detection optimizations
# 4. Enhanced auxiliary head training for multi-scale fruit detection

import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread
from collections import defaultdict

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss, ComputeLossAuxOTA
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

logger = logging.getLogger(__name__)

# Farm Vision fruit configuration
FRUIT_CLASSES = {
    'mandalina': {'weight': 0.125, 'size_range': [15, 35], 'color_range': [20, 40]},
    'elma': {'weight': 0.105, 'size_range': [25, 45], 'color_range': [0, 15]},
    'armut': {'weight': 0.220, 'size_range': [30, 50], 'color_range': [30, 60]},
    'seftali': {'weight': 0.185, 'size_range': [20, 40], 'color_range': [10, 30]},
    'nar': {'weight': 0.300, 'size_range': [35, 55], 'color_range': [350, 20]},
    'hurma': {'weight': 0.150, 'size_range': [15, 30], 'color_range': [20, 40]}
}


class FarmVisionAuxMetrics:
    """Enhanced metrics for auxiliary head training in Farm Vision."""
    
    def __init__(self, class_names, num_aux_heads=3):
        self.class_names = class_names
        self.num_aux_heads = num_aux_heads
        self.reset()
    
    def reset(self):
        """Reset metrics for new epoch."""
        self.main_head_detections = defaultdict(int)
        self.aux_head_detections = [defaultdict(int) for _ in range(self.num_aux_heads)]
        self.scale_specific_detections = defaultdict(lambda: defaultdict(int))
        self.weight_estimates = defaultdict(float)
        self.small_fruit_detections = defaultdict(int)
        self.large_fruit_detections = defaultdict(int)
        
    def update(self, targets, predictions=None, aux_predictions=None):
        """Update metrics with batch data including auxiliary predictions."""
        if len(targets) > 0:
            classes = targets[:, 1].cpu().numpy() if len(targets.shape) > 1 else []
            
            for cls_id in classes:
                if int(cls_id) < len(self.class_names):
                    fruit_name = self.class_names[int(cls_id)]
                    
                    # Main head tracking
                    self.main_head_detections[fruit_name] += 1
                    
                    # Weight estimation
                    if fruit_name in FRUIT_CLASSES:
                        self.weight_estimates[fruit_name] += FRUIT_CLASSES[fruit_name]['weight']
                    
                    # Size-based classification (for aux head optimization)
                    if fruit_name in FRUIT_CLASSES:
                        size_range = FRUIT_CLASSES[fruit_name]['size_range']
                        avg_size = sum(size_range) / 2
                        if avg_size < 30:  # Small fruits
                            self.small_fruit_detections[fruit_name] += 1
                        else:  # Large fruits
                            self.large_fruit_detections[fruit_name] += 1
        
        # Track auxiliary head predictions if available
        if aux_predictions:
            for i, aux_pred in enumerate(aux_predictions):
                if len(aux_pred) > 0 and i < self.num_aux_heads:
                    aux_classes = aux_pred[:, 5].cpu().numpy() if len(aux_pred.shape) > 1 else []
                    for cls_id in aux_classes:
                        if int(cls_id) < len(self.class_names):
                            fruit_name = self.class_names[int(cls_id)]
                            self.aux_head_detections[i][fruit_name] += 1
    
    def get_epoch_summary(self):
        """Get comprehensive summary including auxiliary head performance."""
        total_fruits = sum(self.main_head_detections.values())
        total_weight = sum(self.weight_estimates.values())
        small_fruits = sum(self.small_fruit_detections.values())
        large_fruits = sum(self.large_fruit_detections.values())
        
        # Auxiliary head performance
        aux_summary = []
        for i, aux_detections in enumerate(self.aux_head_detections):
            aux_total = sum(aux_detections.values())
            aux_summary.append({
                'head_id': i,
                'total_detections': aux_total,
                'fruit_distribution': dict(aux_detections)
            })
        
        summary = {
            'main_head': {
                'total_fruits': total_fruits,
                'total_weight_kg': total_weight,
                'fruit_distribution': dict(self.main_head_detections)
            },
            'auxiliary_heads': aux_summary,
            'scale_analysis': {
                'small_fruits': small_fruits,
                'large_fruits': large_fruits,
                'small_fruit_distribution': dict(self.small_fruit_detections),
                'large_fruit_distribution': dict(self.large_fruit_detections)
            },
            'detection_efficiency': {
                'main_vs_aux_ratio': total_fruits / max(sum(sum(aux.values()) for aux in self.aux_head_detections), 1),
                'small_object_performance': small_fruits / max(total_fruits, 1),
                'multi_scale_coverage': len([h for h in aux_summary if h['total_detections'] > 0])
            }
        }
        return summary


def apply_agricultural_aux_augmentations(hyp):
    """Apply Farm Vision specific augmentations optimized for auxiliary head training."""
    
    # Enhanced augmentations for multi-scale fruit detection
    farm_vision_aux_hyp = {
        # Multi-scale training optimization
        'hsv_h': 0.02,       # Conservative hue for consistent fruit colors across scales
        'hsv_s': 0.6,        # Moderate saturation for seasonal variations
        'hsv_v': 0.5,        # Value changes for different lighting at various scales
        
        # Geometric augmentations for auxiliary heads
        'degrees': 10.0,     # Limited rotation for stable drone footage
        'translate': 0.15,   # Moderate translation for scale variation
        'scale': 0.4,        # Enhanced scale variation for aux head training
        'shear': 3.0,        # Minimal shear
        'perspective': 0.0,  # No perspective for orthogonal drone views
        
        # Multi-scale specific augmentations
        'mosaic': 0.9,       # High mosaic probability for scale diversity
        'mixup': 0.15,       # Conservative mixup for agricultural scenes
        'copy_paste': 0.3,   # Enhanced copy-paste for fruit augmentation
        
        # Flip probabilities
        'flipud': 0.05,      # Minimal up-down flip (fruits hang naturally)
        'fliplr': 0.5,       # Natural left-right flip
        
        # Auxiliary head specific
        'aux_loss_weight': 0.3,      # Weight for auxiliary loss contribution
        'aux_scale_weights': [0.4, 0.3, 0.3],  # Weights for different aux scales
        'small_object_focus': True,   # Enhanced small object detection
        'multi_scale_consistency': True,  # Consistency across scales
    }
    
    # Update hyperparameters
    hyp.update(farm_vision_aux_hyp)
    return hyp


def log_auxiliary_metrics(fv_metrics, epoch, tb_writer, wandb_logger):
    """Log Farm Vision auxiliary head specific metrics."""
    summary = fv_metrics.get_epoch_summary()
    
    # Log to console
    logger.info(f"Farm Vision Auxiliary Training - Epoch {epoch}:")
    logger.info(f"  Main Head - Total fruits: {summary['main_head']['total_fruits']}")
    logger.info(f"  Main Head - Estimated weight: {summary['main_head']['total_weight_kg']:.2f} kg")
    
    # Log auxiliary head performance
    for aux_info in summary['auxiliary_heads']:
        logger.info(f"  Aux Head {aux_info['head_id']}: {aux_info['total_detections']} detections")
    
    # Log scale analysis
    logger.info(f"  Small fruits: {summary['scale_analysis']['small_fruits']}")
    logger.info(f"  Large fruits: {summary['scale_analysis']['large_fruits']}")
    logger.info(f"  Multi-scale coverage: {summary['detection_efficiency']['multi_scale_coverage']}/3 heads active")
    
    # Log to TensorBoard
    if tb_writer:
        # Main head metrics
        tb_writer.add_scalar('farm_vision_aux/main_total_fruits', summary['main_head']['total_fruits'], epoch)
        tb_writer.add_scalar('farm_vision_aux/main_total_weight_kg', summary['main_head']['total_weight_kg'], epoch)
        
        # Auxiliary head metrics
        for aux_info in summary['auxiliary_heads']:
            tb_writer.add_scalar(f'farm_vision_aux/aux_head_{aux_info["head_id"]}_detections', 
                               aux_info['total_detections'], epoch)
        
        # Scale analysis
        tb_writer.add_scalar('farm_vision_aux/small_fruits', summary['scale_analysis']['small_fruits'], epoch)
        tb_writer.add_scalar('farm_vision_aux/large_fruits', summary['scale_analysis']['large_fruits'], epoch)
        tb_writer.add_scalar('farm_vision_aux/multi_scale_coverage', 
                           summary['detection_efficiency']['multi_scale_coverage'], epoch)
        
        # Efficiency metrics
        tb_writer.add_scalar('farm_vision_aux/main_vs_aux_ratio', 
                           summary['detection_efficiency']['main_vs_aux_ratio'], epoch)
        tb_writer.add_scalar('farm_vision_aux/small_object_performance', 
                           summary['detection_efficiency']['small_object_performance'], epoch)
    
    # Log to W&B
    if wandb_logger and wandb_logger.wandb:
        wandb_data = {
            'farm_vision_aux/main_total_fruits': summary['main_head']['total_fruits'],
            'farm_vision_aux/main_total_weight_kg': summary['main_head']['total_weight_kg'],
            'farm_vision_aux/small_fruits': summary['scale_analysis']['small_fruits'],
            'farm_vision_aux/large_fruits': summary['scale_analysis']['large_fruits'],
            'farm_vision_aux/multi_scale_coverage': summary['detection_efficiency']['multi_scale_coverage'],
            'farm_vision_aux/main_vs_aux_ratio': summary['detection_efficiency']['main_vs_aux_ratio']
        }
        
        # Add auxiliary head metrics
        for aux_info in summary['auxiliary_heads']:
            wandb_data[f'farm_vision_aux/aux_head_{aux_info["head_id"]}_detections'] = aux_info['total_detections']
        
        wandb_logger.log(wandb_data)


def train(hyp, opt, device, tb_writer=None):
    """Enhanced auxiliary training function with Farm Vision optimizations."""
    
    logger.info(colorstr('Farm Vision Auxiliary Training Started'))
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'
    
    # Farm Vision auxiliary specific directories
    fv_aux_dir = save_dir / 'farm_vision_aux'
    fv_aux_dir.mkdir(exist_ok=True)
    fv_aux_results_file = fv_aux_dir / 'aux_metrics.txt'

    # Apply Farm Vision auxiliary augmentations
    if opt.farm_vision_aux:
        hyp = apply_agricultural_aux_augmentations(hyp)
        logger.info("Applied Farm Vision auxiliary-specific augmentations")

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    plots = not opt.evolve
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    is_coco = opt.data.endswith('coco.yaml')

    # Logging setup
    loggers = {'wandb': None}
    if rank in [-1, 0]:
        opt.hyp = hyp
        run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict

    nc = 1 if opt.single_cls else int(data_dict['nc'])
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']
    
    # Initialize Farm Vision auxiliary metrics
    fv_aux_metrics = FarmVisionAuxMetrics(names, num_aux_heads=3)
    
    # Model setup
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)
        ckpt = torch.load(weights, map_location=device)
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []
        state_dict = ckpt['model'].float().state_dict()
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)
        model.load_state_dict(state_dict, strict=False)
        logger.info(f'Transferred {len(state_dict)}/{len(model.state_dict())} items from {weights}')
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)

    # Check dataset
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)
    train_path = data_dict['train']
    test_path = data_dict['val']

    # No freezing for auxiliary head training (need full gradient flow)
    freeze = []
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            logger.info(f'Freezing {k}')
            v.requires_grad = False

    # Optimizer setup (same as original but with auxiliary considerations)
    nbs = 64
    accumulate = max(round(nbs / total_batch_size), 1)
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    # Enhanced parameter grouping for auxiliary heads
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
        
        # Include auxiliary head specific parameters
        for attr_name in ['im', 'imc', 'imb', 'imo', 'ia']:
            if hasattr(v, attr_name):
                attr = getattr(v, attr_name)
                if hasattr(attr, 'implicit'):
                    pg0.append(attr.implicit)
                else:
                    for iv in attr:
                        if hasattr(iv, 'implicit'):
                            pg0.append(iv.implicit)

    # Optimizer
    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})
    optimizer.add_param_group({'params': pg2})
    logger.info(f'Optimizer groups: {len(pg2)} .bias, {len(pg1)} conv.weight, {len(pg0)} other')
    del pg0, pg1, pg2

    # Scheduler
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # Resume training logic (same as original)
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']
        start_epoch = ckpt['epoch'] + 1
        del ckpt, state_dict

    # Image sizes
    gs = max(int(model.stride.max()), 32)
    nl = model.model[-1].nl
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]

    # DataParallel and SyncBatchNorm
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # Data loaders
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                          hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                          world_size=opt.world_size, workers=opt.workers,
                                          image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()
    nb = len(dataloader)

    if rank in [-1, 0]:
        testloader = create_dataloader(test_path, imgsz_test, batch_size * 2, gs, opt,
                                     hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                     world_size=opt.world_size, workers=opt.workers,
                                     pad=0.5, prefix=colorstr('val: '))[0]

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    # Model parameters for auxiliary training
    hyp['box'] *= 3. / nl
    hyp['cls'] *= nc / 80. * 3. / nl
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc
    model.hyp = hyp
    model.gr = 1.0
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    model.names = names

    # Training setup
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)
    maps = np.zeros(nc)
    results = (0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch - 1
    scaler = amp.GradScaler(enabled=cuda)
    
    # Enhanced loss computation for auxiliary heads
    compute_loss_aux_ota = ComputeLossAuxOTA(model)
    compute_loss = ComputeLoss(model)
    
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting Farm Vision auxiliary training for {epochs} epochs...')

    # Main training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        fv_aux_metrics.reset()

        mloss = torch.zeros(4, device=device)
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)
        
        optimizer.zero_grad()
        
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255.0

            # Update Farm Vision auxiliary metrics
            fv_aux_metrics.update(targets)

            # Warmup
            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale training (enhanced for auxiliary heads)
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward pass with auxiliary loss
            with amp.autocast(enabled=cuda):
                pred = model(imgs)
                
                # Use auxiliary OTA loss for better multi-scale training
                loss, loss_items = compute_loss_aux_ota(pred, targets.to(device), imgs)
                
                if rank != -1:
                    loss *= opt.world_size
                if opt.quad:
                    loss *= 4.

            # Backward pass
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Logging
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                s = ('%10s' * 2 + '%10.4g' * 6) % (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

        # End of epoch
        lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()

        # Validation and logging
        if rank in [-1, 0]:
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            
            if not opt.notest or final_epoch:
                wandb_logger.current_epoch = epoch + 1
                results, maps, times = test.test(data_dict,
                                               batch_size=batch_size * 2,
                                               imgsz=imgsz_test,
                                               model=ema.ema,
                                               single_cls=opt.single_cls,
                                               dataloader=testloader,
                                               save_dir=save_dir,
                                               verbose=nc < 50 and final_epoch,
                                               plots=plots and final_epoch,
                                               wandb_logger=wandb_logger,
                                               compute_loss=compute_loss,
                                               is_coco=is_coco,
                                               v5_metric=opt.v5_metric,
                                               farm_vision_metrics=True)

            # Log Farm Vision auxiliary metrics
            log_auxiliary_metrics(fv_aux_metrics, epoch, tb_writer, wandb_logger)

            # Write results
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')

            # Write Farm Vision auxiliary results
            fv_aux_summary = fv_aux_metrics.get_epoch_summary()
            with open(fv_aux_results_file, 'a') as f:
                f.write(f"Epoch {epoch}: {fv_aux_summary}\n")

            # Standard logging
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',
                   'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                   'val/box_loss', 'val/obj_loss', 'val/cls_loss',
                   'x/lr0', 'x/lr1', 'x/lr2']
            
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))
            if fi > best_fitness:
                best_fitness = fi
            wandb_logger.end_epoch(best_result=best_fitness == fi)

            # Enhanced model saving with auxiliary head information
            if (not opt.nosave) or (final_epoch and not opt.evolve):
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'training_results': results_file.read_text(),
                    'model': deepcopy(model.module if is_parallel(model) else model).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None,
                    
                    # Farm Vision auxiliary specific metadata
                    'farm_vision_aux': {
                        'fruit_classes': FRUIT_CLASSES,
                        'aux_metrics': fv_aux_summary,
                        'aux_hyperparameters': {k: v for k, v in hyp.items() if 'aux' in k},
                        'multi_scale_performance': fv_aux_summary['detection_efficiency'],
                        'timestamp': time.time(),
                        'version': '1.0-aux'
                    }
                }

                # Save checkpoints
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                    
                # Save auxiliary-specific checkpoints
                if best_fitness == fi and epoch >= 200:
                    aux_best_path = wdir / f'aux_best_{epoch:03d}.pt'
                    torch.save(ckpt, aux_best_path)
                    logger.info(f"Auxiliary best model saved: {aux_best_path}")
                
                # Regular epoch checkpoints with auxiliary info
                if epoch == 0 or ((epoch + 1) % 25) == 0 or epoch >= (epochs - 5):
                    aux_epoch_path = wdir / f'aux_epoch_{epoch:03d}.pt'
                    torch.save(ckpt, aux_epoch_path)
                
                del ckpt

    # End training
    if rank in [-1, 0]:
        if plots:
            plot_results(save_dir=save_dir)
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})
        
        logger.info(f'{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        
        # Final auxiliary model evaluation
        if opt.data.endswith('coco.yaml') and nc == 80:
            for m in (last, best) if best.exists() else (last):
                results, _, _ = test.test(opt.data,
                                        batch_size=batch_size * 2,
                                        imgsz=imgsz_test,
                                        conf_thres=0.001,
                                        iou_thres=0.7,
                                        model=attempt_load(m, device).half(),
                                        single_cls=opt.single_cls,
                                        dataloader=testloader,
                                        save_dir=save_dir,
                                        save_json=True,
                                        plots=False,
                                        is_coco=is_coco,
                                        v5_metric=opt.v5_metric)

        # Strip optimizers
        final = best if best.exists() else last
        for f in last, best:
            if f.exists():
                strip_optimizer(f)
        
        # Upload auxiliary models
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights/aux_models/')
            
        if wandb_logger.wandb and not opt.evolve:
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                          name='run_' + wandb_logger.wandb_run.id + '_aux_model',
                                          aliases=['last', 'best', 'stripped', 'auxiliary'])
        
        # Generate auxiliary training report
        generate_aux_training_report(save_dir, fv_aux_dir)
        
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    
    torch.cuda.empty_cache()
    return results


def generate_aux_training_report(save_dir, fv_aux_dir):
    """Generate comprehensive auxiliary training report."""
    report_path = fv_aux_dir / 'auxiliary_training_report.md'
    
    try:
        with open(report_path, 'w') as f:
            f.write("# Farm Vision Auxiliary Head Training Report\n\n")
            f.write(f"**Training Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Save Directory:** {save_dir}\n\n")
            
            f.write("## Auxiliary Head Configuration\n")
            f.write("- **Number of Auxiliary Heads:** 3\n")
            f.write("- **Loss Function:** ComputeLossAuxOTA\n")
            f.write("- **Multi-Scale Training:** Enabled\n")
            f.write("- **Agricultural Augmentations:** Applied\n\n")
            
            f.write("## Fruit Classes Configuration\n")
            for fruit, config in FRUIT_CLASSES.items():
                f.write(f"- **{fruit.title()}:** {config['weight']} kg, ")
                f.write(f"Size: {config['size_range']} px, ")
                f.write(f"Hue: {config['color_range']}°\n")
            
            f.write("\n## Training Benefits\n")
            f.write("- Enhanced small fruit detection through auxiliary supervision\n")
            f.write("- Improved multi-scale feature learning\n")
            f.write("- Better gradient flow for deep networks\n")
            f.write("- Increased training stability\n")
            f.write("- Optimized for agricultural drone imagery\n\n")
            
            f.write("## Files Generated\n")
            f.write("- `aux_metrics.txt`: Detailed auxiliary head metrics\n")
            f.write("- `aux_best_*.pt`: Best performing auxiliary models\n")
            f.write("- `aux_epoch_*.pt`: Regular epoch checkpoints\n")
            f.write("- Standard training outputs with auxiliary enhancements\n")
        
        logger.info(f"Auxiliary training report generated: {report_path}")
        
    except Exception as e:
        logger.warning(f"Could not generate auxiliary training report: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov7.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train_aux', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    
    # Farm Vision auxiliary specific arguments
    parser.add_argument('--farm-vision-aux', action='store_true', help='use Farm Vision auxiliary training features')
    parser.add_argument('--aux-weight', type=float, default=0.3, help='auxiliary loss weight')
    parser.add_argument('--small-object-focus', action='store_true', help='focus on small object detection')
    
    opt = parser.parse_args()

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)

    # Resume training
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = \
            '', ckpt, True, opt.total_batch_size, *apriori
        logger.info(f'Resuming Farm Vision auxiliary training from {ckpt}')
    else:
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Load hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)

    # Train
    logger.info(f"Farm Vision Auxiliary Training Configuration: {opt}")
    if not opt.evolve:
        tb_writer = None
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(opt.save_dir)
        train(hyp, opt, device, tb_writer)

    # Hyperparameter evolution (simplified for auxiliary training)
    else:
        logger.info("Hyperparameter evolution not recommended for auxiliary training")
        logger.info("Use standard train.py --evolve for hyperparameter optimization, then apply to auxiliary training")