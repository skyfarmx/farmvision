# Farm Vision Training Script - Key additions to original train.py

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

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss, ComputeLossOTA
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

logger = logging.getLogger(__name__)

# Farm Vision specific configurations
FRUIT_CLASSES = {
    'mandalina': {'weight': 0.125, 'color': [255, 165, 0]},
    'elma': {'weight': 0.105, 'color': [255, 0, 0]},
    'armut': {'weight': 0.220, 'color': [0, 255, 0]},
    'seftali': {'weight': 0.185, 'color': [255, 20, 147]},
    'nar': {'weight': 0.300, 'color': [139, 0, 0]},
    'hurma': {'weight': 0.150, 'color': [255, 140, 0]}
}


class FarmVisionMetrics:
    """Track Farm Vision specific metrics during training."""
    
    def __init__(self, class_names):
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """Reset metrics for new epoch."""
        self.batch_counts = defaultdict(int)
        self.epoch_detections = defaultdict(int)
        self.weight_estimates = defaultdict(float)
        
    def update(self, targets, predictions=None):
        """Update metrics with batch data."""
        if len(targets) > 0:
            classes = targets[:, 1].cpu().numpy() if len(targets.shape) > 1 else []
            for cls_id in classes:
                if int(cls_id) < len(self.class_names):
                    fruit_name = self.class_names[int(cls_id)]
                    self.batch_counts[fruit_name] += 1
                    if fruit_name in FRUIT_CLASSES:
                        self.weight_estimates[fruit_name] += FRUIT_CLASSES[fruit_name]['weight']
    
    def get_epoch_summary(self):
        """Get summary of current epoch metrics."""
        total_fruits = sum(self.batch_counts.values())
        total_weight = sum(self.weight_estimates.values())
        
        summary = {
            'total_fruits': total_fruits,
            'total_weight_kg': total_weight,
            'fruit_distribution': dict(self.batch_counts),
            'weight_distribution': dict(self.weight_estimates)
        }
        return summary


def apply_farm_vision_augmentations(hyp):
    """Apply Farm Vision specific augmentations for agricultural data."""
    
    # Agricultural specific augmentations
    farm_vision_hyp = {
        # Lighting conditions (dawn/dusk/midday variations)
        'hsv_h': 0.015,  # Slight hue variation for different lighting
        'hsv_s': 0.7,    # Saturation for different seasons
        'hsv_v': 0.4,    # Value for shadow/sunlight variations
        
        # Geometric augmentations for drone perspective
        'degrees': 15.0,     # Limited rotation (drones are usually stable)
        'translate': 0.1,    # Small translations
        'scale': 0.2,        # Scale variations for different altitudes
        'shear': 2.0,        # Minimal shear
        'perspective': 0.0,  # No perspective (drone images are orthogonal)
        
        # Agricultural specific
        'mosaic': 0.8,       # Good for varied fruit densities
        'mixup': 0.1,        # Limited mixup for agricultural scenes
        'copy_paste': 0.2,   # Useful for fruit augmentation
        
        # Flip probabilities
        'flipud': 0.0,       # No up-down flip (fruits hang down)
        'fliplr': 0.5,       # Left-right flip is natural
        
        # Drone specific
        'altitude_scale': True,     # Scale based on altitude simulation
        'wind_blur': 0.1,          # Simulate wind-induced blur
        'shadow_aug': 0.3,         # Simulate shadow variations
    }
    
    # Update hyperparameters
    hyp.update(farm_vision_hyp)
    return hyp


def log_farm_vision_metrics(fv_metrics, epoch, tb_writer, wandb_logger):
    """Log Farm Vision specific metrics."""
    summary = fv_metrics.get_epoch_summary()
    
    # Log to console
    logger.info(f"Farm Vision Metrics - Epoch {epoch}:")
    logger.info(f"  Total fruits detected: {summary['total_fruits']}")
    logger.info(f"  Estimated weight: {summary['total_weight_kg']:.2f} kg")
    
    # Log fruit distribution
    for fruit, count in summary['fruit_distribution'].items():
        logger.info(f"  {fruit}: {count} fruits")
    
    # Log to TensorBoard
    if tb_writer:
        tb_writer.add_scalar('farm_vision/total_fruits', summary['total_fruits'], epoch)
        tb_writer.add_scalar('farm_vision/total_weight_kg', summary['total_weight_kg'], epoch)
        
        for fruit, count in summary['fruit_distribution'].items():
            tb_writer.add_scalar(f'farm_vision/fruits_{fruit}', count, epoch)
    
    # Log to W&B
    if wandb_logger and wandb_logger.wandb:
        wandb_data = {
            'farm_vision/total_fruits': summary['total_fruits'],
            'farm_vision/total_weight_kg': summary['total_weight_kg']
        }
        
        for fruit, count in summary['fruit_distribution'].items():
            wandb_data[f'farm_vision/fruits_{fruit}'] = count
            
        wandb_logger.log(wandb_data)


def save_farm_vision_checkpoint(ckpt, save_dir, epoch, fruit_metrics):
    """Save checkpoint with Farm Vision specific data."""
    
    # Add Farm Vision metadata
    ckpt['farm_vision'] = {
        'fruit_classes': FRUIT_CLASSES,
        'epoch_metrics': fruit_metrics.get_epoch_summary(),
        'timestamp': time.time(),
        'version': '1.0'
    }
    
    # Save specialized checkpoints
    if epoch % 50 == 0:  # Every 50 epochs
        specialized_path = save_dir / 'weights' / f'farm_vision_epoch_{epoch:03d}.pt'
        torch.save(ckpt, specialized_path)
        logger.info(f"Farm Vision checkpoint saved: {specialized_path}")


def train(hyp, opt, device, tb_writer=None):
    """Enhanced training function with Farm Vision features."""
    
    logger.info(colorstr('Farm Vision Training Started'))
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    
    save_dir, epochs, batch_size, total_batch_size, weights, rank, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank, opt.freeze

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'
    
    # Farm Vision specific directories
    fv_dir = save_dir / 'farm_vision'
    fv_dir.mkdir(exist_ok=True)
    fv_results_file = fv_dir / 'fruit_metrics.txt'

    # Apply Farm Vision augmentations
    if opt.farm_vision_aug:
        hyp = apply_farm_vision_augmentations(hyp)
        logger.info("Applied Farm Vision specific augmentations")

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

    # Logging setup (W&B, etc.)
    loggers = {'wandb': None}
    if rank in [-1, 0]:
        opt.hyp = hyp
        run_id = torch.load(weights, map_location=device).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict

    nc = 1 if opt.single_cls else int(data_dict['nc'])
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']
    
    # Initialize Farm Vision metrics
    fv_metrics = FarmVisionMetrics(names)
    
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

    # Freeze layers if specified
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            logger.info(f'Freezing {k}')
            v.requires_grad = False

    # Optimizer setup
    nbs = 64
    accumulate = max(round(nbs / total_batch_size), 1)
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    # Parameter groups
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)

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

    # Resume training logic
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

    # DataParallel
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
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

    # Model parameters
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
    compute_loss_ota = ComputeLossOTA(model)
    compute_loss = ComputeLoss(model)
    
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting Farm Vision training for {epochs} epochs...')

    # Main training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        fv_metrics.reset()  # Reset Farm Vision metrics for new epoch

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

            # Update Farm Vision metrics
            fv_metrics.update(targets)

            # Warmup
            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale training
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward pass
            with amp.autocast(enabled=cuda):
                pred = model(imgs)
                if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
                    loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)
                else:
                    loss, loss_items = compute_loss(pred, targets.to(device))
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

        # Validation
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

            # Log Farm Vision metrics
            log_farm_vision_metrics(fv_metrics, epoch, tb_writer, wandb_logger)

            # Write results
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')

            # Write Farm Vision results
            fv_summary = fv_metrics.get_epoch_summary()
            with open(fv_results_file, 'a') as f:
                f.write(f"Epoch {epoch}: {fv_summary}\n")

            # Log metrics
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

            # Save model
            if (not opt.nosave) or (final_epoch and not opt.evolve):
                ckpt = {'epoch': epoch,
                       'best_fitness': best_fitness,
                       'training_results': results_file.read_text(),
                       'model': deepcopy(model.module if is_parallel(model) else model).half(),
                       'ema': deepcopy(ema.ema).half(),
                       'updates': ema.updates,
                       'optimizer': optimizer.state_dict(),
                       'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

                # Save with Farm Vision metadata
                save_farm_vision_checkpoint(ckpt, save_dir, epoch, fv_metrics)

                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                    
                del ckpt

    # End training
    if rank in [-1, 0]:
        logger.info(f'{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        
        # Final model stripping
        final = best if best.exists() else last
        for f in last, best:
            if f.exists():
                strip_optimizer(f)
                
        logger.info(f"Farm Vision training completed. Final model: {final}")

    return results


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
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=50, first3=0 1 2')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    
    # Farm Vision specific arguments
    parser.add_argument('--farm-vision-aug', action='store_true', help='use Farm Vision specific augmentations')
    parser.add_argument('--fruit-classes', type=str, default='', help='path to fruit classes configuration')
    
    opt = parser.parse_args()

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)

    # Farm Vision specific setup
    if opt.fruit_classes and Path(opt.fruit_classes).exists():
        with open(opt.fruit_classes) as f:
            custom_fruits = yaml.load(f, Loader=yaml.SafeLoader)
            FRUIT_CLASSES.update(custom_fruits)
        logger.info(f"Loaded custom fruit classes: {opt.fruit_classes}")

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
        logger.info(f'Resuming Farm Vision training from {ckpt}')
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
    logger.info(f"Farm Vision Training Configuration: {opt}")
    if not opt.evolve:
        tb_writer = None
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(opt.save_dir)
        train(hyp, opt, device, tb_writer)

    # Hyperparameter evolution for Farm Vision
    else:
        # Farm Vision specific hyperparameter ranges
        meta = {
            'lr0': (1, 1e-5, 1e-1),
            'lrf': (1, 0.01, 1.0),
            'momentum': (0.3, 0.6, 0.98),
            'weight_decay': (1, 0.0, 0.001),
            'warmup_epochs': (1, 0.0, 5.0),
            'warmup_momentum': (1, 0.0, 0.95),
            'warmup_bias_lr': (1, 0.0, 0.2),
            'box': (1, 0.02, 0.2),
            'cls': (1, 0.2, 4.0),
            'cls_pw': (1, 0.5, 2.0),
            'obj': (1, 0.2, 4.0),
            'obj_pw': (1, 0.5, 2.0),
            'iou_t': (0, 0.1, 0.7),
            'anchor_t': (1, 2.0, 8.0),
            'anchors': (2, 2.0, 10.0),
            'fl_gamma': (0, 0.0, 2.0),
            
            # Farm Vision specific parameters
            'hsv_h': (1, 0.0, 0.05),        # Conservative hue for fruits
            'hsv_s': (1, 0.0, 0.9),         # Saturation for seasonal variation
            'hsv_v': (1, 0.0, 0.6),         # Value for lighting conditions
            'degrees': (1, 0.0, 20.0),      # Limited rotation for drone stability
            'translate': (1, 0.0, 0.2),     # Small translations
            'scale': (1, 0.0, 0.3),         # Scale for different altitudes
            'shear': (1, 0.0, 5.0),         # Minimal shear
            'perspective': (0, 0.0, 0.001), # No perspective for orthogonal drone views
            'flipud': (0, 0.0, 0.1),        # Minimal up-down flip
            'fliplr': (0, 0.0, 1.0),        # Natural left-right flip
            'mosaic': (1, 0.0, 1.0),        # Good for varied fruit densities
            'mixup': (1, 0.0, 0.3),         # Limited mixup for agricultural scenes
            'copy_paste': (1, 0.0, 0.5),    # Useful for fruit augmentation
        }

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)
            if 'anchors' not in hyp:
                hyp['anchors'] = 3

        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True
        yaml_file = Path(opt.save_dir) / 'hyp_evolved_farm_vision.yaml'
        
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.txt .')

        logger.info("Starting Farm Vision hyperparameter evolution...")
        best_fitness = 0.0
        
        for generation in range(300):  # generations to evolve
            if Path('evolve.txt').exists():
                # Select parent(s)
                parent = 'single'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))
                x = x[np.argsort(-fitness(x))][:n]
                w = fitness(x) - fitness(x).min()
                
                if parent == 'single' or len(x) == 1:
                    x = x[random.choices(range(n), weights=w)[0]]
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()

                # Mutate with Farm Vision considerations
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])
                ng = len(meta)
                v = np.ones(ng)
                
                while all(v == 1):
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                
                for i, k in enumerate(hyp.keys()):
                    hyp[k] = float(x[i + 7] * v[i])

            # Constrain to Farm Vision limits
            for k, v in meta.items():
                if k in hyp:
                    hyp[k] = max(hyp[k], v[1])  # lower limit
                    hyp[k] = min(hyp[k], v[2])  # upper limit
                    hyp[k] = round(hyp[k], 5)   # significant digits

            # Apply Farm Vision augmentations
            if opt.farm_vision_aug:
                hyp = apply_farm_vision_augmentations(hyp)

            # Train mutation
            logger.info(f"Evolution generation {generation + 1}/300")
            results = train(hyp.copy(), opt, device)
            
            # Track best fitness
            current_fitness = fitness(np.array(results).reshape(1, -1))[0]
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                logger.info(f"New best fitness: {best_fitness:.4f} at generation {generation + 1}")

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot evolution results
        plot_evolution(yaml_file)
        logger.info(f'Farm Vision hyperparameter evolution complete. Best results saved as: {yaml_file}')
        logger.info(f'Command to train with optimized hyperparameters: python train.py --hyp {yaml_file} --farm-vision-aug')


def create_farm_vision_config():
    """Create a sample Farm Vision configuration file."""
    config = {
        'fruit_classes': FRUIT_CLASSES,
        'augmentation_settings': {
            'agricultural_lighting': True,
            'drone_perspective': True,
            'seasonal_variation': True,
            'altitude_simulation': True
        },
        'training_settings': {
            'preferred_batch_size': 16,
            'recommended_epochs': 300,
            'learning_rate_range': [1e-4, 1e-2],
            'image_sizes': [640, 800, 1024]
        },
        'model_settings': {
            'anchor_optimization': True,
            'class_balancing': True,
            'fruit_specific_nms': True
        }
    }
    
    config_path = 'farm_vision_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    logger.info(f"Sample Farm Vision configuration created: {config_path}")
    return config_path


if __name__ == '__main__':
    # Create sample configuration if requested
    if '--create-config' in sys.argv:
        create_farm_vision_config()
        sys.exit(0)
    
    # Standard training execution
    main()