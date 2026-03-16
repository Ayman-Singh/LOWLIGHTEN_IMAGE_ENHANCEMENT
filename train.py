# train.py — training script for ELIEI
#
# settings from the paper:
#   batch=16, crop=160, 200K iters, lr=5e-4, Adam(0.9, 0.99)
#   multistep LR decay at [50%, 75%, 90%, 95%], gamma=0.5
#   loss = NLL + 0.01 * CAL (color alignment loss)
#   RRDB frozen for first 50% of training, CAL kicks in at step 500
#
# usage: python train.py --config confs/IR-RGB.yaml [--resume]

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')

import gc
import sys
import time
import argparse
import logging
import random
import numpy as np
import traceback
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.model import build_model
from models.loss import ELIEILoss
from data.dataset import build_dataloader
from utils.utils import (
    load_config, setup_logger, save_checkpoint, load_checkpoint,
    MultiStepLR_Restart, CosineAnnealingLR_Custom,
    compute_metrics, save_image, opt_get
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train ELIEI')
    parser.add_argument('--config', type=str, default='confs/IR-RGB.yaml')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--crop_size', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # device setup
    has_cuda = torch.cuda.is_available()
    gpu_ids = cfg.get('gpu_ids', [0])
    device = torch.device(f'cuda:{gpu_ids[0]}' if has_cuda and gpu_ids else 'cpu')

    # on CPU we use gradient accumulation to match the paper batch size
    if not has_cuda:
        paper_batch = cfg['datasets']['train'].get('batch_size', 16)
        accum_steps = paper_batch
        cfg['datasets']['train']['batch_size'] = 1
    else:
        accum_steps = 1

    if args.batch_size:
        cfg['datasets']['train']['batch_size'] = args.batch_size
        accum_steps = 1
    if args.crop_size:
        cfg['datasets']['train']['GT_size'] = args.crop_size

    cfg['datasets']['val']['batch_size'] = 1

    # experiment dirs
    exp_name = cfg.get('name', 'ELIEI')
    exp_dir = os.path.join('experiments', exp_name)
    model_dir = os.path.join(exp_dir, 'models')
    log_dir = os.path.join(exp_dir, 'logs')
    vis_dir = os.path.join(exp_dir, 'vis')
    for d in [model_dir, log_dir, vis_dir]:
        os.makedirs(d, exist_ok=True)

    # logging
    logger = setup_logger('train', log_dir)
    logger.info(f'experiment: {exp_name}')
    logger.info(f'config: {args.config}')
    logger.info(f'device: {device}')
    eff_batch = cfg['datasets']['train']['batch_size'] * accum_steps
    logger.info(f'batch: {cfg["datasets"]["train"]["batch_size"]} x {accum_steps} accum = {eff_batch}')
    logger.info(f'crop: {cfg["datasets"]["train"]["GT_size"]}')

    # seed
    seed = args.seed or opt_get(cfg, ['train', 'manual_seed'], 10)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if has_cuda:
        torch.cuda.manual_seed_all(seed)

    if has_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # build model
    model = build_model(cfg).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_rrdb = sum(p.numel() for p in model.get_encoder_params())
    logger.info(f'params: {n_total:,} total ({n_rrdb:,} RRDB)')

    if not has_cuda:
        model.encoder.use_checkpoint = True
        model.flow.use_checkpoint = True
        logger.info('gradient checkpointing enabled (CPU mode)')

    # loss
    train_cfg = cfg['train']
    criterion = ELIEILoss(
        weight_fl=train_cfg.get('weight_fl', 1.0),
        weight_emd=train_cfg.get('weight_emd', 0.01),
    ).to(device)
    logger.info(f'loss: NLL(w={criterion.weight_fl}) + CAL(w={criterion.weight_emd})')

    # optimizer (paper: Adam, lr=5e-4, betas=(0.9, 0.99))
    lr_G = float(train_cfg.get('lr_G', 5e-4))
    beta1 = train_cfg.get('beta1', 0.9)
    beta2 = train_cfg.get('beta2', 0.99)
    wd = float(train_cfg.get('weight_decay_G', 0))

    optimizer = torch.optim.Adam([
        {'params': model.get_other_params(), 'lr': lr_G, 'weight_decay': wd},
        {'params': model.get_encoder_params(), 'lr': lr_G * 0.1, 'weight_decay': 1e-5},
    ], betas=(beta1, beta2))

    # lr scheduler
    total_iters = train_cfg.get('niter', 200000)
    lr_scheme = train_cfg.get('lr_scheme', 'MultiStepLR')
    lr_steps_rel = train_cfg.get('lr_steps_rel', [0.5, 0.75, 0.9, 0.95])

    if lr_scheme == 'CosineAnnealing':
        min_lr = float(train_cfg.get('min_lr', 1e-6))
        warmup = int(train_cfg.get('warmup_iter', 0))
        scheduler = CosineAnnealingLR_Custom(optimizer, total_iters,
                                              min_lr=min_lr, warmup_iters=warmup)
        logger.info(f'lr: cosine annealing max={lr_G:.1e} min={min_lr:.1e}')
    else:
        scheduler = MultiStepLR_Restart(optimizer, lr_steps_rel, total_iters,
                                        gamma=train_cfg.get('lr_gamma', 0.5))
        logger.info(f'lr: multistep milestones={scheduler.milestones} gamma=0.5')

    # data
    train_loader = build_dataloader(cfg, split='train')
    val_loader = build_dataloader(cfg, split='val')

    # rrdb freeze/unfreeze schedule
    train_rrdb_delay = float(cfg['network_G'].get('train_RRDB_delay', 0.5))
    rrdb_unfreeze_step = int(train_rrdb_delay * total_iters)
    rrdb_unfrozen = cfg['network_G'].get('train_RRDB', True)
    logger.info(f'RRDB unfreeze: {"already" if rrdb_unfrozen else f"step {rrdb_unfreeze_step}"}')

    # resume from checkpoint
    start_step = 0
    best_psnr = 0.0
    if args.resume or opt_get(cfg, ['path', 'resume_state'], None) == 'auto':
        ckpts = sorted(
            [f for f in os.listdir(model_dir)
             if f.endswith('_G.pth') and not f.startswith('best')],
            key=lambda x: int(x.split('_')[0])
        ) if os.path.isdir(model_dir) else []
        if ckpts:
            resume_path = os.path.join(model_dir, ckpts[-1])
            start_step, best_psnr = load_checkpoint(
                resume_path, model, optimizer, None, str(device))
            scheduler.current_step = start_step
            if hasattr(scheduler, '_update_lr'):
                scheduler._update_lr()
            logger.info(f'resumed from {resume_path} step={start_step} best_psnr={best_psnr:.2f}')
            logger.info(f'lr after resume: {scheduler.get_lr():.2e}')

    # tensorboard (optional)
    tb_writer = None
    if cfg.get('use_tb_logger', False):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=os.path.join(exp_dir, 'tb'))
        except Exception:
            pass

    val_freq = train_cfg.get('val_freq', 1000)
    save_freq = int(float(cfg.get('logger', {}).get('save_checkpoint_freq', 5000)))
    print_freq = cfg.get('logger', {}).get('print_freq', 100)
    cal_start = 500  # paper: CAL loss starts at step 500

    logger.info(f'schedule: val@{val_freq}, save@{save_freq}, CAL from step {cal_start}')
    logger.info(f'training: step {start_step} -> {total_iters}')

    # ----- training loop -----
    gc_every = 1 if not has_cuda else 0
    model.train()
    step = start_step
    train_iter = iter(train_loader)

    while step < total_iters:
        # unfreeze RRDB if it's time
        if not rrdb_unfrozen and step >= rrdb_unfreeze_step:
            if model.set_rrdb_training(True):
                rrdb_params = model.get_encoder_params()
                if rrdb_params:
                    optimizer.param_groups[1]['params'] = rrdb_params
                logger.info(f'[step {step}] RRDB unfrozen ({len(rrdb_params)} params)')
            rrdb_unfrozen = True

        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        agg_losses = {}

        try:
            for _micro in range(accum_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                lr_img = batch['lr'].to(device, non_blocking=True)
                hr_img = batch['hr'].to(device, non_blocking=True)

                # forward pass: GT -> latent
                z, logdet = model(hr_img, lr_img, reverse=False)

                # CAL loss: enhance the low-light input and compare histograms
                do_cal = (criterion.weight_emd > 0
                          and step >= cal_start
                          and _micro == 0)

                pred = None
                if do_cal:
                    if not has_cuda:
                        gc.collect()
                    model.eval()
                    with torch.no_grad():
                        pred = model.enhance(lr_img, heat=0.0)
                    model.train()

                total_loss, loss_dict = criterion(z, logdet,
                                                  pred=pred, target=hr_img)
                total_loss = total_loss / accum_steps
                total_loss.backward()

                del z, logdet, total_loss, pred, lr_img, hr_img, batch
                for k, v in loss_dict.items():
                    agg_losses[k] = agg_losses.get(k, 0.0) + v / accum_steps
                del loss_dict

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        except Exception as e:
            logger.error(f'step {step} crashed: {e}')
            traceback.print_exc()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            gc.collect()
            continue

        scheduler.step()
        step += 1
        elapsed = time.time() - t0

        if gc_every and step % gc_every == 0:
            gc.collect()

        # logging
        if step % print_freq == 0:
            lr_now = scheduler.get_lr()
            parts = [f'[{step}/{total_iters}]', f'lr={lr_now:.2e}']
            parts += [f'{k}={v:.4f}' for k, v in agg_losses.items()]
            parts.append(f't={elapsed:.2f}s')
            logger.info('  '.join(parts))
            if tb_writer:
                try:
                    for k, v in agg_losses.items():
                        tb_writer.add_scalar(f'loss/{k}', v, step)
                    tb_writer.add_scalar('lr', lr_now, step)
                except Exception:
                    pass

        # validation
        if step % val_freq == 0:
            try:
                psnr_avg, ssim_avg = validate(model, val_loader, device,
                                              vis_dir, step)
                logger.info(f'[val {step}] PSNR={psnr_avg:.4f}  SSIM={ssim_avg:.4f}')
                if tb_writer:
                    try:
                        tb_writer.add_scalar('val/psnr', psnr_avg, step)
                        tb_writer.add_scalar('val/ssim', ssim_avg, step)
                    except Exception:
                        pass
                is_best = psnr_avg > best_psnr
                if is_best:
                    best_psnr = psnr_avg
                save_checkpoint(model, optimizer, scheduler, step, model_dir,
                                name='G', is_best=is_best, best_psnr=best_psnr)
            except Exception as e:
                logger.error(f'validation failed at step {step}: {e}')
                traceback.print_exc()
            model.train()
            gc.collect()

        # periodic save
        if save_freq > 0 and step % save_freq == 0:
            save_checkpoint(model, optimizer, scheduler, step, model_dir, name='G')

    logger.info(f'done. best PSNR: {best_psnr:.4f}')
    if tb_writer:
        try:
            tb_writer.close()
        except Exception:
            pass


@torch.no_grad()
def validate(model, val_loader, device, vis_dir, step):
    model.eval()
    psnr_list, ssim_list = [], []
    val_vis_dir = os.path.join(vis_dir, f'step_{step:06d}')
    os.makedirs(val_vis_dir, exist_ok=True)

    for idx, batch in enumerate(val_loader):
        lr_img = batch['lr'].to(device)
        hr_img = batch['hr'].to(device)
        enhanced = model.enhance(lr_img, heat=0.0)
        metrics = compute_metrics(enhanced, hr_img)
        psnr_list.append(metrics['psnr'])
        ssim_list.append(metrics['ssim'])
        if idx < 4:
            save_image(enhanced[0], os.path.join(val_vis_dir, f'{idx:04d}_enhanced.png'))
            save_image(hr_img[0], os.path.join(val_vis_dir, f'{idx:04d}_gt.png'))
        del enhanced, lr_img, hr_img, batch

    gc.collect()
    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
