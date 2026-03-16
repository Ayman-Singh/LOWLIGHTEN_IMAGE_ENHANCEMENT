# utility functions
# metrics (psnr, ssim), logging, lr schedulers, checkpoint handling

import os
import math
import yaml
import logging
import shutil
import torch
import numpy as np
from PIL import Image


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def setup_logger(name, log_dir, level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{name}.log')

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []

    fmt = logging.Formatter(
        '[%(asctime)s][%(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# --- image metrics ---

def tensor_to_numpy(t):
    """(C,H,W) float [0,1] -> (H,W,C) uint8"""
    arr = t.detach().cpu().clamp(0, 1).numpy()
    return (np.transpose(arr, (1, 2, 0)) * 255).astype(np.uint8)


def psnr(pred, target, max_val=1.0):
    mse = ((pred - target) ** 2).mean(dim=[1, 2, 3]).clamp(min=1e-10)
    return (10 * torch.log10(max_val ** 2 / mse)).mean().item()


def ssim_single(pred, target, win_size=11, sigma=1.5, k1=0.01, k2=0.03):
    """ssim for a single (C,H,W) pair"""
    pred = pred.unsqueeze(0)
    target = target.unsqueeze(0)
    C = pred.shape[1]

    # gaussian window
    coords = torch.arange(win_size, dtype=torch.float32) - win_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.outer(g).unsqueeze(0).unsqueeze(0).to(pred.device)
    window = window.expand(C, 1, -1, -1)

    pad = win_size // 2
    conv = lambda x: torch.nn.functional.conv2d(x, window, padding=pad, groups=C)

    mu1, mu2 = conv(pred), conv(target)
    sigma1_sq = conv(pred * pred) - mu1 ** 2
    sigma2_sq = conv(target * target) - mu2 ** 2
    sigma12 = conv(pred * target) - mu1 * mu2

    C1, C2 = (k1 * 1.0) ** 2, (k2 * 1.0) ** 2
    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return (num / den).mean().item()


def compute_metrics(pred_batch, target_batch):
    p = psnr(pred_batch, target_batch)
    s_vals = [ssim_single(pred_batch[i], target_batch[i])
              for i in range(pred_batch.shape[0])]
    return {'psnr': p, 'ssim': np.mean(s_vals)}


# --- lr schedulers ---

class MultiStepLR_Restart:
    """multistep with relative milestones (e.g. [0.5, 0.75, 0.9, 0.95])"""
    def __init__(self, optimizer, lr_steps_rel, total_iters, gamma=0.5):
        self.optimizer = optimizer
        self.gamma = gamma
        self.milestones = [int(r * total_iters) for r in lr_steps_rel]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step in self.milestones:
            for pg in self.optimizer.param_groups:
                pg['lr'] *= self.gamma
            print(f"  lr decay at step {self.current_step}")

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return {'current_step': self.current_step,
                'milestones': self.milestones, 'gamma': self.gamma}

    def load_state_dict(self, s):
        self.current_step = s['current_step']
        self.milestones = s['milestones']
        self.gamma = s['gamma']


class CosineAnnealingLR_Custom:
    """cosine lr with optional warmup"""
    def __init__(self, optimizer, total_iters, min_lr=1e-6, warmup_iters=0):
        self.optimizer = optimizer
        self.total_iters = total_iters
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_step = 0

    def _update_lr(self):
        if self.current_step <= self.warmup_iters and self.warmup_iters > 0:
            alpha = self.current_step / max(1, self.warmup_iters)
            for pg, blr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = blr * alpha
        else:
            denom = max(1, self.total_iters - self.warmup_iters)
            prog = min(1.0, (self.current_step - self.warmup_iters) / denom)
            cos_f = 0.5 * (1 + math.cos(math.pi * prog))
            for pg, blr in zip(self.optimizer.param_groups, self.base_lrs):
                ratio = blr / self.base_lrs[0] if self.base_lrs[0] > 0 else 1.0
                mlr = self.min_lr * ratio
                pg['lr'] = mlr + (blr - mlr) * cos_f

    def step(self):
        self.current_step += 1
        self._update_lr()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return {'type': 'CosineAnnealing', 'current_step': self.current_step,
                'total_iters': self.total_iters, 'min_lr': self.min_lr,
                'warmup_iters': self.warmup_iters, 'base_lrs': self.base_lrs}

    def load_state_dict(self, s):
        self.current_step = s['current_step']
        self.total_iters = s['total_iters']
        self.min_lr = s['min_lr']
        self.warmup_iters = s.get('warmup_iters', 0)
        self.base_lrs = s['base_lrs']
        self._update_lr()


# --- checkpoint handling ---

def save_checkpoint(model, optimizer, scheduler, step, save_dir,
                    name='G', is_best=False, best_psnr=0.0):
    os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        'step': step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict() if scheduler else None,
        'best_psnr': best_psnr,
    }

    path = os.path.join(save_dir, f'{step}_{name}.pth')
    torch.save(ckpt, path)

    if is_best:
        best_path = os.path.join(save_dir, f'best_psnr_{name}.pth')
        shutil.copyfile(path, best_path)
        print(f"  best model saved: {best_path} (PSNR={best_psnr:.4f})")

        # try to also save to kaggle persistent storage
        try:
            kaggle_dir = '/kaggle/input/eliei-dataset/models/'
            os.makedirs(kaggle_dir, exist_ok=True)
            shutil.copyfile(path, os.path.join(kaggle_dir, f'{step}_{name}.pth'))
            shutil.copyfile(path, os.path.join(kaggle_dir, f'best_psnr_{name}.pth'))
        except Exception:
            pass

    return path


def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'], strict=False)

    if optimizer and 'optimizer_state' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        except (ValueError, RuntimeError) as e:
            print(f'  warning: couldn\'t restore optimizer ({e})')

    if scheduler and ckpt.get('scheduler_state'):
        try:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        except Exception as e:
            print(f'  warning: couldn\'t restore scheduler ({e})')

    step = ckpt.get('step', 0)
    best_psnr = ckpt.get('best_psnr', 0.0)
    print(f"  loaded checkpoint: {path} (step={step})")
    return step, best_psnr


def save_image(tensor, path):
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = tensor_to_numpy(tensor)
    Image.fromarray(arr).save(path)


def opt_get(opt, keys, default=None):
    val = opt
    for k in keys:
        if not isinstance(val, dict) or k not in val:
            return default
        val = val[k]
    return val
