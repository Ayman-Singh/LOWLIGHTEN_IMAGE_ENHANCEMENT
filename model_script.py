#!/usr/bin/env python3
"""
LLFlow: Low-Light Image Enhancement with Conditional Normalizing Flows
Corrected and compacted version that preserves original functionality.

Key fixes:
 - Conditional prior μ(x) used for inference and NLL (removes mid-gray output).
 - Proper log-determinant accumulation from coupling layers.
 - Removed invalid torch.load kwarg(s).
 - Stabilized coupling scale with bounded tanh.
 - Preserves: full training/eval/enhance/demo modes, tensorboard, checkpoints,
   differentiable histogram CAL, triptych visualizer, AMP, gradient clipping.
"""

import os, json, time, logging, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

# ---------------- logging / image resampling compatibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("LLFlow")
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS

# ---------------- reproducibility
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ---------------- configuration (kept original defaults)
@dataclass
class Config:
    """Configuration dataclass for clean parameter management"""
    # Data paths - Update these to match your directory structure
    train_low_dir: str = "train-20241116T084056Z-001-20250713T124140Z-1-001/train-20241116T084056Z-001/train/low"
    train_high_dir: str = "train-20241116T084056Z-001-20250713T124140Z-1-001/train-20241116T084056Z-001/train/high"
    val_low_dir: str = "eval-20241116T084057Z-001-20250713T124140Z-1-001/eval-20241116T084057Z-001/eval/low"
    val_high_dir: str = "eval-20241116T084057Z-001-20250713T124140Z-1-001/eval-20241116T084057Z-001/eval/high"

    image_size: Tuple[int, int] = (256, 256)
    in_channels: int = 3
    flow_channels: int = 256
    num_flow_blocks: int = 4
    condition_channels: int = 64

    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 5.0

    nll_weight: float = 1.0
    cal_weight: float = 0.01
    l1_weight: float = 1.0

    histogram_bins: int = 64
    wasserstein_p: int = 1

    use_amp: bool = True
    num_workers: int = 4
    pin_memory: bool = True

    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    results_dir: str = "results"
    save_freq: int = 10
    val_freq: int = 5

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# Create backup default dirs so logs/checkpoints exist even before CLI config loaded
os.makedirs(Config.checkpoint_dir, exist_ok=True)
os.makedirs(Config.log_dir, exist_ok=True)
os.makedirs(Config.results_dir, exist_ok=True)

# ---------------- Dataset
class LowLightDataset(Dataset):
    def __init__(self, low_dir: str, high_dir: str, image_size=(256,256), augment=True, return_paths=False):
        self.low_dir = Path(low_dir)
        self.high_dir = Path(high_dir)
        self.image_size = image_size
        self.augment = augment
        self.return_paths = return_paths
        self.image_pairs = self._find_paired_images()
        if len(self.image_pairs) == 0:
            raise ValueError(f"No paired images found in {low_dir} and {high_dir}")
        logger.info(f"Found {len(self.image_pairs)} paired images")

    def _find_paired_images(self) -> List[Tuple[Path, Path]]:
        pairs = []
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        low_images = []
        for ext in extensions:
            low_images.extend(self.low_dir.glob(ext))
        for low_path in sorted(low_images):
            high_path = self.high_dir / low_path.name
            if high_path.exists():
                pairs.append((low_path, high_path))
            else:
                for ext in extensions:
                    ext_clean = ext.replace('*', '')
                    high_path = self.high_dir / (low_path.stem + ext_clean)
                    if high_path.exists():
                        pairs.append((low_path, high_path))
                        break
        return pairs

    def __len__(self): 
        return len(self.image_pairs)

    def __getitem__(self, idx: int):
        low_path, high_path = self.image_pairs[idx]
        try:
            low_img = Image.open(low_path).convert('RGB').resize(self.image_size, RESAMPLE)
            high_img = Image.open(high_path).convert('RGB').resize(self.image_size, RESAMPLE)
            low_np = np.array(low_img, dtype=np.float32) / 255.0
            high_np = np.array(high_img, dtype=np.float32) / 255.0
            if self.augment:
                if np.random.rand() > 0.5:
                    low_np, high_np = np.fliplr(low_np), np.fliplr(high_np)
                if np.random.rand() > 0.5:
                    low_np, high_np = np.flipud(low_np), np.flipud(high_np)
                k = np.random.randint(0,4)
                if k > 0:
                    low_np, high_np = np.rot90(low_np, k).copy(), np.rot90(high_np, k).copy()
            low_t = torch.from_numpy(low_np.transpose(2,0,1))
            high_t = torch.from_numpy(high_np.transpose(2,0,1))
            if self.return_paths:
                return low_t, high_t, low_path.name
            return low_t, high_t
        except Exception as e:
            logger.error(f"Error loading images {low_path} / {high_path}: {e}")
            fallback = torch.zeros(3, *self.image_size)
            if self.return_paths:
                return fallback, fallback, low_path.name
            return fallback, fallback

# ---------------- Conditional coupling / invertible block
class ConditionalCouplingLayer(nn.Module):
    def __init__(self, in_channels: int, cond_channels: int):
        super().__init__()
        self.split_channels = in_channels // 2
        self.net = nn.Sequential(
            nn.Conv2d(self.split_channels + cond_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.split_channels * 2, 3, padding=1)
        )
        last = self.net[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor, reverse: bool = False, return_logdet: bool = False):
        x1, x2 = x.chunk(2, dim=1)
        x1_cond = torch.cat([x1, condition], dim=1)
        params = self.net(x1_cond)
        scale, shift = params.chunk(2, dim=1)
        # bound scale to stable range
        scale = 0.5 * torch.tanh(scale)
        if not reverse:
            x2 = x2 * torch.exp(scale) + shift
            logdet = scale.flatten(1).sum(dim=1)
        else:
            x2 = (x2 - shift) * torch.exp(-scale)
            logdet = -scale.flatten(1).sum(dim=1)
        out = torch.cat([x1, x2], dim=1)
        return (out, logdet) if return_logdet else out

class InvertibleBlock(nn.Module):
    def __init__(self, channels: int, cond_channels: int):
        super().__init__()
        self.coupling = ConditionalCouplingLayer(channels, cond_channels)
    
    def forward(self, x, condition, reverse=False, return_logdet=False):
        return self.coupling(x, condition, reverse, return_logdet)

# ---------------- ConditionalFlow (encoder, flows, proj)
class ConditionalFlow(nn.Module):
    def __init__(self, num_layers: int = 4, flow_channels: int = 256):
        super().__init__()
        self.num_layers = num_layers
        self.flow_channels = flow_channels
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, flow_channels, 3, padding=1), nn.ReLU(inplace=True)
        )
        # conditional prior mean μ(x)
        self.prior_mu = nn.Conv2d(flow_channels, flow_channels, 1)
        self.flow_layers = nn.ModuleList([InvertibleBlock(flow_channels, flow_channels) for _ in range(num_layers)])
        self.input_projection = nn.Conv2d(3, flow_channels, 1)
        self.output_projection = nn.Conv2d(flow_channels, 3, 1)

    def forward(self, y: torch.Tensor, x: torch.Tensor, reverse: bool = False):
        """
        If reverse=False: y -> z; returns (z, log_det_sum, mu)
        If reverse=True: z -> y_recon (y arg is actually z)
        """
        condition = self.condition_encoder(x)
        mu = self.prior_mu(condition)
        if not reverse:
            z = self.input_projection(y)
            log_det_sum = torch.zeros(y.shape[0], device=y.device)
            for layer in self.flow_layers:
                z, ld = layer(z, condition, reverse=False, return_logdet=True)
                log_det_sum = log_det_sum + ld
            return z, log_det_sum, mu
        else:
            z = y
            for layer in reversed(self.flow_layers):
                z = layer(z, condition, reverse=True, return_logdet=False)
            y_recon = self.output_projection(z)
            y_recon = torch.sigmoid(y_recon)
            return y_recon

# ---------------- Differentiable histogram + Wasserstein CAL
def differentiable_histogram(x: torch.Tensor, num_bins: int = 64) -> torch.Tensor:
    B, C, H, W = x.shape
    bins = torch.linspace(0, 1, num_bins + 1, device=x.device)
    bin_centers = (bins[:-1] + bins[1:]) / 2  # [bins]
    x_flat = x.view(B, C, -1)  # [B, C, HW]
    histograms = []
    delta = 0.1
    for c in range(C):
        x_channel = x_flat[:, c:c+1, :]  # [B, 1, HW]
        distances = torch.abs(x_channel.unsqueeze(-1) - bin_centers.view(1, 1, 1, -1))  # [B,1,HW,bins]
        soft_assignment = 1.0 / (1.0 + delta * distances ** 2)
        soft_assignment = soft_assignment / (soft_assignment.sum(dim=-1, keepdim=True) + 1e-8)
        histogram = soft_assignment.sum(dim=2)  # [B, 1, bins]
        histograms.append(histogram)
    histograms = torch.cat(histograms, dim=1)  # [B, C, bins]
    histograms = histograms / (histograms.sum(dim=-1, keepdim=True) + 1e-8)
    return histograms

def wasserstein_distance(hist1: torch.Tensor, hist2: torch.Tensor) -> torch.Tensor:
    cumsum1 = torch.cumsum(hist1, dim=-1)
    cumsum2 = torch.cumsum(hist2, dim=-1)
    w_distance = torch.abs(cumsum1 - cumsum2).sum(dim=-1)
    return w_distance.mean()

def color_alignment_loss(y_true: torch.Tensor, y_pred: torch.Tensor, num_bins: int = 64) -> torch.Tensor:
    hist_true = differentiable_histogram(y_true, num_bins)
    hist_pred = differentiable_histogram(y_pred, num_bins)
    return wasserstein_distance(hist_true, hist_pred)

# ---------------- LLFlowModel (forward/infer, compute_loss)
class LLFlowModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.flow = ConditionalFlow(num_layers=config.num_flow_blocks, flow_channels=config.flow_channels)
        self.to(config.device)

    def forward(self, x: torch.Tensor, sample_std: float = 0.0) -> torch.Tensor:
        # Build conditional prior mean mu(x) and sample around it
        with torch.no_grad():
            cond = self.flow.condition_encoder(x)
            mu = self.flow.prior_mu(cond)
        if sample_std and sample_std > 0:
            z = mu + sample_std * torch.randn_like(mu)
        else:
            z = mu
        y_pred = self.flow(z, x, reverse=True)
        return y_pred

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        z, log_det, mu = self.flow(y, x, reverse=False)
        # NLL under N(mu, I)
        z_centered = (z - mu)
        prior_log_prob = -0.5 * z_centered.flatten(1).pow(2).sum(dim=1)  # [B]
        nll_loss = -(prior_log_prob + log_det).mean()
        # reconstruction + CAL
        y_pred = self.flow(z, x, reverse=True)
        l1_loss = F.l1_loss(y_pred, y)
        cal_loss = color_alignment_loss(y, y_pred, self.config.histogram_bins)
        total_loss = (self.config.nll_weight * nll_loss +
                      self.config.cal_weight * cal_loss +
                      self.config.l1_weight * l1_loss)
        return {'total_loss': total_loss, 'nll_loss': nll_loss, 'cal_loss': cal_loss, 'l1_loss': l1_loss}

# ---------------- Trainer (train/validate/checkpoint)
class Trainer:
    def __init__(self, model: LLFlowModel, config: Config):
        self.model = model
        self.config = config
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, config.num_epochs))
        self.scaler = GradScaler() if config.use_amp else None
        self.writer = SummaryWriter(config.log_dir)
        self.global_step = 0
        self.best_psnr = 0.0

    def train_step(self, batch):
        x, y = batch[0].to(self.config.device), batch[1].to(self.config.device)
        self.optimizer.zero_grad()
        if self.config.use_amp and self.scaler is not None:
            with autocast():
                losses = self.model.compute_loss(x, y)
                total_loss = losses['total_loss']
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses = self.model.compute_loss(x, y)
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()
        self.global_step += 1
        return {k: v.item() for k, v in losses.items()}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        num_samples = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    x, y, _ = batch
                else:
                    x, y = batch
                x = x.to(self.config.device)
                y = y.to(self.config.device)
                y_pred = self.model(x, sample_std=0.0)
                y_np = y.cpu().numpy()
                yp_np = y_pred.cpu().numpy()
                for i in range(y_np.shape[0]):
                    gt = np.transpose(y_np[i], (1,2,0))
                    pred = np.transpose(yp_np[i], (1,2,0))
                    ps = psnr_metric(gt, pred, data_range=1.0)
                    ss = ssim_metric(gt, pred, channel_axis=2, data_range=1.0)
                    if isinstance(ss, tuple): 
                        ss = ss[0]
                    total_psnr += float(ps)
                    total_ssim += float(ss)
                    num_samples += 1
        self.model.train()
        avg_psnr = total_psnr / max(1, num_samples)
        avg_ssim = total_ssim / max(1, num_samples)
        return {'psnr': avg_psnr, 'ssim': avg_ssim}

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        logger.info("Starting LLFlow training...")
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_losses = {'total_loss': 0.0, 'nll_loss': 0.0, 'cal_loss': 0.0, 'l1_loss': 0.0}
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            for batch_idx, batch in enumerate(pbar):
                losses = self.train_step(batch)
                for k, v in losses.items():
                    epoch_losses[k] += v
                pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})
                if self.global_step % 10 == 0:
                    for k, v in losses.items():
                        self.writer.add_scalar(f'train/{k}', v, self.global_step)
            for k in epoch_losses:
                epoch_losses[k] /= max(1, len(train_loader))
            if (epoch + 1) % self.config.val_freq == 0:
                val_metrics = self.validate(val_loader)
                logger.info(f"Epoch {epoch+1} - Validation: PSNR: {val_metrics['psnr']:.2f}, SSIM: {val_metrics['ssim']:.4f}")
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f'val/{k}', v, epoch+1)
                if val_metrics['psnr'] > self.best_psnr:
                    self.best_psnr = val_metrics['psnr']
                    self.save_checkpoint(epoch+1, is_best=True)
            if (epoch + 1) % self.config.save_freq == 0:
                self.save_checkpoint(epoch+1)
            self.scheduler.step()
            logger.info(f"Epoch {epoch+1} - Train Loss: {epoch_losses['total_loss']:.4f}, NLL: {epoch_losses['nll_loss']:.4f}, CAL: {epoch_losses['cal_loss']:.4f}, L1: {epoch_losses['l1_loss']:.4f}")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr,
            'global_step': self.global_step
        }
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.config.checkpoint_dir, 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                logger.warning("Couldn't fully restore optimizer state (optimizer mismatch).")
        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception:
                logger.warning("Couldn't fully restore scheduler state.")
        self.best_psnr = float(checkpoint.get('best_psnr', 0.0))
        self.global_step = int(checkpoint.get('global_step', 0))
        logger.info(f"Checkpoint loaded: {path}")
        return checkpoint.get('epoch', 0)

# ---------------- Utilities: inference / save / eval
def make_triptych(low: np.ndarray, pred: np.ndarray, high: Optional[np.ndarray]) -> Image.Image:
    low_u8 = (np.clip(low, 0, 1) * 255).astype(np.uint8)
    pred_u8 = (np.clip(pred, 0, 1) * 255).astype(np.uint8)
    imgs = [low_u8, pred_u8]
    if high is not None:
        imgs.append((np.clip(high, 0, 1) * 255).astype(np.uint8))
    concat = np.hstack(imgs)
    return Image.fromarray(concat)

def evaluate_and_save(model: LLFlowModel, loader: DataLoader, config: Config, out_dir: str) -> Dict[str, float]:
    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(out_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'metrics.csv')
    json_path = os.path.join(out_dir, 'summary.json')
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0
    import csv as _csv
    with open(csv_path, 'w', newline='') as fcsv:
        writer = _csv.writer(fcsv)
        writer.writerow(['filename', 'psnr', 'ssim'])
        with torch.no_grad():
            for batch in tqdm(loader, desc='Evaluate+Save'):
                if len(batch) == 3:
                    x, y, names = batch
                else:
                    x, y = batch
                    names = [f'image_{i:06d}.png' for i in range(x.size(0))]
                x = x.to(config.device)
                y = y.to(config.device)
                y_pred = model(x, sample_std=0.0)
                x_np = x.cpu().numpy()
                y_np = y.cpu().numpy()
                yp_np = y_pred.cpu().numpy()
                for i in range(y_np.shape[0]):
                    gt = np.transpose(y_np[i], (1,2,0))
                    pred = np.transpose(yp_np[i], (1,2,0))
                    low = np.transpose(x_np[i], (1,2,0))
                    psnr_v = psnr_metric(gt, pred, data_range=1.0)
                    ssim_v = ssim_metric(gt, pred, channel_axis=2, data_range=1.0)
                    if isinstance(ssim_v, tuple): 
                        ssim_v = ssim_v[0]
                    total_psnr += float(psnr_v)
                    total_ssim += float(ssim_v)
                    n += 1
                    out_name = os.path.splitext(names[i])[0] + '.png'
                    Image.fromarray((pred * 255.0).clip(0,255).astype(np.uint8)).save(os.path.join(images_dir, out_name))
                    make_triptych(low, pred, gt).save(os.path.join(images_dir, os.path.splitext(names[i])[0] + '_viz.png'))
                    writer.writerow([out_name, f'{psnr_v:.4f}', f'{ssim_v:.6f}'])
    avg_psnr = total_psnr / max(1, n)
    avg_ssim = total_ssim / max(1, n)
    summary = {'num_images': int(n), 'psnr': float(avg_psnr), 'ssim': float(avg_ssim)}
    with open(json_path, 'w') as fj:
        json.dump(summary, fj, indent=2)
    logger.info(f"Saved per-image results to {out_dir}")
    return {'psnr': avg_psnr, 'ssim': avg_ssim}

def enhance_image(model: LLFlowModel, image_path: str, output_path: str, config: Config):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image = image.resize(config.image_size, RESAMPLE)
    image_np = np.array(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np.transpose(2,0,1)).unsqueeze(0).to(config.device)
    with torch.no_grad():
        enhanced_tensor = model(image_tensor, sample_std=0.0)
    enhanced_np = enhanced_tensor[0].cpu().numpy().transpose(1,2,0)
    enhanced_np = np.clip(enhanced_np * 255, 0, 255).astype(np.uint8)
    enhanced_image = Image.fromarray(enhanced_np).resize(original_size, RESAMPLE)
    enhanced_image.save(output_path)
    logger.info(f"Enhanced image saved to {output_path}")

# ---------------- main CLI
def main():
    parser = argparse.ArgumentParser(description='LLFlow: Low-Light Image Enhancement')
    parser.add_argument('--mode', choices=['train','evaluate','enhance','demo'], default='train')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--input', type=str, help='Input image (for enhance)')
    parser.add_argument('--output', type=str, help='Output image (for enhance)')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    args = parser.parse_args()

    config = Config()
    if args.config:
        with open(args.config, 'r') as f:
            cfg_dict = json.load(f)
            for k, v in cfg_dict.items():
                if hasattr(config, k):
                    setattr(config, k, v)

    # ensure directories after potential config overrides
    for dir_path in [config.checkpoint_dir, config.log_dir, config.results_dir]:
        os.makedirs(dir_path, exist_ok=True)

    model = LLFlowModel(config)

    if args.mode == 'train':
        train_dataset = LowLightDataset(config.train_low_dir, config.train_high_dir, config.image_size, augment=True)
        val_dataset   = LowLightDataset(config.val_low_dir,   config.val_high_dir,   config.image_size, augment=False)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=config.pin_memory)
        val_loader   = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)
        trainer = Trainer(model, config)
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        trainer.train(train_loader, val_loader)

    elif args.mode == 'evaluate':
        if not args.checkpoint: 
            raise ValueError("Checkpoint is required for evaluation (--checkpoint)")
        ck = torch.load(args.checkpoint, map_location=config.device)
        model.load_state_dict(ck['model_state_dict'])
        test_dataset = LowLightDataset(config.val_low_dir, config.val_high_dir, config.image_size, augment=False, return_paths=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)
        trainer = Trainer(model, config)
        val_metrics = trainer.validate(test_loader)
        logger.info(f"Test Results - PSNR: {val_metrics['psnr']:.2f}, SSIM: {val_metrics['ssim']:.4f}")
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join(config.results_dir, f'eval_{timestamp}')
        _ = evaluate_and_save(model, test_loader, config, out_dir)
        results = {'psnr': float(val_metrics['psnr']), 'ssim': float(val_metrics['ssim']), 'checkpoint': args.checkpoint, 'num_test_samples': len(test_dataset), 'results_dir': out_dir}
        with open(os.path.join(config.results_dir, 'evaluation_results.json'), 'w') as f: 
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {os.path.join(config.results_dir, 'evaluation_results.json')}")

    elif args.mode == 'enhance':
        if not args.checkpoint: 
            raise ValueError("Checkpoint required for enhance")
        if not args.input or not args.output: 
            raise ValueError("Provide --input and --output for enhance")
        ck = torch.load(args.checkpoint, map_location=config.device)
        model.load_state_dict(ck['model_state_dict'])
        enhance_image(model, args.input, args.output, config)
        logger.info("Enhancement complete!")

    elif args.mode == 'demo':
        logger.info("Demo: quick train on val set (1 epoch) then evaluate")
        train_dataset = LowLightDataset(config.val_low_dir, config.val_high_dir, config.image_size, augment=True)
        val_dataset   = LowLightDataset(config.val_low_dir, config.val_high_dir, config.image_size, augment=False, return_paths=True)
        train_loader = DataLoader(train_dataset, batch_size=min(4, config.batch_size), shuffle=True, num_workers=max(0, config.num_workers-2), pin_memory=config.pin_memory)
        val_loader   = DataLoader(val_dataset,   batch_size=min(4, config.batch_size), shuffle=False, num_workers=max(0, config.num_workers-2), pin_memory=config.pin_memory)
        orig_epochs = config.num_epochs
        config.num_epochs = 1
        trainer = Trainer(model, config)
        trainer.train(train_loader, val_loader)
        config.num_epochs = orig_epochs
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join(config.results_dir, f'demo_{timestamp}')
        _ = evaluate_and_save(model, val_loader, config, out_dir)
        logger.info(f"Demo results saved to {out_dir}")

if __name__ == '__main__':
    main()
