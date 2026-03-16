# dataset loading for ELIEI
# handles paired low-light / normal-light images
#
# folder structure:
#   root/
#     high/   -> ground truth (normal-light)
#     low/    -> IR-encoded low-light input
#
# preprocessing: random 160x160 crop, h-flip, log transform on low-light

import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T


def log_transform(x, eps=1e-3):
    """log transform to expand dynamic range. maps [0,1] -> ~[0,1]"""
    return torch.log(x + eps) / np.log(1 + eps)


def get_image_paths(folder):
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(root, f))
    return sorted(paths)


class IRRGBDataset(Dataset):
    """paired low/high dataset for IR-RGB enhancement"""

    def __init__(self, root, crop_size=160, use_flip=True, log_low=True,
                 quant=32, split='train', n_max=None):
        super().__init__()
        self.crop_size = crop_size
        self.use_flip = use_flip
        self.log_low = log_low
        self.quant = quant
        self.split = split

        high_dir = os.path.join(root, 'high')
        low_dir = os.path.join(root, 'low')
        assert os.path.isdir(high_dir), f"missing: {high_dir}"
        assert os.path.isdir(low_dir), f"missing: {low_dir}"

        high_paths = get_image_paths(high_dir)
        low_paths = get_image_paths(low_dir)
        self.pairs = self._match(high_paths, low_paths)

        if n_max and n_max > 0:
            self.pairs = self.pairs[:n_max]

        print(f"[{split}] {len(self.pairs)} pairs from {root}")

    def _match(self, high_paths, low_paths):
        """match pairs by filename stem"""
        high_map = {os.path.splitext(os.path.basename(p))[0]: p for p in high_paths}
        low_map = {os.path.splitext(os.path.basename(p))[0]: p for p in low_paths}
        common = sorted(set(high_map) & set(low_map))

        if not common:
            # fallback: pair by index if names don't match
            n = min(len(high_paths), len(low_paths))
            print(f"  warning: no matching filenames, pairing by index")
            return list(zip(high_paths[:n], low_paths[:n]))

        return [(high_map[k], low_map[k]) for k in common]

    def __len__(self):
        return len(self.pairs)

    def _load(self, path):
        img = Image.open(path).convert('RGB')
        return TF.to_tensor(img)

    def _quantize(self, img):
        if self.quant:
            img = torch.round(img * self.quant) / self.quant
        return img

    def __getitem__(self, idx):
        high_path, low_path = self.pairs[idx]
        high = self._load(high_path)
        low = self._load(low_path)

        assert high.shape == low.shape, \
            f"size mismatch: {high.shape} vs {low.shape}"

        # random crop
        if self.crop_size is not None:
            _, H, W = high.shape
            if H < self.crop_size or W < self.crop_size:
                scale = max(self.crop_size / H, self.crop_size / W) + 0.01
                new_H, new_W = int(H * scale), int(W * scale)
                high = TF.resize(high, [new_H, new_W],
                                 interpolation=T.InterpolationMode.BICUBIC)
                low = TF.resize(low, [new_H, new_W],
                                interpolation=T.InterpolationMode.BICUBIC)
                _, H, W = high.shape

            i = random.randint(0, H - self.crop_size)
            j = random.randint(0, W - self.crop_size)
            high = TF.crop(high, i, j, self.crop_size, self.crop_size)
            low = TF.crop(low, i, j, self.crop_size, self.crop_size)

        # h-flip augmentation
        if self.use_flip and random.random() > 0.5:
            high = TF.hflip(high)
            low = TF.hflip(low)

        high = self._quantize(high).clamp(0.0, 1.0)
        low = self._quantize(low).clamp(0.0, 1.0)

        if self.log_low:
            low = log_transform(low)

        return {
            'lr': low,
            'hr': high,
            'lr_path': low_path,
            'hr_path': high_path,
        }


def build_dataloader(opt, split='train'):
    ds_cfg = opt['datasets'][split]
    is_train = (split == 'train')

    dataset = IRRGBDataset(
        root=ds_cfg['root'],
        crop_size=ds_cfg.get('GT_size', 160) if is_train else None,
        use_flip=ds_cfg.get('use_flip', True) if is_train else False,
        log_low=ds_cfg.get('log_low', True),
        quant=ds_cfg.get('quant', 32),
        split=split,
        n_max=ds_cfg.get('n_max', None),
    )

    # windows/CPU: force workers=0 to avoid spawn issues
    import platform
    if platform.system() == 'Windows' or not torch.cuda.is_available():
        n_workers = 0
        pin_mem = False
    else:
        n_workers = ds_cfg.get('n_workers', 4)
        pin_mem = True

    return DataLoader(
        dataset,
        batch_size=ds_cfg.get('batch_size', 16),
        shuffle=ds_cfg.get('use_shuffle', True) if is_train else False,
        num_workers=n_workers,
        pin_memory=pin_mem,
        drop_last=is_train,
        persistent_workers=(n_workers > 0),
    )
