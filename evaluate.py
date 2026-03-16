# evaluate.py — compute PSNR/SSIM/LPIPS between predicted and GT images
#
# usage: python evaluate.py --pred_dir results/ --gt_dir data/eval/high [--use_lpips]

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')

import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.utils import compute_metrics, tensor_to_numpy
import torchvision.transforms.functional as TF


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pred_dir', type=str, required=True)
    p.add_argument('--gt_dir', type=str, required=True)
    p.add_argument('--use_lpips', action='store_true')
    return p.parse_args()


def load_tensor(path):
    return TF.to_tensor(Image.open(path).convert('RGB'))


def find_pairs(pred_dir, gt_dir):
    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    preds = {Path(f).stem.replace('_enhanced', ''): os.path.join(pred_dir, f)
             for f in os.listdir(pred_dir) if Path(f).suffix.lower() in exts}
    gts = {Path(f).stem: os.path.join(gt_dir, f)
           for f in os.listdir(gt_dir) if Path(f).suffix.lower() in exts}
    common = set(preds) & set(gts)
    return [(preds[k], gts[k]) for k in sorted(common)]


def main():
    args = parse_args()
    pairs = find_pairs(args.pred_dir, args.gt_dir)
    print(f'found {len(pairs)} pairs')

    if not pairs:
        print('no matching pairs found')
        return

    lpips_fn = None
    if args.use_lpips:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net='alex').eval()
            print('lpips loaded')
        except ImportError:
            print('lpips not installed, skipping')

    psnr_list, ssim_list, lpips_list = [], [], []

    for pred_path, gt_path in pairs:
        pred = load_tensor(pred_path)
        gt = load_tensor(gt_path)

        if pred.shape != gt.shape:
            pred = F.interpolate(pred.unsqueeze(0), size=gt.shape[1:],
                                 mode='bilinear', align_corners=False).squeeze(0)

        metrics = compute_metrics(pred.unsqueeze(0), gt.unsqueeze(0))
        psnr_list.append(metrics['psnr'])
        ssim_list.append(metrics['ssim'])

        if lpips_fn is not None:
            with torch.no_grad():
                lp = lpips_fn(pred.unsqueeze(0) * 2 - 1,
                              gt.unsqueeze(0) * 2 - 1).item()
            lpips_list.append(lp)

    print(f'\n--- evaluation ({len(pairs)} images) ---')
    print(f'PSNR:  {np.mean(psnr_list):.4f} dB')
    print(f'SSIM:  {np.mean(ssim_list):.4f}')
    if lpips_list:
        print(f'LPIPS: {np.mean(lpips_list):.4f}')


if __name__ == '__main__':
    main()
