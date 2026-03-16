# test.py — run inference on eval set
#
# usage:
#   python test.py --config confs/IR-RGB.yaml --checkpoint experiments/ELIEI_IR_RGB/models/best_psnr_G.pth
#   python test.py --config confs/IR-RGB.yaml --checkpoint <ckpt> --heat 0.0 --output_dir results/

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')

import sys
import argparse
import time
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.model import build_model
from data.dataset import IRRGBDataset, log_transform
from utils.utils import load_config, load_checkpoint, save_image, compute_metrics


def parse_args():
    p = argparse.ArgumentParser(description='Test ELIEI')
    p.add_argument('--config', type=str, default='confs/IR-RGB.yaml')
    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--heat', type=float, default=0.0)
    p.add_argument('--output_dir', type=str, default='results')
    p.add_argument('--data_root', type=str, default=None)
    return p.parse_args()


@torch.no_grad()
def test():
    args = parse_args()
    cfg = load_config(args.config)

    gpu_ids = cfg.get('gpu_ids', [0])
    device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    model = build_model(cfg).to(device)
    model.eval()

    ckpt_path = args.checkpoint or cfg.get('model_path')
    if ckpt_path and os.path.isfile(ckpt_path):
        load_checkpoint(ckpt_path, model, device=device)
    else:
        print(f'warning: no checkpoint at {ckpt_path}, using random weights')

    data_root = args.data_root or cfg['datasets']['val']['root']
    log_low = cfg['datasets']['val'].get('log_low', True)

    dataset = IRRGBDataset(
        root=data_root, crop_size=None, use_flip=False,
        log_low=log_low, split='val')

    os.makedirs(args.output_dir, exist_ok=True)
    psnr_list, ssim_list = [], []

    print(f'testing on {len(dataset)} images...')
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        lr = sample['lr'].unsqueeze(0).to(device)
        hr = sample['hr'].unsqueeze(0).to(device)
        lr_path = sample['lr_path']

        t0 = time.time()
        enhanced = model.enhance(lr, heat=args.heat)
        dt = time.time() - t0

        metrics = compute_metrics(enhanced, hr)
        psnr_list.append(metrics['psnr'])
        ssim_list.append(metrics['ssim'])

        fname = os.path.splitext(os.path.basename(lr_path))[0]
        out_path = os.path.join(args.output_dir, f'{fname}_enhanced.png')
        save_image(enhanced[0], out_path)

        if (idx + 1) % 10 == 0 or idx == 0:
            print(f'[{idx+1}/{len(dataset)}] PSNR={metrics["psnr"]:.2f} '
                  f'SSIM={metrics["ssim"]:.4f} time={dt:.3f}s')

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    print(f'\n--- results ---')
    print(f'PSNR: {avg_psnr:.4f} dB')
    print(f'SSIM: {avg_ssim:.4f}')
    print(f'saved {len(dataset)} images to {args.output_dir}')

    with open(os.path.join(args.output_dir, 'results.txt'), 'w') as f:
        f.write(f'PSNR: {avg_psnr:.4f}\nSSIM: {avg_ssim:.4f}\n')
        for i, (p, s) in enumerate(zip(psnr_list, ssim_list)):
            f.write(f'{i:04d}: PSNR={p:.4f} SSIM={s:.4f}\n')


if __name__ == '__main__':
    test()
