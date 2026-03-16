# visual_compare.py — run the model on eval images and display side-by-side comparisons
#
# usage:
#   python visual_compare.py --checkpoint experiments/ELIEI_IR_RGB/models/best_psnr_G.pth
#   python visual_compare.py --checkpoint <path> --n 8 --save comparison_grid.png
#
# shows: low-light input | model prediction | ground truth
# also prints PSNR/SSIM for each image

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')

import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.model import build_model
from data.dataset import IRRGBDataset, log_transform
from utils.utils import load_config, load_checkpoint, compute_metrics


def parse_args():
    p = argparse.ArgumentParser(description='Visual comparison on eval set')
    p.add_argument('--config', type=str, default='confs/IR-RGB-kaggle.yaml')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='path to trained model checkpoint')
    p.add_argument('--data_root', type=str, default=None,
                   help='override eval data root (default: from config)')
    p.add_argument('--n', type=int, default=6,
                   help='number of images to compare')
    p.add_argument('--heat', type=float, default=0.0,
                   help='sampling temperature (0 = deterministic)')
    p.add_argument('--save', type=str, default=None,
                   help='save the comparison grid to this path (e.g. comparison.png)')
    p.add_argument('--no_display', action='store_true',
                   help='skip plt.show() (for headless environments)')
    return p.parse_args()


def tensor_to_img(t):
    """(C,H,W) tensor [0,1] -> (H,W,C) numpy for matplotlib"""
    return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def undo_log_transform(x, eps=1e-3):
    """reverse the log transform to get back approximate original pixel values"""
    return torch.exp(x * np.log(1 + eps)) - eps


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # build and load model
    model = build_model(cfg).to(device)
    model.eval()

    if os.path.isfile(args.checkpoint):
        load_checkpoint(args.checkpoint, model, device=str(device))
    else:
        print(f'ERROR: checkpoint not found: {args.checkpoint}')
        return

    # load eval dataset
    data_root = args.data_root or cfg['datasets']['val']['root']
    log_low = cfg['datasets']['val'].get('log_low', True)

    dataset = IRRGBDataset(
        root=data_root, crop_size=None, use_flip=False,
        log_low=log_low, split='eval')

    n = min(args.n, len(dataset))
    print(f'comparing {n} images...\n')

    # collect results
    results = []
    for idx in range(n):
        sample = dataset[idx]
        lr = sample['lr'].unsqueeze(0).to(device)
        hr = sample['hr'].unsqueeze(0).to(device)
        lr_path = sample['lr_path']

        enhanced = model.enhance(lr, heat=args.heat)
        metrics = compute_metrics(enhanced, hr)

        # undo log transform on the low-light input for display
        if log_low:
            lr_display = undo_log_transform(sample['lr']).clamp(0, 1)
        else:
            lr_display = sample['lr']

        fname = os.path.basename(lr_path)
        print(f'  [{idx+1}/{n}] {fname}  '
              f'PSNR={metrics["psnr"]:.2f} dB  SSIM={metrics["ssim"]:.4f}')

        results.append({
            'name': fname,
            'low': tensor_to_img(lr_display),
            'enhanced': tensor_to_img(enhanced[0]),
            'gt': tensor_to_img(sample['hr']),
            'psnr': metrics['psnr'],
            'ssim': metrics['ssim'],
        })

    # compute averages
    avg_psnr = np.mean([r['psnr'] for r in results])
    avg_ssim = np.mean([r['ssim'] for r in results])
    print(f'\naverage over {n} images:')
    print(f'  PSNR: {avg_psnr:.4f} dB')
    print(f'  SSIM: {avg_ssim:.4f}')

    # plot comparison grid
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, r in enumerate(results):
        axes[i][0].imshow(r['low'])
        axes[i][0].set_title(f'Low-light Input\n{r["name"]}', fontsize=10)
        axes[i][0].axis('off')

        axes[i][1].imshow(r['enhanced'])
        axes[i][1].set_title(f'Enhanced (Ours)\n'
                             f'PSNR={r["psnr"]:.2f} SSIM={r["ssim"]:.4f}',
                             fontsize=10)
        axes[i][1].axis('off')

        axes[i][2].imshow(r['gt'])
        axes[i][2].set_title('Ground Truth', fontsize=10)
        axes[i][2].axis('off')

    plt.suptitle(f'ELIEI Results — Avg PSNR: {avg_psnr:.2f} dB, '
                 f'Avg SSIM: {avg_ssim:.4f}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f'\nsaved grid to {args.save}')

    if not args.no_display:
        plt.show()
    else:
        print('(display skipped, use --save to save the grid)')


if __name__ == '__main__':
    main()
