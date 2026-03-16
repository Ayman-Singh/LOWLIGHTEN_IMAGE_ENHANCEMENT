# ELIEI Implementation — Walkthrough
## Enhancing Low-Light Images Using Infrared Encoded Images (ICIP 2023)

> **Implemented from scratch** in `d:\Low-lighten image\ELIEI_Implementation\`

---

## What Was Built

A complete PyTorch implementation of the ELIEI paper, including model, training, evaluation, and inference scripts — all written from scratch based on the paper and its official config.

---

## File Structure

```
ELIEI_Implementation/
├── confs/
│   └── IR-RGB.yaml          # Paper-accurate training config
├── models/
│   ├── encoder.py           # RRDB conditional encoder (nb=4, nf=32)
│   ├── flow.py              # Normalizing flow (K=4, L=3 levels)
│   ├── loss.py              # NLL + Color Alignment Loss (EMD)
│   └── model.py             # Combined ELIEI model wrapper
├── data/
│   └── dataset.py           # IR-RGB paired dataset + log transform
├── utils/
│   └── utils.py             # PSNR, SSIM, LR scheduler, checkpointing
├── train.py                 # Full training loop (AMP, TensorBoard)
├── test.py                  # Batch inference + metric computation
├── evaluate.py              # Standalone PSNR/SSIM/LPIPS evaluator
├── smoke_test.py            # Unit tests for all components
├── requirements.txt
└── README.md
```

---

## Architecture (Paper-Exact)

| Component | Details |
|---|---|
| **Conditional Encoder** | 4 RRDB blocks, nf=32, gc=32 |
| **Condition dim** | 64 per level × 4^l channels |
| **Flow levels (L)** | 3 |
| **Flow steps (K)** | 4 per level |
| **Coupling** | CondAffineSeparatedAndCond (affine) |
| **No-affine steps** | 2 per level (additive coupling) |
| **Squeeze** | pixel_unshuffle ×2 per level |
| **Optimizer** | Adam β1=0.9 β2=0.99, lr=5e-4 |
| **LR decay** | ×0.5 at 50%, 75%, 90%, 95% of 200k iters |
| **Batch size** | 16, crop 160×160 |
| **Total params** | ~29.5M |

---

## Loss Functions (Paper-Exact)

**Total:** `L = L_nll + 0.01 × L_CA`

- **NLL loss**: Negative log-likelihood of the normalizing flow
- **Color Alignment Loss (CAL)**: Earth Mover's Distance between per-channel differentiable histograms
  - Kernel: `1 / (1 + (x-μ)²/σ²)`, σ=0.002
  - 64 bins, range [0,1]
  - Images downsampled to 40×40 before histogram computation
  - EMD via CDF difference

---

## Smoke Test Results ✅

```
[1/5] Testing imports...                    OK
[2/5] Testing Color Alignment Loss...       OK (loss=0.0241)
[3/5] Testing NLL Loss...                   OK
[4/5] Testing conditional encoder...        OK — L0:(1,64,64,64) | L1:(1,256,32,32) | L2:(1,1024,16,16)
[5/5] Testing full model (forward+enhance). OK — z:(1,48,16,16), enhanced:(1,3,64,64)
[Bonus] Combined ELIEILoss...               OK
[Bonus2] PSNR / SSIM metrics...             OK

ALL TESTS PASSED!
Paper model (K=4,L=3,nf=32) params: 29,468,476
```

---

## Dataset

| Split | Path | Count |
|---|---|---|
| Train | `train-20241116T084056Z-001/train/` | 236 pairs |
| Eval  | `eval-20241116T084057Z-001/eval/`   | ~80 pairs |

Naming: `high/1.png` ↔ `low/1.png` (matched by filename stem).

---

## How to Train

```bash
cd "d:\Low-lighten image\ELIEI_Implementation"
pip install -r requirements.txt
python train.py --config confs/IR-RGB.yaml
```

Training checkpoints saved to `experiments/ELIEI_IR_RGB/models/`.

---

## How to Test

```bash
python test.py --config confs/IR-RGB.yaml \
    --checkpoint experiments/ELIEI_IR_RGB/models/best_psnr_G.pth

# standalone metrics
python evaluate.py \
    --pred_dir results/ \
    --gt_dir ../eval-20241116T084057Z-001/eval/high \
    --use_lpips
```

---

## Paper Target Metrics

| PSNR | SSIM | LPIPS |
|------|------|-------|
| 26.23 dB | 0.899 | 0.116 |

---

## Bug Fixes Applied During Training Run

Once training was launched on the real machine, 5 issues were caught and fixed:

| # | Bug | Fix |
|---|-----|-----|
| 1 | TensorBoard protobuf conflict (`MessageFactory.GetPrototype`) | Wrapped import in broad `except Exception` — training continues without TB |
| 2 | `MemoryError` from `n_workers=4` on Windows (`spawn`) | Force `num_workers=0` on Windows in `build_dataloader` |
| 3 | `pin_memory` warning with no CUDA | `pin_memory = torch.cuda.is_available()` |
| 4 | `torch.amp.autocast('cuda')` with no GPU | `amp_device = 'cuda' if use_amp else 'cpu'` |
| 5 | `model.enhance()` called `self.eval()` internally during training | Removed self.eval() from enhance(); training loop manages mode explicitly |

## End-to-End Training Verification ✅

```
[Dataset] train: 236 pairs loaded
Batch loaded. lr: torch.Size([2, 3, 64, 64])  hr: torch.Size([2, 3, 64, 64])
Flow forward OK. z: torch.Size([2, 48, 16, 16])
Enhance OK. pred: torch.Size([2, 3, 64, 64])  range: [0.0, 1.0]
Loss OK: {'nll': 2.813, 'cal': 0.021, 'total': 2.814}
Backward + step OK
=== 1 TRAINING STEP COMPLETED SUCCESSFULLY ===
```
