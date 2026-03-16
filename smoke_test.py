# quick sanity check — run this to verify everything imports + forward pass works
# usage: python smoke_test.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

print("running smoke tests...\n")

# imports
print("[1/5] imports...", end=" ", flush=True)
from models.model import build_model, ELIEIModel
from models.encoder import ConditionalEncoder
from models.flow import LLFlowNet, ActNorm, InvConv1x1, CondAffineSeparatedAndCond
from models.loss import ELIEILoss, ColorAlignmentLoss, NLLLoss
from data.dataset import IRRGBDataset, log_transform
from utils.utils import load_config, compute_metrics, MultiStepLR_Restart
print("ok")

# loss functions
print("[2/5] losses...", end=" ", flush=True)
cal = ColorAlignmentLoss(bins=64, downsample_size=40, sigma=0.002)
x = torch.rand(2, 3, 160, 160, requires_grad=True)
y = torch.rand(2, 3, 160, 160)
loss_cal = cal(x, y)
assert loss_cal.item() >= 0
loss_cal.backward()
assert x.grad is not None

nll = NLLLoss()
z = torch.randn(2, 3, 40, 40, requires_grad=True)
ld = torch.zeros(2)
loss_nll = nll(z, ld)
loss_nll.backward()
print(f"ok (cal={loss_cal.item():.4f}, nll={loss_nll.item():.4f})")

# encoder
print("[3/5] encoder...", end=" ", flush=True)
enc = ConditionalEncoder(in_nc=3, nf=32, nb=4, gc=32,
                         concat_histeq=True, stackRRDB_blocks=[1])
x_lr = torch.randn(1, 3, 64, 64)
fea = enc(x_lr)
assert fea['fea_up2'].shape == (1, 32, 32, 32)
assert fea['fea_up1'].shape == (1, 32, 16, 16)
assert fea['fea_up0'].shape == (1, 32, 8, 8)
print(f"ok — shapes: {', '.join(f'{k}:{list(v.shape)}' for k,v in fea.items())}")

# actnorm should be trainable (nn.Parameter, not buffer)
print("[4/5] actnorm...", end=" ", flush=True)
an = ActNorm(16)
assert isinstance(an.bias, nn.Parameter), "bias should be nn.Parameter"
assert isinstance(an.logs, nn.Parameter), "logs should be nn.Parameter"
print("ok — params are trainable")

# full model forward + reverse
print("[5/5] full model...", end=" ", flush=True)
cfg = {
    'network_G': {
        'in_nc': 3, 'out_nc': 3, 'nf': 32, 'nb': 4, 'train_RRDB': True,
        'flow': {'K': 4, 'L': 3, 'additionalFlowNoAffine': 2,
                 'stackRRDB': {'blocks': [1], 'concat': True}}},
    'concat_histeq': True,
}
model = build_model(cfg)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"\n      params: {n_params:,}")

B, C, H, W = 1, 3, 64, 64
with torch.no_grad():
    z, ld = model(torch.randn(B, C, H, W), torch.randn(B, C, H, W), reverse=False)
    assert z.shape == (B, C * 64, H // 8, W // 8)
    enh = model.enhance(torch.randn(B, C, H, W), heat=0.0)
    assert enh.shape == (B, C, H, W)
    assert enh.min() >= 0 and enh.max() <= 1
print(f"      forward: z={list(z.shape)} -> enhanced={list(enh.shape)}")

# backward pass
model.train()
z, ld = model(torch.randn(B, C, H, W), torch.randn(B, C, H, W))
loss = NLLLoss()(z, ld)
loss.backward()
grads = sum(1 for p in model.parameters() if p.grad is not None)
total = sum(1 for _ in model.parameters())
print(f"      backward: {grads}/{total} params have gradients")

print(f"\nall tests passed. model has {n_params:,} parameters.")
