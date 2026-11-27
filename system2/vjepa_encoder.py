"""Frozen V-JEPA 2.1 image encoder (ViT-L/16, distilled from ViT-G), loaded by hand.

The hub loader in third_party/vjepa2 points at a localhost test URL, so the architecture is built here with the same
kwargs as src/hub/backbones.py::_make_vjepa2_1_model and the EMA encoder weights are loaded from the local checkpoint
(data/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt, key 'ema_encoder', strict).

Single frames go through the image path: input (B, 3, 1, H, W), ImageNet normalisation, H = W multiple of 16 (RoPE
interpolates, so 256 works as well as the native 384). Output: patch tokens (B, (H/16)*(W/16), 1024).
`pool_grid` averages the tokens over a g x g spatial grid -> (B, g*g, 1024) (g = 1 gives the global mean).
"""
from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
VJ = ROOT / 'third_party' / 'vjepa2'
CKPT = ROOT / 'data' / 'vjepa2' / 'vjepa2_1_vitl_dist_vitG_384.pt'
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


def load_encoder(ckpt=CKPT, device='cuda', dtype=torch.float32):   # fp32 weights; encode() runs under fp16 autocast
                                                                      # (RoPE upcasts q/k, so a .half() model breaks SDPA)
    if str(VJ) not in sys.path:
        sys.path.insert(0, str(VJ))
    from app.vjepa_2_1.models import vision_transformer as vit
    enc = vit.vit_large(patch_size=16, img_size=(384, 384), num_frames=64, tubelet_size=2, use_sdpa=True,
                        use_SiLU=False, wide_SiLU=True, uniform_power=False, use_rope=True,
                        img_temporal_dim_size=1, interpolate_rope=True)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)['ema_encoder']
    sd = {k.replace('module.', '').replace('backbone.', ''): v for k, v in sd.items()}
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f'V-JEPA 2.1 weights missing keys: {missing[:5]} ...')
    enc.unexpected_keys = unexpected
    return enc.to(device=device, dtype=dtype).eval().requires_grad_(False)


@torch.no_grad()
def encode(enc, frames_uint8: torch.Tensor):
    """frames_uint8: (B, H, W, 3) uint8 on any device -> tokens (B, N, 1024) in the encoder dtype."""
    p = next(enc.parameters())
    x = frames_uint8.to(p.device).permute(0, 3, 1, 2).float().div_(255).unsqueeze(2)   # (B, 3, 1, H, W)
    x = (x - MEAN.to(x.device)) / STD.to(x.device)
    with torch.autocast('cuda', dtype=torch.float16):
        return enc(x).float()


def pool_grid(tokens: torch.Tensor, g: int):
    """(B, N, D) patch tokens on a square grid -> (B, g*g, D) averaged over g x g spatial cells."""
    B, N, D = tokens.shape
    s = int(round(N ** 0.5))
    x = tokens.transpose(1, 2).reshape(B, D, s, s).float()
    return F.adaptive_avg_pool2d(x, g).flatten(2).transpose(1, 2)
