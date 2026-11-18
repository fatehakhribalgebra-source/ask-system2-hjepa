"""Initialise an H-JEPA from a pretrained checkpoint whose level-1 action space is smaller.

Used to fine-tune the FourRoom H-JEPA (2-d move actions) on Crafter (14-d action vectors whose first two
dims are the same image-space move vector). Every tensor with a matching shape is copied. The level-1
action embedder's input layers are copied into the movement slots of each frameskip step; new action
channels pass through unchanged (patch_embed) and start with small random weights (embed.0).
"""
from __future__ import annotations

import torch
from loguru import logger as logging


def _map_cols(old_dim: int, new_dim: int, frameskip: int) -> list[tuple[int, int]]:
    """(old_index, new_index) for the first old_dim channels of each step (step-major layout)."""
    return [(s * old_dim + d, s * new_dim + d) for s in range(frameskip) for d in range(old_dim)]


@torch.no_grad()
def init_from_pretrained(model, ckpt_path: str, old_action_dim: int, new_action_dim: int, frameskip: int):
    src = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    src_sd, dst_sd = src.state_dict(), model.state_dict()
    copied, skipped = 0, []
    for k, v in dst_sd.items():
        if k in src_sd and src_sd[k].shape == v.shape:
            v.copy_(src_sd[k]); copied += 1
        else:
            skipped.append(k)
    pairs = _map_cols(old_action_dim, new_action_dim, frameskip)
    pre = 'jepas.0.action_embed.'
    # patch_embed: Conv1d(C_in, C_out, 1). Output width is unchanged (the rest of the embedder is copied as-is);
    # pretrained input columns go to the movement slots of each step, new input columns start small.
    w_key, b_key = pre + 'patch_embed.weight', pre + 'patch_embed.bias'
    if w_key in dst_sd and w_key in src_sd:
        W, Wo = dst_sd[w_key], src_sd[w_key]
        if W.shape[0] == Wo.shape[0]:
            W.normal_(0.0, 0.01)
            for oi, ni in pairs:
                W[:, ni, :] = Wo[:, oi, :]
            dst_sd[b_key].copy_(src_sd[b_key])
            skipped = [k for k in skipped if k not in (w_key, b_key)]
    # embed.0: Linear(C, H) -> pretrained columns for movement, small random for the rest
    e_key = pre + 'embed.0.weight'
    if e_key in dst_sd and e_key in src_sd and dst_sd[e_key].shape != src_sd[e_key].shape:
        W, Wo = dst_sd[e_key], src_sd[e_key]
        W.normal_(0.0, 0.01)
        for oi, ni in pairs:
            W[:, ni] = Wo[:, oi]
        dst_sd[pre + 'embed.0.bias'].copy_(src_sd[pre + 'embed.0.bias'])
        skipped = [k for k in skipped if k not in (e_key, pre + 'embed.0.bias')]
    model.load_state_dict(dst_sd)
    logging.info(f'init_from_pretrained: copied {copied} tensors from {ckpt_path}; '
                 f'action layers remapped ({old_action_dim}->{new_action_dim} x{frameskip}); '
                 f'left at init: {skipped}')
    return model
