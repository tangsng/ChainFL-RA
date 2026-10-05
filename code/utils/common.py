# -*- coding: utf-8 -*-
"""Shared utilities: seeding, evaluation, update vector math."""
import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic kernels so identical configs reproduce exactly across runs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    crit = torch.nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += crit(out, y).item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / max(1, total), loss_sum / max(1, total)


def state_to_vec(sd):
    return torch.cat([v.detach().reshape(-1).float().cpu() for v in sd.values()])


def vec_sub(a, b):
    return {k: a[k].detach().float().cpu() - b[k].detach().float().cpu() for k in a}


def vec_add_scaled(base, delta, scale=1.0):
    return {k: base[k].detach().float().cpu() + scale * delta[k] for k in base}


def cosine(u, v, eps=1e-12):
    return float(torch.dot(u, v) / (u.norm() * v.norm() + eps))


def model_hash(sd):
    h = 0
    import hashlib
    m = hashlib.sha256()
    for k in sorted(sd.keys()):
        m.update(k.encode())
        m.update(sd[k].detach().float().cpu().numpy().tobytes())
    return m.hexdigest()
