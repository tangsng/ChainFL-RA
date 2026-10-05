# -*- coding: utf-8 -*-
"""Diagnose per-client quality dynamics for chainfl under label-flip (seed 1)."""
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.nets import build_model
from data.partition import load_dataset, partition_targets, get_targets, make_client_loaders
from algorithms.local_train import local_train
from algorithms.aggregators import ReputationBank, chainfl_ra_aggregate
from utils.common import set_seed, state_to_vec, cosine, vec_sub

SEED = 1
set_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
train_ds, _ = load_dataset("mnist", "../data", 12000)
targets = get_targets(train_ds)
client_idx = partition_targets(targets, 10, "dirichlet", 0.5, seed=SEED)
loaders = make_client_loaders(train_ds, client_idx, 32)
mal = {0, 1, 2}
model = build_model("mnist").to(device)
bank = ReputationBank(list(range(10)))

for rnd in range(1, 11):
    gsd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    updates, counts = [], []
    for cid in range(10):
        gm = copy.deepcopy(model)
        lsd, n_i, _ = local_train(gm, loaders[cid], device, epochs=2, lr=0.05,
                                  attack="labelflip" if cid in mal else None)
        updates.append(vec_sub(lsd, gsd))
        counts.append(n_i)
    agg, w, st = chainfl_ra_aggregate(updates, list(range(10)), counts, bank,
                                      warmup=(rnd <= 2))
    qs = st["qualities"]
    print(f"r{rnd} q: " + " ".join(
        f"{'M' if i in mal else 'h'}{i}:{q:.2f}" for i, q in enumerate(qs)), flush=True)
    print(f"   w: " + " ".join(f"{i}:{x:.2f}" for i, x in enumerate(w))
          + " | rep: " + " ".join(f"{i}:{bank.rep[i]:.2f}" for i in range(10)), flush=True)
    model.load_state_dict({k: (gsd[k] + agg[k]).to(device) for k in gsd})
