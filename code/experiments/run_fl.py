# -*- coding: utf-8 -*-
"""Main experiment runner.

Example:
  python -m experiments.run_fl --algo chainfl --dataset mnist --partition dirichlet \
      --alpha 0.5 --clients 10 --rounds 20 --attack labelflip --malicious-ratio 0.2 \
      --seed 0 --out ../../results/raw/mnist_dir_chainfl_labelflip02_s0.csv
"""
import argparse
import copy
import csv
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.nets import build_model
from data.partition import load_dataset, partition_targets, get_targets, make_client_loaders
from algorithms.local_train import local_train
from algorithms.aggregators import (weighted_average, krum_select, coordinate_median,
                                    trimmed_mean, fltrust_aggregate, rflpa_aggregate,
                                    flame_aggregate,
                                    ReputationBank, chainfl_ra_aggregate,
                                    FedRoLABank, fedrola_aggregate)
from chain.simchain import SimChain
from utils.common import set_seed, evaluate, vec_sub, vec_add_scaled, model_hash


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="fedavg",
                   choices=["fedavg", "fedprox", "scaffold", "krum", "chainfl",
                            "median", "trimmed", "fltrust", "rflpa", "flame",
                            "fedrola_lasi", "fedrola_pcsi"])
    p.add_argument("--dataset", default="mnist", choices=["mnist", "fmnist", "cifar10"])
    p.add_argument("--data-root", default="../../data")
    p.add_argument("--partition", default="iid", choices=["iid", "dirichlet"])
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--prox-mu", type=float, default=0.01)
    p.add_argument("--attack", default="none",
                   choices=["none", "labelflip", "scale", "minmax"])
    p.add_argument("--ablate", default="full",
                   choices=["full", "no_rep", "no_qual", "avg"],
                   help="ablation mode for chainfl; use --v3-opts nogate for the "
                        "v3 magnitude-gate ablation (replaces v2 no_ng)")
    p.add_argument("--malicious-ratio", type=float, default=0.0)
    p.add_argument("--scale-factor", type=float, default=-10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wcap", type=float, default=0.0,
                   help="bounded-influence cap on normalized weights (0=off), e.g. 0.15 for 1.5/N at N=10")
    p.add_argument("--normcap", type=float, default=0.0,
                   help="magnitude-channel slash threshold: slash if ||d_i|| > normcap * median||d|| (0=off)")
    p.add_argument("--eps", type=float, default=0.5,
                   help="tempering constant for trusted-client weight floor")
    p.add_argument("--lam", type=float, default=0.6,
                   help="momentum-anchor coefficient lambda in Eq. (2) (0=disable anchor)")
    p.add_argument("--warmup-w", type=int, default=2,
                   help="warm-up window W in rounds (0=disable warm-up)")
    p.add_argument("--slash-gamma", type=float, default=0.5,
                   help="reputation slash factor gamma")
    p.add_argument("--recover", type=float, default=0.05,
                   help="reputation additive recovery rate rho")
    p.add_argument("--tau-coef", type=float, default=0.25,
                   help="adaptive threshold coefficient: tau_t = max(0.05, tau_coef*median(q))")
    p.add_argument("--variant", default="v2", choices=["v2", "v3"],
                   help="RAgg variant: v2 (paper default) or v3 (spectral ref + "
                        "adaptive tempering + dual-channel product gate)")
    p.add_argument("--n-subsets", type=int, default=8,
                   help="v3 only: number of random subsets for the RSGM reference")
    p.add_argument("--subset-frac", type=float, default=0.75,
                   help="v3 only: fraction controlling RSGM subset size")
    p.add_argument("--v3-opts", default="",
                   help="v3 ablation: comma list of nospec,notemp,nodual")
    p.add_argument("--max-train", type=int, default=12000,
                   help="subsample training set for tractable runs")
    p.add_argument("--max-test", type=int, default=2000)
    p.add_argument("--out", required=True)
    p.add_argument("--chain-out", default="")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, test_ds = load_dataset(args.dataset, args.data_root, args.max_train)
    if args.max_test and len(test_ds) > args.max_test:
        g = torch.Generator().manual_seed(999)
        idx = torch.randperm(len(test_ds), generator=g)[:args.max_test].tolist()
        test_ds = torch.utils.data.Subset(test_ds, idx)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    targets = get_targets(train_ds)
    # FLTrust / RFLPA: carve out a small clean root dataset for the server before partitioning
    root_loader = None
    if args.algo in ("fltrust", "rflpa"):
        import numpy as _np
        _rng = _np.random.default_rng(12345)
        _perm = _rng.permutation(len(train_ds))
        _root_idx, _rest = _perm[:500].tolist(), _perm[500:].tolist()
        root_loader = DataLoader(torch.utils.data.Subset(train_ds, _root_idx),
                                 batch_size=args.batch_size, shuffle=True)
        train_ds = torch.utils.data.Subset(train_ds, _rest)
        targets = targets[_rest]
    client_idx = partition_targets(targets, args.clients, args.partition,
                                   args.alpha, seed=args.seed)
    loaders = make_client_loaders(train_ds, client_idx, args.batch_size)
    counts = [len(ix) for ix in client_idx]

    num_mal = int(round(args.clients * args.malicious_ratio))
    # adversaries are drawn uniformly at random per seed (not fixed indices)
    import numpy as _np
    _arng = _np.random.default_rng(args.seed * 1000 + 777)
    mal_ids = set(_arng.choice(args.clients, size=num_mal, replace=False).tolist())
    client_ids = list(range(args.clients))

    model = build_model(args.dataset).to(device)
    global_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    num_params = sum(v.numel() for v in global_sd.values())

    rep_bank = (ReputationBank(client_ids, recover=args.recover,
                               slash_gamma=args.slash_gamma)
                if args.algo == "chainfl" else None)
    fedrola_bank = (FedRoLABank(client_ids)
                    if args.algo in ("fedrola_lasi", "fedrola_pcsi") else None)
    mom_ref = None  # momentum reference direction (chainfl only)
    prev_agg = None  # server-side aggregate momentum (chainfl only)
    chain = SimChain(seed=args.seed) if args.algo == "chainfl" else None
    server_c, client_ci = None, {}
    if args.algo == "scaffold":
        server_c = {k: torch.zeros_like(v) for k, v in global_sd.items()}
        client_ci = {cid: {k: torch.zeros_like(v) for k, v in global_sd.items()}
                     for cid in client_ids}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows, chain_rows, client_rows = [], [], []
    for rnd in range(1, args.rounds + 1):
        t0 = time.perf_counter()
        updates, upd_ids, upd_counts = [], [], []
        minmax_slots = {}
        for cid in client_ids:
            is_mal = cid in mal_ids and args.attack != "none"
            attack = "labelflip" if (is_mal and args.attack == "labelflip") else None
            scaff = None
            if args.algo == "scaffold":
                scaff = (server_c, client_ci[cid])
            gm = copy.deepcopy(model)
            local_sd, n_i, new_ci = local_train(
                gm, loaders[cid], device, epochs=args.local_epochs, lr=args.lr,
                prox_mu=args.prox_mu if args.algo == "fedprox" else 0.0,
                scaffold_cv=scaff, attack=attack)
            if args.algo == "scaffold":
                client_ci[cid] = new_ci
            delta = vec_sub(local_sd, global_sd)
            if is_mal and args.attack == "scale":
                delta = {k: v * args.scale_factor for k, v in delta.items()}
            if is_mal and args.attack == "minmax":
                # craft adaptively after seeing honest updates (strongest model)
                minmax_slots[len(updates)] = delta
            updates.append(delta)
            upd_ids.append(cid)
            upd_counts.append(n_i)
        if minmax_slots:
            # Min-Max adaptive attack (AGR-style, Shejwalkar & Houmansadr):
            # adversaries observe the honest updates, estimate the benign
            # reference (coordinate-wise median), and submit median + γ·d with
            # d = -median/||median|| and γ = max honest deviation, so the
            # crafted update is distance-indistinguishable from the most
            # deviant honest one (strongest adaptive assumption).
            from algorithms.aggregators import coordinate_median as _cmed
            from utils.common import state_to_vec as _s2v
            honest = [u for j, u in enumerate(updates) if j not in minmax_slots]
            med = _cmed(honest)
            med_v = _s2v(med)
            dev_max = max(float((_s2v(u) - med_v).norm()) for u in honest)
            unit = -med_v / (med_v.norm() + 1e-12)
            for j in minmax_slots:
                offset, new_delta = 0, {}
                for k in med:
                    numel = med[k].numel()
                    seg = unit[offset:offset + numel].reshape_as(med[k])
                    new_delta[k] = med[k] + dev_max * seg
                    offset += numel
                updates[j] = new_delta

        chain_stat = {}
        if args.algo == "fedavg" or args.algo in ("fedprox", "scaffold"):
            agg, weights = weighted_average(updates, upd_counts)
        elif args.algo == "median":
            agg = coordinate_median(updates)
            weights = [1.0 / len(updates)] * len(updates)
        elif args.algo == "trimmed":
            agg = trimmed_mean(updates, trim_ratio=0.3)
            weights = [1.0 / len(updates)] * len(updates)
        elif args.algo == "fltrust":
            smodel = copy.deepcopy(model)
            server_sd, _, _ = local_train(smodel, root_loader, device,
                                          epochs=args.local_epochs, lr=args.lr)
            server_delta = vec_sub(server_sd, global_sd)
            agg, weights = fltrust_aggregate(updates, server_delta)
        elif args.algo == "rflpa":
            smodel = copy.deepcopy(model)
            server_sd, _, _ = local_train(smodel, root_loader, device,
                                          epochs=args.local_epochs, lr=args.lr)
            server_delta = vec_sub(server_sd, global_sd)
            agg, weights = rflpa_aggregate(updates, server_delta)
        elif args.algo == "flame":
            agg, weights = flame_aggregate(updates, global_sd)
        elif args.algo == "krum":
            agg, sel = krum_select(updates, num_mal)
            weights = [1.0 if i == sel else 0.0 for i in range(len(updates))]
        elif args.algo in ("fedrola_lasi", "fedrola_pcsi"):
            variant = "LASI" if args.algo == "fedrola_lasi" else "PCSI"
            agg, weights = fedrola_aggregate(updates, upd_ids, upd_counts,
                                             fedrola_bank, variant=variant)
        elif args.algo == "chainfl":
            agg, weights, st = chainfl_ra_aggregate(updates, upd_ids, upd_counts,
                                                    rep_bank, mode=args.ablate,
                                                    warmup=(rnd <= args.warmup_w),
                                                    momentum_ref=mom_ref,
                                                    lam=args.lam,
                                                    wcap=args.wcap,
                                                    normcap=args.normcap,
                                                    eps=args.eps,
                                                    tau_coef=args.tau_coef,
                                                    variant=args.variant,
                                                    num_mal=num_mal,
                                                    n_subsets=args.n_subsets,
                                                    subset_frac=args.subset_frac,
                                                    v3_opts=args.v3_opts)
            from utils.common import state_to_vec as _s2v
            agg_vec = _s2v(agg)
            mom_ref = agg_vec if mom_ref is None else args.lam * mom_ref + (1.0 - args.lam) * agg_vec
            recs = [{"client": cid, "weight": round(w, 4), "quality": round(q, 4),
                     "reputation": round(rep_bank.rep[cid], 4)}
                    for cid, w, q in zip(upd_ids, weights, st["qualities"])]
            # per-client detection log (quality/weight/reputation/slash vs ground truth)
            for cid, w, q in zip(upd_ids, weights, st["qualities"]):
                client_rows.append({
                    "round": rnd, "client": cid,
                    "malicious": int(cid in mal_ids and args.attack != "none"),
                    "attack": args.attack, "malicious_ratio": args.malicious_ratio,
                    "seed": args.seed, "ablate": args.ablate,
                    "quality": round(q, 4), "weight": round(w, 6),
                    "reputation": round(rep_bank.rep[cid], 4),
                    "slashed": int(cid in st["slashes"]),
                })

        if args.algo == "scaffold":
            # server control variate: c <- c + (1/N) * sum(ci_new - ci_old) approx.
            # simplified: recompute as average of client control variates
            server_c = {k: torch.stack([client_ci[c][k] for c in client_ids]).mean(0)
                        for k in server_c}
        global_sd = vec_add_scaled(global_sd, agg, 1.0)
        model.load_state_dict({k: v.to(device) for k, v in global_sd.items()})

        if args.algo == "chainfl":
            # commit the NEW global model hash so block t attests to w^{t+1}
            chain_stat = chain.commit_round(rnd, model_hash(global_sd), recs,
                                            st["rewards"], st["slashes"])
            chain_rows.append(chain_stat)

        acc, loss = evaluate(model, test_loader, device)
        comm = 2 * num_params * 4 * len(upd_ids)  # up + down, fp32
        row = {
            "round": rnd, "algo": args.algo, "ablate": args.ablate, "dataset": args.dataset,
            "partition": args.partition, "alpha": args.alpha, "attack": args.attack,
            "malicious_ratio": args.malicious_ratio, "seed": args.seed,
            "test_acc": round(acc, 4), "test_loss": round(loss, 4),
            "round_time_s": round(time.perf_counter() - t0, 3),
            "comm_bytes": comm,
        }
        row.update({("chain_" + k): v for k, v in chain_stat.items() if k != "round"})
        rows.append(row)
        print(f"[{args.algo}|{args.dataset}|{args.partition}|{args.attack}|"
              f"mal={args.malicious_ratio}|s={args.seed}] "
              f"round {rnd:03d} acc={acc:.4f} loss={loss:.4f}", flush=True)

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    if chain is not None and args.chain_out:
        with open(args.chain_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(chain_rows[0].keys()))
            w.writeheader()
            w.writerows(chain_rows)
        print("CHAIN_SUMMARY", chain.summary())
    if client_rows:
        clients_out = args.out.replace(".csv", "_clients.csv")
        with open(clients_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(client_rows[0].keys()))
            w.writeheader()
            w.writerows(client_rows)
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
