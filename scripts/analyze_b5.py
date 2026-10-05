# -*- coding: utf-8 -*-
"""Analyze b5 (5-baseline + ChainFL-RA v3) full matrix -> paper tables."""
import csv
import glob
import os
import statistics as st
from collections import defaultdict

RAW = "raw_b5"
ALGOS = ["fedavg", "krum", "fedrola_lasi", "flame", "rflpa", "chainfl"]
LABEL = {"fedavg": "FedAvg", "krum": "Krum", "fedrola_lasi": "FedRoLA",
         "flame": "FLAME", "rflpa": "RFLPA", "chainfl": "ChainFL-RA"}


def finals(pattern, seeds):
    out = {}
    for algo in ALGOS:
        vals = []
        for s in seeds:
            fs = glob.glob(os.path.join(RAW, pattern.format(algo=algo, s=s)))
            fs = [f for f in fs if not f.endswith("_chain.csv")
                  and not f.endswith("_clients.csv")]
            if not fs:
                continue
            rows = list(csv.DictReader(open(fs[0])))
            vals.append(float(max(rows, key=lambda r: int(r["round"]))["test_acc"]) * 100)
        if vals:
            out[algo] = (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0, len(vals))
    return out


def fmt(d, algo):
    m, s, n = d[algo]
    return f"{m:.2f} ± {s:.2f}"


def best_baseline(d):
    return max((d[a][0] for a in d if a != "chainfl"), default=0.0)


def table(title, d, seeds_label):
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    hdr = "".join(f"{LABEL[a]:>18}" for a in ALGOS)
    print(hdr)
    print(f"{'ChainFL-RA':>18}{fmt(d,'chainfl')}")
    for a in ALGOS:
        if a == "chainfl":
            continue
        print(f"{LABEL[a]:>18}{fmt(d, a)}")
    bb = best_baseline(d)
    ours = d["chainfl"][0]
    print(f"\n  best baseline = {bb:.2f}   ChainFL-RA = {ours:.2f}   margin = {ours-bb:+.2f}")


# ---- Table I: attack-free (3 seeds) ----
for ds in ["mnist", "fmnist"]:
    for part in ["iid", "dirichlet"]:
        d = finals(f"{ds}_{part}_{{algo}}_s{{s}}.csv", [0, 1, 2])
        table(f"Table I · {ds.upper()} {part}", d, "3 seeds")

# ---- Table II: poisoning MNIST dirichlet (5 seeds) ----
for atk in ["labelflip", "scale"]:
    for mr in ["0.1", "0.2", "0.3"]:
        d = finals(f"mnist_{atk}_m{mr}_{{algo}}_s{{s}}.csv", [0, 1, 2, 3, 4])
        table(f"Table II · {atk} {float(mr)*100:.0f}%", d, "5 seeds")

# ---- Table III: minmax (5 seeds) ----
for mr in ["0.2", "0.3"]:
    d = finals(f"mnist_minmax_m{mr}_{{algo}}_s{{s}}.csv", [0, 1, 2, 3, 4])
    table(f"Table III · minmax {float(mr)*100:.0f}%", d, "5 seeds")

# ---- Table VII: generalization (5 seeds) ----
for atk in ["labelflip", "scale"]:
    d = finals(f"fmnist_{atk}_m0.3_{{algo}}_s{{s}}.csv", [0, 1, 2, 3, 4])
    table(f"Table VII · FMNIST {atk} 30%", d, "5 seeds")
for mr in ["0.1", "0.2"]:
    d = finals(f"cifar10_labelflip_m{mr}_{{algo}}_s{{s}}.csv", [0, 1, 2, 3, 4])
    table(f"Table VII · CIFAR-10 labelflip {float(mr)*100:.0f}%", d, "5 seeds")
