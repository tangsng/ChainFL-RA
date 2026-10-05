# ChainFL-RA

**Blockchain-Enabled Decentralized Federated Learning with Reputation-Weighted Robust Aggregation and On-Chain Auditable Incentives**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10%2F2.11-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/IEEE%20Access-submitted-red.svg)](paper/)

Official implementation and experimental artifacts of **ChainFL-RA**, a blockchain-enabled decentralized federated learning (FL) framework that unifies robust aggregation, on-chain reputation, and token settlement under a single verifiable quality signal.

---

## Overview

Federated learning concentrates three systemic risks in one aggregator: a single point of trust, vulnerability to poisoning, and the absence of auditable credit for honest participants. Existing defenses treat these risks separately. ChainFL-RA closes the loop with one principle:

> **One verifiable quality signal drives the entire trust loop.**

The same per-update evidence — cosine similarity to a momentum-anchored robust reference, guarded by a magnitude check — simultaneously determines:
1. **Aggregation weights** (reputation-weighted robust aggregation, RAgg)
2. **On-chain reputation** (slash-only memory with slow recovery)
3. **Token settlement** (smart-contract rewards and penalties)

Every decision is committed to a hash-chained, validator-re-executable ledger. RAgg requires **neither a clean validation set nor knowledge of the number of adversaries**.

### Key results

Against four baselines (FedAvg, Krum, FedRoLA, FLAME) across MNIST, Fashion-MNIST, and CIFAR-10 under label-flipping, update-scaling, and an adaptive Min-Max attack at 10–30% malicious ratios:

- Statistically **tied with the best baseline without attack**, and **leads 15 of 19 comparison cells**, including every MNIST poisoning cell and all seven CIFAR-10 cells.
- Retains **96.94%–97.36%** under update-scaling where FedAvg collapses below 9.7%.
- Attains **96.78% / 93.74%** under the Min-Max attack at 20% / 30% adversaries.
- Reaches **46.13%** on CIFAR-10 under 30% scaling vs. the best baseline's 39.86%.
- Slash precision **> 0.92** with honest false-slash rates **≤ 0.9%**.
- Audit layer costs **0.06M–1.1M gas**, 1–4 KB storage, ~2.2 s consensus latency per round.

---

## Repository structure

```
ChainFL-RA/
├── README.md                  # This file
├── LICENSE                    # MIT License
├── .gitignore
├── requirements.txt           # Python dependencies
├── data/
│   └── README.md              # Dataset preparation (auto-download; NOT stored here)
├── code/
│   ├── algorithms/            # Aggregation rules: RAgg + baselines
│   │   ├── aggregators.py     #   Krum, median, trimmed, FedRoLA, FLAME, RAgg (v2/v3)
│   │   └── local_train.py     #   Local client training
│   ├── chain/
│   │   └── simchain.py        # Deterministic on-chain simulator (hash chain, gas, latency)
│   ├── data/
│   │   └── partition.py       # IID / Dirichlet non-IID partitioning
│   ├── models/                # CNN / MLP models
│   ├── utils/                 # Metrics, logging helpers
│   ├── experiments/
│   │   └── run_fl.py          # Main entry point (single training run)
│   └── runners/
│       ├── jobs_b5.txt        # 432 jobs — main accuracy matrix (Tables 1–5, 9, 10)
│       ├── jobs_abl.txt       # 146 jobs — ablations + sensitivity (Tables 7, 8)
│       ├── jobs_cifar.txt     # 115 jobs — CIFAR-10 generalization (Table 9)
│       ├── b5_runner.sh       # Parallel driver (GNU parallel / xargs)
│       ├── abl_runner.sh
│       └── cifar_runner.sh
├── results/
│   ├── raw_b5/                # 528 CSVs — main experiment raw outputs
│   ├── raw_abl/               # 268 CSVs — ablation + sensitivity raw outputs
│   ├── raw_cifar/             # CIFAR-10 raw outputs
│   ├── b5_summary.csv         # Aggregated summary of the main matrix
│   ├── tables/                # Rendered tables (CSV + LaTeX) used in the paper
│   └── figures/               # All 6 paper figures (PNG, 300 dpi)
├── scripts/
│   ├── analyze_b5.py          # Table generation from raw CSVs
│   ├── analyze_detection.py   # Slash precision/recall + token balance
│   ├── redraw_figs_final.py   # Figure generation
│   └── redraw_fig3.py
└── paper/                     # IEEE Access LaTeX source (see below)
```

---

## Installation

```bash
git clone https://github.com/tangsng/ChainFL-RA.git
cd ChainFL-RA
pip install -r requirements.txt
```

Requirements: Python ≥ 3.10, PyTorch ≥ 2.0 (CUDA optional but recommended), torchvision, NumPy, SciPy, pandas, matplotlib.

---

## Datasets

Datasets are **not** stored in this repository (they are downloaded automatically by torchvision on first use). See [`data/README.md`](data/README.md).

| Dataset | Source | Used for |
|---|---|---|
| MNIST | `torchvision.datasets.MNIST` | Main matrix, ablations, sensitivity |
| Fashion-MNIST | `torchvision.datasets.FashionMNIST` | 30% malicious-ratio cells |
| CIFAR-10 | `torchvision.datasets.CIFAR10` | Generalization study |

Set `--data-root` to the directory where datasets are (or will be) stored.

---

## Reproducing the experiments

### 1. A single run

```bash
cd code
python3 experiments/run_fl.py \
  --dataset mnist --data-root data_root \
  --partition dirichlet --alpha 0.5 \
  --clients 10 --rounds 25 --local-epochs 2 \
  --lr 0.05 --batch-size 32 \
  --attack labelflip --malicious-ratio 0.3 \
  --algo chainfl --variant v3 --seed 0 \
  --warmup-w 2 --lam 0.6 --normcap 5.0 --eps 1.0 --tau-coef 0.25 \
  --max-train 12000 --max-test 2000 \
  --out ../results/raw_b5/demo_s0.csv \
  --chain-out ../results/raw_b5/demo_s0_chain.csv
```

Key options:

| Option | Choices | Meaning |
|---|---|---|
| `--algo` | `fedavg, fedprox, scaffold, krum, chainfl, median, trimmed, fltrust, rflpa, flame, fedrola_lasi` | Aggregation rule |
| `--variant` | `v2, v3` | ChainFL-RA variant (`v3` = paper default, dual-channel product gate) |
| `--attack` | `none, labelflip, scale, minmax` | Attack family |
| `--ablate` | `full, no_rep, no_qual, avg` | Ablation mode |
| `--v3-opts` | `nogate, notemp, nodual, nospec, noanchor` | Fine-grained v3 ablations |

### 2. Full matrix (reproduces all paper tables)

```bash
cd code
bash runners/b5_runner.sh      # 432 jobs -> ../results/raw_b5
bash runners/abl_runner.sh     # 146 jobs -> ../results/raw_abl
bash runners/cifar_runner.sh   # 115 jobs -> ../results/raw_cifar
```

Each runner is idempotent: already-existing non-empty outputs are skipped, so runs can be safely resumed.

> **Compute note:** The full suite (~693 jobs) was executed on NVIDIA V100 (32 GB) GPUs. On a single modern GPU expect several hours; the drivers parallelize across devices (`CUDA_VISIBLE_DEVICES` is assigned per job).

### 3. Regenerate tables and figures

```bash
cd scripts
python3 analyze_b5.py          # -> ../results/tables/*.csv and *.tex
python3 analyze_detection.py   # slash precision / recall, token balance
python3 redraw_figs_final.py   # -> ../results/figures/*.png
python3 redraw_fig3.py         # Fig. 3 (attack robustness bars)
```

---

## Raw data format

Each run writes a CSV with per-round records:

| Column | Description |
|---|---|
| `round` | Communication round index |
| `algo`, `dataset`, `partition`, `attack`, `malicious_ratio`, `seed` | Configuration |
| `test_acc`, `test_loss` | Global model accuracy / loss |
| `adversarial_weight` | Total aggregation weight assigned to malicious clients |

Companion files:
- `*_chain.csv` — on-chain audit records (block hashes, gas, storage bytes, latency, settlement events)
- `*_clients.csv` — per-client quality, weight, reputation, slash flags (used for Tables 4–5)

---

## Mapping: raw data → paper tables

| Paper table | Source data |
|---|---|
| Table 1 (attack-free) | `results/raw_b5` (attack = `none`) |
| Table 2 (poisoning) | `results/raw_b5` (`labelflip`, `scale`) |
| Table 3 (Wilcoxon tests) | computed by `scripts/analyze_b5.py` |
| Table 4 (Min-Max) | `results/raw_b5` (`minmax`) |
| Table 5 (slash detection) | `*_clients.csv` in `results/raw_b5` |
| Table 6 (token balance) | `*_clients.csv` |
| Table 7 (ablation) | `results/raw_abl` |
| Table 8 (sensitivity) | `results/raw_abl` (sweep variants) |
| Table 9 (CIFAR-10) | `results/raw_cifar` |
| Table 10 (overhead) | `*_chain.csv` (gas model) |
| Table 11 (wall-clock) | run logs |

---

## Paper

The manuscript (IEEE Access format, LaTeX) is in [`paper/`](paper/) together with the compiled PDF.

```bibtex
@article{tang2026chainflra,
  title   = {ChainFL-RA: Blockchain-Enabled Decentralized Federated Learning With Reputation-Weighted Robust Aggregation and On-Chain Auditable Incentives},
  author  = {Tang, Song and Jin, Zhigang and Wang, Zhiqiang},
  journal = {IEEE Access},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.0000000}
}
```

---

## Authors

- **Song Tang** — Tianjin University; Institute of Applied Mathematics, Hebei Academy of Sciences
- **Zhigang Jin** (corresponding) — Tianjin University
- **Zhiqiang Wang** — Institute of Applied Mathematics, Hebei Academy of Sciences

---

## Acknowledgments and funding

Supported by the Hebei Provincial Science and Technology Program Project (Grant No. **25360301D**): *Research and Application Demonstration of Key Technologies for Public Data Authorization and Operation Based on Trusted Data Space*.

---

## License

Released under the [MIT License](LICENSE). Dataset licenses follow their respective sources.
