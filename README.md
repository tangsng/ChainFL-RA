# ChainFL-RA

**Blockchain-Enabled Decentralized Federated Learning with Reputation-Weighted Robust Aggregation and On-Chain Auditable Incentives**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10%2F2.11-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23438833-blue.svg)](https://doi.org/10.5281/zenodo.23438833)

Official implementation and experimental artifacts of **ChainFL-RA**, a blockchain-enabled decentralized federated learning (FL) framework that unifies robust aggregation, on-chain reputation, and token settlement under a single verifiable quality signal.

---

## Description

Federated learning concentrates three systemic risks in one aggregator: a single point of trust, vulnerability to Byzantine poisoning, and the absence of auditable credit for honest participants. Existing defenses treat these risks separately. ChainFL-RA closes the loop with one principle:

> **One verifiable quality signal drives the entire trust loop.**

The same per-update evidence — cosine similarity to a momentum-anchored robust reference, guarded by a magnitude check — simultaneously determines:

1. **Aggregation weights** (reputation-weighted robust aggregation, RAgg)
2. **On-chain reputation** (slash-only memory with slow recovery)
3. **Token settlement** (smart-contract rewards and penalties)

Every decision is committed to a hash-chained, validator-re-executable ledger. RAgg requires **neither a clean validation set nor knowledge of the number of adversaries**.

### Key results

Against four baselines (FedAvg, Krum, FedRoLA, FLAME) across MNIST, Fashion-MNIST, and CIFAR-10 under label-flipping, update-scaling, and an adaptive Min-Max attack at 10–30% malicious ratios, ChainFL-RA is **the only method that remains robust in every setting**: it matches the best baseline without attack, outperforms all baselines on the MNIST and CIFAR-10 poisoning matrices and under the Min-Max attack, and never collapses, whereas every baseline fails catastrophically in at least one setting.

- Retains **96.94%–97.36%** accuracy under update-scaling where FedAvg collapses below 9.7%.
- Attains **96.78% / 93.74%** under the Min-Max attack at 20% / 30% adversaries.
- Reaches **46.13%** on CIFAR-10 under 30% scaling vs. the best baseline's 39.86%.
- Slash precision **> 0.92** with honest false-slash rates **≤ 0.9%**.
- Audit layer costs **0.06M–1.1M gas**, 1–4 KB storage, ~2.2 s consensus latency per round.

---

## Dataset Information

Three public benchmark datasets are used. **None are stored in this repository** — all are downloaded unmodified from their original sources via `torchvision` (see [`data/README.md`](data/README.md) for details and manual setup).

| Dataset | Original source | Reference | Used for |
|---|---|---|---|
| MNIST | [http://yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist/) | LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-based learning applied to document recognition.* Proceedings of the IEEE, 86(11), 2278–2324. | Main matrix, ablations, sensitivity |
| Fashion-MNIST | [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist) | Xiao, H., Rasul, K., & Vollmer, R. (2017). *Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms.* arXiv:1708.07747. | 30% malicious-ratio cells |
| CIFAR-10 | [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html) | Krizhevsky, A. (2009). *Learning multiple layers of features from tiny images.* Technical Report, University of Toronto. | Generalization study |

---

## Code Information

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
└── scripts/
    ├── analyze_b5.py          # Table generation from raw CSVs
    ├── analyze_detection.py   # Slash precision/recall + token balance
    ├── redraw_figs_final.py   # Figure generation
    └── redraw_fig3.py
```

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 (CUDA optional but recommended), torchvision
- NumPy, SciPy, pandas, matplotlib

```bash
git clone https://github.com/tangsng/ChainFL-RA.git
cd ChainFL-RA
pip install -r requirements.txt
```

---

## Usage Instructions

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

## Methodology

Each training run executes the following pipeline (fully scripted and deterministic given the seed):

1. **Data partitioning** — the chosen dataset is subsampled (12,000 train / 2,000 test, fixed seed) and split over *N* = 10 clients, either IID (uniform random shards) or non-IID (per-class Dirichlet allocation, α = 0.5).
2. **Local training** — each client trains the shared model (LeNet-5 variant or compact 3-conv CNN) for 2 local epochs per round (SGD, lr 0.05, batch 32).
3. **Attack injection** — a bounded fraction of clients (10–30%) is turned adversarial: label-flipping, update-scaling (s = −10), or an adaptive Min-Max construction against the coordinate-wise median.
4. **Robust aggregation (RAgg)** — updates pass a dual-channel gate (cosine quality against a momentum-anchored median reference × magnitude check), then reputation-weighted tempered aggregation (Eqs. 2–6 of the paper).
5. **On-chain settlement** — a deterministic in-process simulator (`code/chain/simchain.py`) records the model hash, per-client evidence tuple, and block hash per round; a three-zone contract settles token rewards and slashes; gas, storage bytes, and consensus latency are measured per round.
6. **Analysis** — per-round CSVs are aggregated by `scripts/analyze_b5.py` and `scripts/analyze_detection.py` into the paper's tables; figures are rendered by `scripts/redraw_figs_final.py`.

### Raw data format

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

### Mapping: raw data → paper tables

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

## Citations

If you use this code or data in your research, please cite:

```bibtex
@article{tang2026chainflra,
  title   = {ChainFL-RA: Blockchain-Enabled Decentralized Federated Learning With Reputation-Weighted Robust Aggregation and On-Chain Auditable Incentives},
  author  = {Tang, Song and Jin, Zhigang and Wang, Zhiqiang},
  journal = {PeerJ Computer Science},
  year    = {2026},
  note    = {Submitted},
  doi     = {10.5281/zenodo.23438833}
}
```

Dataset references (please also cite the original sources when using MNIST, Fashion-MNIST, or CIFAR-10):

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
- Xiao, H., Rasul, K., & Vollmer, R. (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. *arXiv preprint* arXiv:1708.07747.
- Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. *Technical Report*, University of Toronto.

---

## Authors

- **Song Tang** — Tianjin University; Institute of Applied Mathematics, Hebei Academy of Sciences
- **Zhigang Jin** (corresponding) — Tianjin University
- **Zhiqiang Wang** — Institute of Applied Mathematics, Hebei Academy of Sciences

## Funding

Supported by the Hebei Provincial Science and Technology Program Project (Grant No. **25360301D**): *Research and Application Demonstration of Key Technologies for Public Data Authorization and Operation Based on Trusted Data Space*. The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript.

---

## License & Contribution Guidelines

Released under the [MIT License](LICENSE). Dataset licenses follow their respective sources.

Contributions are welcome via issues and pull requests; please open an issue describing the bug or proposed change before submitting a PR. Experiments must remain reproducible (fixed seeds, scripted pipelines).
