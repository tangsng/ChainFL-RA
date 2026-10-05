# Datasets

The benchmark datasets are **not** included in this repository. They are downloaded
automatically by `torchvision` on first use, or you may place them manually under a
directory and pass it via `--data-root`.

| Dataset | torchvision class | Size | Notes |
|---|---|---|---|
| MNIST | `torchvision.datasets.MNIST` | ~11 MB | 60,000 train / 10,000 test, 28x28 grayscale, 10 classes |
| Fashion-MNIST | `torchvision.datasets.FashionMNIST` | ~30 MB | Same shape/split as MNIST |
| CIFAR-10 | `torchvision.datasets.CIFAR10` | ~170 MB | 50,000 train / 10,000 test, 32x32 RGB, 10 classes |

## Quick setup

```bash
python - <<'EOF'
from torchvision import datasets
for cls, name, root in [
    (datasets.MNIST,          "MNIST",         "data_root"),
    (datasets.FashionMNIST,   "FashionMNIST",  "data_root"),
    (datasets.CIFAR10,        "CIFAR-10",      "data_root"),
]:
    cls(root=root, download=True)
    print("ready:", name)
EOF
```

Then run experiments with `--data-root data_root`.

## Non-IID partitioning

Non-IID splits use a Dirichlet label-skew partition with concentration `alpha`
(`--partition dirichlet --alpha 0.5` by default; `alpha=0.1` for extreme skew and
`alpha=1.0` for near-IID in the sensitivity study). The partition is deterministic
given `--seed`, so runs are reproducible.

## Expected layout

```
data_root/
├── MNIST/raw/...
├── FashionMNIST/raw/...
└── cifar-10-batches-py/...
```

## Licenses

- MNIST: CC BY-SA 3.0
- Fashion-MNIST: MIT
- CIFAR-10: MIT

Please comply with the respective dataset licenses when redistributing derivatives.
