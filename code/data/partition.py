# -*- coding: utf-8 -*-
"""Dataset loading and IID / Non-IID (Dirichlet) partitioning."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def load_dataset(name: str, root: str, max_train: int = None):
    tf = transforms.Compose([transforms.ToTensor()])
    if name == "mnist":
        train = datasets.MNIST(root, train=True, download=True, transform=tf)
        test = datasets.MNIST(root, train=False, download=True, transform=tf)
    elif name == "fmnist":
        train = datasets.FashionMNIST(root, train=True, download=True, transform=tf)
        test = datasets.FashionMNIST(root, train=False, download=True, transform=tf)
    elif name == "cifar10":
        train = datasets.CIFAR10(root, train=True, download=True, transform=tf)
        test = datasets.CIFAR10(root, train=False, download=True, transform=tf)
    else:
        raise ValueError(name)
    if max_train is not None and len(train) > max_train:
        g = torch.Generator().manual_seed(1234)
        idx = torch.randperm(len(train), generator=g)[:max_train].tolist()
        train = Subset(train, idx)
    return train, test


def partition_targets(targets: np.ndarray, num_clients: int, mode: str,
                      alpha: float = 0.5, seed: int = 0):
    """Return list of index lists, one per client."""
    rng = np.random.default_rng(seed)
    n = len(targets)
    if mode == "iid":
        idx = rng.permutation(n)
        shards = np.array_split(idx, num_clients)
        return [s.tolist() for s in shards]
    if mode == "dirichlet":
        classes = np.unique(targets)
        client_idx = [[] for _ in range(num_clients)]
        for c in classes:
            c_idx = np.where(targets == c)[0]
            rng.shuffle(c_idx)
            props = rng.dirichlet([alpha] * num_clients)
            cuts = (np.cumsum(props) * len(c_idx)).astype(int)[:-1]
            for i, shard in enumerate(np.split(c_idx, cuts)):
                client_idx[i].extend(shard.tolist())
        for i in range(num_clients):
            rng.shuffle(client_idx[i])
        return client_idx
    raise ValueError(mode)


def get_targets(ds) -> np.ndarray:
    if isinstance(ds, Subset):
        base = ds.dataset
        t = base.targets if hasattr(base, "targets") else base.labels
        return np.asarray(t)[ds.indices]
    t = ds.targets if hasattr(ds, "targets") else ds.labels
    return np.asarray(t)


def make_client_loaders(train_ds, client_indices, batch_size=32):
    loaders = []
    for idx in client_indices:
        loaders.append(DataLoader(Subset(train_ds, idx), batch_size=batch_size,
                                  shuffle=True, drop_last=False))
    return loaders
