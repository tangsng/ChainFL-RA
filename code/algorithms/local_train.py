# -*- coding: utf-8 -*-
"""Client-side local training for FedAvg / FedProx / SCAFFOLD (shared engine)."""
import copy

import torch


def local_train(global_model, loader, device, epochs=1, lr=0.01, momentum=0.0,
                prox_mu=0.0, scaffold_cv=None, attack=None, num_classes=10):
    """Train locally and return (local_state_dict, num_samples, new_client_cv).

    prox_mu > 0 enables the FedProx proximal term.
    scaffold_cv = (server_c, client_ci) as dicts of tensors enables SCAFFOLD.
    attack = "labelflip" flips labels (y -> num_classes-1-y) during training.
    """
    model = copy.deepcopy(global_model).to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    crit = torch.nn.CrossEntropyLoss()
    global_params = [p.detach().clone() for p in global_model.parameters()]
    steps = 0
    n_samples = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if attack == "labelflip":
                y = (num_classes - 1) - y
            opt.zero_grad()
            loss = crit(model(x), y)
            if prox_mu > 0:
                prox = sum(((p - g) ** 2).sum()
                           for p, g in zip(model.parameters(), global_params))
                loss = loss + 0.5 * prox_mu * prox
            loss.backward()
            if scaffold_cv is not None:
                c, ci = scaffold_cv
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        p.grad += (c[name].to(device) - ci[name].to(device))
            opt.step()
            steps += 1
            n_samples += y.numel()
    new_ci = None
    if scaffold_cv is not None:
        c, ci = scaffold_cv
        new_ci = {}
        with torch.no_grad():
            for (name, p), g in zip(model.named_parameters(), global_model.parameters()):
                g = g.to(device)
                # SCAFFOLD option II control-variate update
                new_ci[name] = ci[name].to(device) - c[name].to(device) + \
                    (g - p) / max(1, steps) / lr
                new_ci[name] = new_ci[name].detach().cpu()
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}, n_samples, new_ci
