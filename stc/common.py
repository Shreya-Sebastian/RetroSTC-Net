"""Data, model and training helpers shared by all experiments."""
from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# Split-MNIST pairs; each task is a binary problem (first digit -> 0, second -> 1)
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

_cache: dict = {}


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def mnist():
    if "mnist" not in _cache:
        from torchvision import datasets
        tr = datasets.MNIST(DATA, train=True, download=True)
        te = datasets.MNIST(DATA, train=False, download=True)
        f = lambda d: ((d.data.float() / 255.0 - 0.1307) / 0.3081).view(-1, 784)
        _cache["mnist"] = (f(tr), tr.targets.clone(), f(te), te.targets.clone())
    return _cache["mnist"]


def task(pair, train=True):
    key = ("task", pair, train)
    if key not in _cache:
        xtr, ytr, xte, yte = mnist()
        x, y = (xtr, ytr) if train else (xte, yte)
        m = (y == pair[0]) | (y == pair[1])
        _cache[key] = (x[m], (y[m] == pair[1]).long())
    return _cache[key]


class MLP(nn.Module):
    """Shared two-layer trunk. `heads=1`: one shared 2-way output (domain-incremental).
    `heads=k`: a separate 2-way output head per task (task-incremental)."""

    def __init__(self, hidden=256, heads=1):
        super().__init__()
        self.heads = heads
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 2 * heads)

    def features(self, x):
        return F.relu(self.fc2(F.relu(self.fc1(x))))

    def forward(self, x, head=0):
        out = self.fc3(self.features(x))
        if self.heads == 1:
            return out
        return out[:, 2 * head: 2 * head + 2]


class Sampler:
    def __init__(self, pair, batch=64):
        self.x, self.y = task(pair, True)
        self.batch = batch

    def __call__(self):
        i = torch.randint(0, len(self.x), (self.batch,))
        return self.x[i], self.y[i]


@torch.no_grad()
def accuracy(model, pair, head=0) -> float:
    x, y = task(pair, False)
    return float((model(x, head).argmax(1) == y).float().mean())


def sgd_step(model, opt, batch, head=0, extra_loss=None) -> float:
    x, y = batch
    opt.zero_grad()
    loss = F.cross_entropy(model(x, head), y)
    total = loss + extra_loss() if extra_loss is not None else loss
    total.backward()
    opt.step()
    return loss.item()
