"""E3: consolidation as explanation.

Cross-task protocol (A, short gap, salient B, long delay). For each run we read
the memory provenance log and test whether it is faithful:

Deletion test. Rank all parameters by (a) the change the log says was
consolidated from A, (b) the same for B, (c) diagonal Fisher information for A,
(d) Fisher for B, (e) random. Reset the top-k% to their values before any
learning and measure accuracy on A and B. A faithful log should identify the
weights that hold each memory at least as well as Fisher, and dissociate A from B.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F

from stc.common import accuracy, task
from stc.protocol import TASK_A, TASK_B, cross
from stc.stc import STCConfig

torch.set_num_threads(4)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)
LEVELS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


def fisher(model, pair, head, n=1000):
    x, y = task(pair, True)
    idx = torch.randperm(len(x))[:n]
    fis = {k: torch.zeros_like(p) for k, p in model.named_parameters()}
    for i in idx.split(50):
        model.zero_grad()
        logp = F.log_softmax(model(x[i], head), 1)
        # empirical Fisher, accumulated per mini-batch of 50
        for j in range(len(i)):
            model.zero_grad()
            logp[j, y[i][j]].backward(retain_graph=True)
            for k, p in model.named_parameters():
                fis[k] += p.grad.detach() ** 2
    return fis


def flat(d, names):
    return torch.cat([d[n].flatten() for n in names])


def deletion(model, before, score, names, levels, rng):
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    init = flat(before, names)
    cur = flat(base, names)
    order = torch.argsort(score, descending=True) if score is not None else torch.from_numpy(rng.permutation(len(cur)))
    out = []
    for lv in levels:
        k = int(lv * len(cur))
        w = cur.clone()
        if k:
            w[order[:k]] = init[order[:k]]
        off = 0
        with torch.no_grad():
            for n, p in model.named_parameters():
                m = p.numel(); p.copy_(w[off:off + m].view_as(p)); off += m
        out.append({"level": lv, "acc_A": accuracy(model, TASK_A, 0), "acc_B": accuracy(model, TASK_B, 1)})
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(base[n])
    return out


results = {"deletion": [], "provenance": [], "attribution": []}
t0 = time.time()
for seed in range(5):
    r = cross(STCConfig(), 75, seed=seed, keep=True)
    model, con, before = r["model"], r["con"], r["before"]
    names = [n for n, _ in model.named_parameters()]
    cons_a, cons_b = con.consolidated("A"), con.consolidated("B")
    zero = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    s_log_a = flat({**zero, **{k: v.abs() for k, v in cons_a.items()}}, names)
    s_log_b = flat({**zero, **{k: v.abs() for k, v in cons_b.items()}}, names)
    torch.manual_seed(1000 + seed)
    s_f_a = flat(fisher(model, TASK_A, 0), names)
    s_f_b = flat(fisher(model, TASK_B, 1), names)
    rng = np.random.default_rng(seed)
    for label, sc in [("log_A", s_log_a), ("log_B", s_log_b), ("fisher_A", s_f_a),
                      ("fisher_B", s_f_b), ("random", None)]:
        for row in deletion(model, before, sc, names, LEVELS, rng):
            results["deletion"].append({"seed": seed, "ranking": label, **row})
    for row in con.provenance():
        results["provenance"].append({"seed": seed, **row})
    results["attribution"].append({k: r[k] for k in (
        "seed", "acc_A_end_B", "acc_B_end_B", "late_only_A_end_B", "late_only_B_end_B", "acc_A", "acc_B")})

    if seed == 0:
        # feature images: first-layer receptive-field changes from A that were kept vs lost
        prod = con.produced["A"]["fc1.weight"]
        kept = cons_a["fc1.weight"]
        lost = prod - kept
        k_units = kept.norm(dim=1).argsort(descending=True)[:6]
        l_units = (lost.norm(dim=1) - kept.norm(dim=1)).argsort(descending=True)[:6]
        results["features"] = {
            "kept_units": [int(u) for u in k_units],
            "lost_units": [int(u) for u in l_units],
            "kept": [kept[u].view(28, 28).tolist() for u in k_units],
            "lost": [prod[u].view(28, 28).tolist() for u in l_units],
        }
        results["timeline_seed0"] = con.timeline[::5]
    print(f"seed {seed} done ({time.time()-t0:.0f}s)"); sys.stdout.flush()

with open(os.path.join(OUT, "e3_explain.json"), "w") as f:
    json.dump(results, f)
print("saved")
