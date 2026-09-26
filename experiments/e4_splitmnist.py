"""E4: Split-MNIST (5 binary tasks) with surprise-gated STC vs standard methods.

Methods: plain SGD, EWC, SI, STC with salience at task onset (oracle), STC with
salience from an intrinsic surprise detector (no task labels).
Scenarios: task-incremental (one head per task) and domain-incremental (shared head).
Hyperparameters of EWC, SI and STC are chosen on a separate tuning seed (100)
and then evaluated on seeds 0-4.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F

from stc.common import MLP, PAIRS, Sampler, accuracy, seed_all, sgd_step, task
from stc.stc import Consolidator, STCConfig

torch.set_num_threads(4)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
STEPS = 500
LR = 0.05


class Surprise:
    """Flags a salient moment when the loss jumps well above its running level."""

    def __init__(self, z=3.0, beta=0.02, refractory=150):
        self.m, self.v, self.z, self.beta, self.ref, self.cool, self.n = 0.0, 0.0, z, beta, refractory, 0, 0

    def __call__(self, loss):
        self.n += 1
        fired = False
        if self.n > 20 and self.cool == 0 and loss > self.m + self.z * (self.v ** 0.5 + 1e-3):
            fired = True
            self.cool = self.ref
        self.cool = max(0, self.cool - 1)
        d = loss - self.m
        self.m += self.beta * d
        self.v = (1 - self.beta) * (self.v + self.beta * d * d)
        return fired


def fisher_diag(model, pair, head, n=500):
    x, y = task(pair, True)
    idx = torch.randperm(len(x))[:n]
    fis = {k: torch.zeros_like(p) for k, p in model.named_parameters()}
    for i in idx.split(25):
        for j in i:
            model.zero_grad()
            out = F.log_softmax(model(x[j:j + 1], head), 1)
            out[0, y[j]].backward()
            for k, p in model.named_parameters():
                fis[k] += p.grad.detach() ** 2 / n
    return fis


def run(method, scenario, seed, hp):
    seed_all(seed)
    heads = 5 if scenario == "task" else 1
    model = MLP(heads=heads)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    con = Consolidator(model, STCConfig(**hp.get("stc", {}))) if method.startswith("stc") else None
    detector = Surprise() if method in ("stc_surprise", "stc_anchor") else None
    anchors = []            # EWC/SI: (importance, anchor)
    si_w = {k: torch.zeros_like(p) for k, p in model.named_parameters()}
    si_start = {k: p.detach().clone() for k, p in model.named_parameters()}
    events = []
    acc_matrix = []

    def stc_anchor_penalty():
        # importance = everything the consolidation log says was made permanent
        imp = {}
        for rec in con.captured.values():
            for k, v in rec.items():
                imp[k] = imp.get(k, 0) + v.abs()
        top = max(float(v.max()) for v in imp.values()) or 1.0
        imp = {k: v / top for k, v in imp.items()}      # normalised to max 1
        tot = 0.0
        for k, p in model.named_parameters():
            if k in imp:
                tot = tot + (imp[k] * (p - con.L[k]) ** 2).sum()
        return hp.get("lam", 0.0) * tot

    def penalty():
        tot = 0.0
        for imp, anc in anchors:
            for k, p in model.named_parameters():
                tot = tot + (imp[k] * (p - anc[k]) ** 2).sum()
        return hp.get("lam", 0.0) * tot

    for ti, pair in enumerate(PAIRS):
        head = ti if heads > 1 else 0
        smp = Sampler(pair)
        if con is not None:
            con.set_source(f"T{ti+1}")
            if method == "stc_oracle" and ti > 0:
                con.salience(1.0, f"onset T{ti+1}")
                events.append(con.t)
        for step in range(STEPS):
            old = {k: p.detach().clone() for k, p in model.named_parameters()} if method == "si" else None
            extra = penalty if anchors else None
            if method == "stc_anchor" and con.captured:
                extra = stc_anchor_penalty
            loss = sgd_step(model, opt, smp(), head=head, extra_loss=extra)
            if method == "si":
                for k, p in model.named_parameters():
                    g = p.grad.detach()
                    si_w[k] += -g * (p.detach() - old[k])
            if detector is not None and detector(loss):
                con.salience(1.0, f"surprise@{con.t}")
                events.append(con.t)
            if con is not None:
                con.tick()
        # end of task bookkeeping
        if method == "ewc":
            anchors.append((fisher_diag(model, pair, head),
                            {k: p.detach().clone() for k, p in model.named_parameters()}))
        if method == "si":
            imp = {}
            for k, p in model.named_parameters():
                d = p.detach() - si_start[k]
                imp[k] = si_w[k] / (d ** 2 + 1e-3)
                imp[k] = imp[k].clamp(min=0)
            anchors.append((imp, {k: p.detach().clone() for k, p in model.named_parameters()}))
            si_w = {k: torch.zeros_like(p) for k, p in model.named_parameters()}
            si_start = {k: p.detach().clone() for k, p in model.named_parameters()}
        acc_matrix.append([accuracy(model, PAIRS[j], j if heads > 1 else 0) for j in range(ti + 1)])
    final = acc_matrix[-1]
    return {"method": method, "scenario": scenario, "seed": seed, "hp": hp,
            "final": final, "avg": sum(final) / len(final), "matrix": acc_matrix, "events": events}


GRID = {
    "plain": [{}],
    "ewc": [{"lam": v} for v in (10.0, 30.0, 100.0, 300.0)],
    "si": [{"lam": v} for v in (0.01, 0.1, 1.0)],
    "stc_oracle": [{"stc": {"tau_E": v}} for v in (1000.0, 3000.0)],
    "stc_surprise": [{"stc": {"tau_E": v}} for v in (1000.0, 3000.0)],
    "stc_anchor": [{"lam": v} for v in (1.0, 3.0, 10.0)],
}

if __name__ == "__main__":
    t0 = time.time()
    results = {"tuning": [], "eval": []}
    for scenario in ("task", "domain"):
        for method, grid in GRID.items():
            best, best_hp = -1, None
            for hp in grid:
                r = run(method, scenario, 100, hp)
                results["tuning"].append({k: r[k] for k in ("method", "scenario", "hp", "avg")})
                if r["avg"] > best:
                    best, best_hp = r["avg"], hp
            for s in range(5):
                r = run(method, scenario, s, best_hp)
                results["eval"].append(r)
            avgs = [x["avg"] for x in results["eval"] if x["method"] == method and x["scenario"] == scenario]
            print(f"{scenario:6s} {method:13s} hp={best_hp} avg={sum(avgs)/len(avgs):.3f} ({time.time()-t0:.0f}s)")
            sys.stdout.flush()
    with open(os.path.join(OUT, "e4_splitmnist.json"), "w") as f:
        json.dump(results, f)
    print("saved")
