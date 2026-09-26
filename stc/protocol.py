"""Behavioural-tagging protocols.

Time is measured in steps. After training, every protocol ends with a long idle
`delay` (the analogue of testing a day later): early-phase changes have then
decayed, so what remains is what was consolidated.

window(): weak training on task A; one salience pulse at offset `delta` from the
          END of A (negative = before, positive = after); test A after the delay.
cross():  weak training on task A, an idle gap, then training on a novel task B
          whose onset is salient; test A and B after the delay.
"""
from __future__ import annotations

import torch

from .common import MLP, Sampler, accuracy, seed_all, sgd_step
from .stc import Consolidator, STCConfig

TASK_A, TASK_B = (3, 5), (4, 9)   # two hard binary problems


def _snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def window(cfg: STCConfig, delta, seed=0, pre=700, n_a=60, delay=4000, lr=0.05, keep=False):
    seed_all(seed)
    model = MLP(heads=1)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    con = Consolidator(model, cfg)
    sa = Sampler(TASK_A)
    end_a = pre + n_a
    t_event = None if delta is None else end_a + delta
    horizon = max(end_a, (t_event or 0) + 1)
    before = None
    acc_after = None
    for t in range(horizon):
        if t_event is not None and t == t_event:
            con.salience(1.0, "novelty")
        if pre <= t < end_a:
            if t == pre:
                before = _snapshot(model)
            con.set_source("A")
            sgd_step(model, opt, sa())
        else:
            con.set_source("idle")
        con.tick()
        if t == end_a - 1:
            acc_after = accuracy(model, TASK_A)
    con.idle(delay)
    out = {"delta": delta, "seed": seed, "acc_A_after_A": acc_after, "acc_A": accuracy(model, TASK_A),
           "frac_A": con.consolidated_fraction("A")}
    if keep:
        out.update(model=model, con=con, before=before)
    return out


def cross(cfg: STCConfig, gap, seed=0, n_a=60, n_b=400, delay=4000, lr=0.05,
          salient_b=True, keep=False):
    seed_all(seed)
    model = MLP(heads=2)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    con = Consolidator(model, cfg)
    sa, sb = Sampler(TASK_A), Sampler(TASK_B)
    before = _snapshot(model)
    con.set_source("A")
    for _ in range(n_a):
        sgd_step(model, opt, sa(), head=0); con.tick()
    acc_after = accuracy(model, TASK_A, 0)
    con.set_source("idle")
    for _ in range(gap):
        con.tick()
    if salient_b:
        con.salience(1.0, "B onset")
    con.set_source("B")
    for _ in range(n_b):
        sgd_step(model, opt, sb(), head=1); con.tick()
    acc_a_end_b = accuracy(model, TASK_A, 0)
    acc_b_end_b = accuracy(model, TASK_B, 1)
    con.load("late")                       # memory-source attribution: long-term part only
    late_a, late_b = accuracy(model, TASK_A, 0), accuracy(model, TASK_B, 1)
    con.load("full")
    con.set_source("idle")
    con.idle(delay)
    out = {"gap": gap, "seed": seed, "acc_A_after_A": acc_after, "acc_A_end_B": acc_a_end_b,
           "acc_B_end_B": acc_b_end_b, "late_only_A_end_B": late_a, "late_only_B_end_B": late_b,
           "acc_A": accuracy(model, TASK_A, 0), "acc_B": accuracy(model, TASK_B, 1),
           "frac_A": con.consolidated_fraction("A"), "frac_B": con.consolidated_fraction("B")}
    if keep:
        out.update(model=model, con=con, before=before)
    return out
