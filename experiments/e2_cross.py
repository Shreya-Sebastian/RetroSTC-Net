"""E2b: cross-task rescue and selectivity under interference.

Weak training on task A, an idle gap, then a novel task B whose onset is salient.
Does B's salience rescue A retroactively, and does the rescue still depend on
the gap when B's own learning also sets tags?
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from stc.protocol import cross
from stc.stc import STCConfig

torch.set_num_threads(4)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT, exist_ok=True)

GAPS = [0, 75, 150, 300, 450, 600, 900]
KEYS = ("gap", "seed", "acc_A_after_A", "acc_A_end_B", "acc_A", "acc_B", "frac_A", "frac_B")
results = {}
t0 = time.time()
for name, cfg in [("weight_tags", STCConfig()), ("change_tags", STCConfig(tag_binding="change"))]:
    rows = []
    for g in GAPS:
        for s in range(5):
            r = cross(cfg, g, seed=s)
            rows.append({k: r[k] for k in KEYS})
    results[name] = rows
    print(f"{name}: done ({time.time()-t0:.0f}s)"); sys.stdout.flush()
rows = []
for s in range(5):
    r = cross(STCConfig(), 0, seed=s, salient_b=False)
    rows.append({k: r[k] for k in KEYS})
results["no_salience"] = rows
with open(os.path.join(OUT, "e2_cross.json"), "w") as f:
    json.dump(results, f, indent=1)
print(f"saved ({time.time()-t0:.0f}s)")
