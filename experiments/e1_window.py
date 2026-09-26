"""E1 + E2: the behavioural-tagging window and its ablations.

Weak training on task A, one salience pulse at offset delta from the end of A,
test after a long delay. Retention = share of A's learning that became permanent
(consolidated fraction) and test accuracy on A.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from stc.protocol import window
from stc.stc import STCConfig

torch.set_num_threads(4)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT, exist_ok=True)

DELTAS = [-450, -300, -225, -150, -75, 0, 75, 150, 225, 300, 450, 600]
CONDITIONS = {
    "full":            (STCConfig(), 5),
    "no_tags":         (STCConfig(use_tags=False), 5),
    "prp_per_synapse": (STCConfig(prp_mode="weight"), 5),
    "tau_h_75":        (STCConfig(tau_h=75), 3),
    "tau_h_300":       (STCConfig(tau_h=300), 3),
    "tau_p_40":        (STCConfig(tau_p=40), 3),
    "tau_p_160":       (STCConfig(tau_p=160), 3),
}

results = {}
t0 = time.time()
for name, (cfg, seeds) in CONDITIONS.items():
    rows = []
    for d in [None] + DELTAS:
        for s in range(seeds):
            r = window(cfg, d, seed=s)
            rows.append({k: r[k] for k in ("delta", "seed", "acc_A_after_A", "acc_A", "frac_A")})
    results[name] = rows
    print(f"{name}: done ({time.time()-t0:.0f}s)"); sys.stdout.flush()

# Size-triggered capture (no salience at all), Li & van Rossum-style comparison
rows = []
for s in range(5):
    r = window(STCConfig(size_trigger=0.002), None, seed=s)
    rows.append({k: r[k] for k in ("delta", "seed", "acc_A_after_A", "acc_A", "frac_A")})
results["size_triggered"] = rows

with open(os.path.join(OUT, "e1_window.json"), "w") as f:
    json.dump(results, f, indent=1)
print(f"saved ({time.time()-t0:.0f}s)")
