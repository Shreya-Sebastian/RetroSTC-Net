"""Unit tests for the STC consolidator. Run: python -m pytest tests  (or python tests/test_stc.py)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn

from stc.stc import Consolidator, STCConfig


def make(cfg=None):
    torch.manual_seed(0)
    m = nn.Linear(4, 3)
    return m, Consolidator(m, cfg or STCConfig(tau_E=100, tau_h=20, tau_p=10, kappa=0.2, tag_threshold=1e-3))


def learn(m, con, source="A", size=0.05, steps=5):
    con.set_source(source)
    for _ in range(steps):
        with torch.no_grad():
            m.weight += size
        con.tick()


def test_decays_without_salience():
    m, con = make()
    w0 = m.weight.detach().clone()
    learn(m, con)
    con.idle(2000)
    assert torch.allclose(m.weight, w0, atol=1e-6), "early changes must fade without an event"
    assert con.consolidated_fraction("A") == 0.0


def test_event_during_tag_consolidates():
    m, con = make()
    w0 = m.weight.detach().clone()
    learn(m, con)
    con.salience(1.0, "ev"); con.idle(2000)
    kept = (m.weight - w0).mean().item()
    assert kept > 0.1, kept
    assert con.consolidated_fraction("A") > 0.5


def test_event_after_tag_fades_does_nothing():
    m, con = make()
    learn(m, con)
    con.idle(400)                        # tag (tau 20) long gone, E (tau 100) mostly gone
    con.salience(1.0, "late"); con.idle(2000)
    assert con.consolidated_fraction("A") < 0.01


def test_per_synapse_prp_needs_tag_at_event_time():
    m, con = make(STCConfig(tau_E=100, tau_h=20, tau_p=10, kappa=0.2, prp_mode="weight"))
    con.salience(1.0, "early"); con.tick()   # event before any learning
    learn(m, con); con.idle(2000)
    assert con.consolidated_fraction("A") == 0.0


def test_provenance_is_exact():
    m, con = make()
    w0 = m.weight.detach().clone()
    learn(m, con, "A"); learn(m, con, "B", size=-0.02)
    con.salience(1.0, "ev"); con.idle(3000)
    total = sum(con.consolidated(s)["weight"] for s in ("A", "B"))
    assert torch.allclose(m.weight - w0, total, atol=1e-5), "consolidated changes must add up to the weight change"


def test_bulk_idle_matches_stepwise():
    m1, c1 = make(); m2, c2 = make()
    learn(m1, c1); learn(m2, c2)
    c1.idle(500)
    c2.set_source("idle")
    for _ in range(500):
        c2.tick()
    assert torch.allclose(m1.weight, m2.weight, atol=1e-6)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
