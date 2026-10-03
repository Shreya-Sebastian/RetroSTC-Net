"""Synaptic tagging and capture (STC) for backprop-trained networks.

Each managed parameter is split into a consolidated late component L and a labile
early component E. The optimiser updates the ordinary parameter as usual; after
every step `Consolidator.tick()` moves that update into E and runs the STC
dynamics:

    E_k <- E_k * exp(-1/tau_E)                     early-phase decay (per source k)
    h_k <- h_k * exp(-1/tau_h) + |delta_k|         synaptic tag (per source k)
    p   <- p * exp(-1/tau_p) + alpha * s(t)        plasticity-related products
    c   =  clamp(kappa * p * gate, 0, 1)           capture rate
    L  += c * E_k ;  E_k -= c * E_k                early -> late
    w   =  L + sum_k E_k                           effective weight

Sources k are learning episodes (e.g. task A, task B), labelled by the caller.
Tracking E per source makes attribution to episodes exact. Captures are labelled
with the most recent salience event, which is exact only when events do not
overlap within the PRP lifetime.

tag_binding="weight": one tag per weight (sum over sources). A weight tagged by
    any episode captures all early change it holds, whichever episode made it.
tag_binding="change": each episode's early change carries its own tag, so it is
    captured only while its own tag is set.
prp_mode="unit": PRPs are shared by all incoming weights of a unit. Every unit
    receives the same pulse, so in effect PRPs are one global signal.
prp_mode="weight": PRPs are made only at synapses tagged at event time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class STCConfig:
    tau_E: float = 1000.0
    tau_h: float = 150.0
    tau_p: float = 80.0
    kappa: float = 0.05
    alpha: float = 1.0
    tag_threshold: float = 1e-3
    use_tags: bool = True
    tag_binding: str = "weight"   # "weight" | "change"
    prp_mode: str = "unit"        # "unit" | "weight"
    size_trigger: float | None = None  # capture where |E| > value, no salience (Li & van Rossum style)


class Consolidator:
    def __init__(self, model: nn.Module, cfg: STCConfig):
        self.cfg = cfg
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.L = {n: p.detach().clone() for n, p in self.params.items()}
        self.E: dict[str, dict[str, torch.Tensor]] = {n: {} for n in self.params}
        self.h: dict[str, dict[str, torch.Tensor]] = {n: {} for n in self.params}
        self.prev = {n: p.detach().clone() for n, p in self.params.items()}
        self.prp = {n: (torch.zeros(p.shape[0]) if cfg.prp_mode == "unit" else torch.zeros_like(p))
                    for n, p in self.params.items()}
        self.dE = math.exp(-1.0 / cfg.tau_E)
        self.dh = math.exp(-1.0 / cfg.tau_h)
        self.dp = math.exp(-1.0 / cfg.tau_p)
        self.t = 0
        self.source = "idle"
        self.prp_source: str | None = None
        self._pending = 0.0
        self._pending_label: str | None = None
        # provenance: (event, source) -> {param: consolidated change}
        self.captured: dict[tuple, dict[str, torch.Tensor]] = {}
        # total early change produced per source (for consolidated fractions)
        self.produced: dict[str, dict[str, torch.Tensor]] = {}
        self.timeline: list[dict] = []

    # ------------------------------------------------------------------ control
    def set_source(self, label: str) -> None:
        self.source = label

    def salience(self, s: float = 1.0, label: str = "event") -> None:
        self._pending += s
        self._pending_label = label

    def _E_total(self, n):
        es = self.E[n]
        return sum(es.values()) if es else torch.zeros_like(self.L[n])

    def _h_total(self, n):
        hs = self.h[n]
        return sum(hs.values()) if hs else torch.zeros_like(self.L[n])

    # --------------------------------------------------------------------- step
    @torch.no_grad()
    def tick(self) -> dict:
        cfg = self.cfg
        s = self._pending
        if s > 0:
            self.prp_source = self._pending_label
        self._pending = 0.0
        cap_total, n_tagged, n_all = 0.0, 0, 0
        for n, p in self.params.items():
            delta = p.detach() - self.prev[n]
            Es, hs = self.E[n], self.h[n]
            for k in Es:
                Es[k] *= self.dE
                hs[k] *= self.dh
            if self.source != "idle" and float(delta.abs().max()) > 0:
                if self.source not in Es:
                    Es[self.source] = torch.zeros_like(p)
                    hs[self.source] = torch.zeros_like(p)
                    self.produced.setdefault(self.source, {})[n] = torch.zeros_like(p)
                Es[self.source] += delta
                hs[self.source] += delta.abs()
                self.produced[self.source].setdefault(n, torch.zeros_like(p))
                self.produced[self.source][n] += delta

            h_tot = self._h_total(n)
            tagged = h_tot > cfg.tag_threshold
            n_tagged += int(tagged.sum()); n_all += tagged.numel()

            pool = self.prp[n]
            pool *= self.dp
            if s > 0:
                pool += cfg.alpha * s * (1.0 if cfg.prp_mode == "unit" else tagged.float())

            if Es:
                if cfg.size_trigger is not None:
                    E_tot = self._E_total(n)
                    base = (E_tot.abs() > cfg.size_trigger).float() * cfg.kappa
                    rates = {k: base for k in Es}
                    ev = "size"
                else:
                    pb = pool.view(-1, *([1] * (p.dim() - 1))) if cfg.prp_mode == "unit" else pool
                    ev = self.prp_source
                    if float(pb.max()) <= 0:
                        rates = {}
                    elif not cfg.use_tags:
                        r = (cfg.kappa * pb).clamp(0, 1)
                        rates = {k: r for k in Es}
                    elif cfg.tag_binding == "weight":
                        r = (cfg.kappa * pb * tagged.float()).clamp(0, 1)
                        rates = {k: r for k in Es}
                    else:
                        rates = {k: (cfg.kappa * pb * (hs[k] > cfg.tag_threshold).float()).clamp(0, 1)
                                 for k in Es}
                for k, r in rates.items():
                    amt = r * Es[k]
                    tot = float(amt.abs().sum())
                    if tot == 0:
                        continue
                    self.L[n] += amt
                    Es[k] -= amt
                    cap_total += tot
                    rec = self.captured.setdefault((ev, k), {})
                    rec[n] = rec.get(n, torch.zeros_like(p)) + amt

            p.copy_(self.L[n] + self._E_total(n))
            self.prev[n] = p.detach().clone()

        info = {"t": self.t, "salience": s, "capture": cap_total,
                "tagged_frac": n_tagged / max(1, n_all),
                "prp": float(sum(v.mean() for v in self.prp.values()) / len(self.prp))}
        self.timeline.append(info)
        self.t += 1
        return info

    @torch.no_grad()
    def idle(self, n_steps: int) -> None:
        """Let steps pass without learning; simulate step by step while PRPs are
        present, then apply the remaining decay in one go."""
        self.set_source("idle")
        k = 0
        while k < n_steps:
            busy = self._pending > 0 or any(float(v.max()) > 1e-4 for v in self.prp.values())
            if busy:
                self.tick(); k += 1
                continue
            m = n_steps - k
            fE, fh, fp = self.dE ** m, self.dh ** m, self.dp ** m
            for n, p in self.params.items():
                for key in self.E[n]:
                    self.E[n][key] *= fE
                    self.h[n][key] *= fh
                self.prp[n] *= fp
                p.copy_(self.L[n] + self._E_total(n))
                self.prev[n] = p.detach().clone()
            self.t += m
            k = n_steps

    # --------------------------------------------------------------- analysis
    @torch.no_grad()
    def load(self, which: str = "full") -> None:
        """Put 'full' (L+E) or 'late' (L only) weights into the model."""
        for n, p in self.params.items():
            p.copy_(self.L[n] + (self._E_total(n) if which == "full" else 0))

    def consolidated(self, source: str) -> dict[str, torch.Tensor]:
        """Change from `source` that has been consolidated, summed over events."""
        out: dict[str, torch.Tensor] = {}
        for (ev, k), rec in self.captured.items():
            if k == source:
                for n, v in rec.items():
                    out[n] = out.get(n, 0) + v
        return out

    def consolidated_fraction(self, source: str) -> float:
        """Share of the episode's learning that became permanent, measured by
        projecting the consolidated change onto the episode's total change."""
        prod = self.produced.get(source, {})
        cons = self.consolidated(source)
        num = sum(float((cons[n] * prod[n]).sum()) for n in prod if n in cons)
        den = sum(float((prod[n] * prod[n]).sum()) for n in prod)
        return num / den if den > 0 else 0.0

    def provenance(self) -> list[dict]:
        rows = []
        for (ev, k), rec in self.captured.items():
            amount = sum(float(v.abs().sum()) for v in rec.values())
            if amount <= 0:
                continue
            hidden = rec.get("fc1.weight")
            top_units = []
            if hidden is not None:
                per_unit = hidden.abs().sum(1)
                top_units = [int(i) for i in per_unit.argsort(descending=True)[:5]]
                share = float((per_unit > per_unit.max() * 0.1).float().mean())
            else:
                share = 0.0
            rows.append({"event": ev, "source": k, "amount": amount,
                         "fraction": self.consolidated_fraction(k) if k in self.produced else None,
                         "top_hidden_units": top_units, "units_involved": share})
        rows.sort(key=lambda r: -r["amount"])
        return rows
