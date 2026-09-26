"""Turn results/*.json into figures/*.png, results/summary.md and demo/index.html."""
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES, FIG, DEMO = (os.path.join(ROOT, d) for d in ("results", "figures", "demo"))
os.makedirs(FIG, exist_ok=True); os.makedirs(DEMO, exist_ok=True)

TAG, PRP, LATE, GRAY, INK = "#12906A", "#B42E86", "#2B4C8C", "#8A94A3", "#16202C"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#9AA3AE", "axes.labelcolor": INK,
                     "xtick.color": "#4A5563", "ytick.color": "#4A5563", "figure.dpi": 150,
                     "savefig.bbox": "tight", "legend.frameon": False})


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def agg(rows, key, by):
    xs = sorted({r[by] for r in rows if r[by] is not None})
    m = [st.mean(r[key] for r in rows if r[by] == x) for x in xs]
    s = [st.pstdev([r[key] for r in rows if r[by] == x]) for x in xs]
    return np.array(xs), np.array(m), np.array(s)


def none_mean(rows, key, by):
    v = [r[key] for r in rows if r[by] is None]
    return st.mean(v) if v else None


summary = ["# Results summary", ""]
demo = {}

# ---------------------------------------------------------------- E1 window
e1 = load("e1_window.json")
if e1:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for ax, key, lab in [(axes[0], "frac_A", "Share of task A consolidated"),
                         (axes[1], "acc_A", "Task A accuracy after delay")]:
        for name, col, ls, label in [("full", TAG, "-", "STC (tags + shared PRPs)"),
                                     ("no_tags", GRAY, "--", "No tags"),
                                     ("prp_per_synapse", PRP, ":", "PRPs per synapse")]:
            x, m, s = agg(e1[name], key, "delta")
            ax.plot(x, m, ls, color=col, lw=2, marker="o", ms=3.5, label=label)
            ax.fill_between(x, m - s, m + s, color=col, alpha=0.12, lw=0)
        base = none_mean(e1["full"], key, "delta")
        ax.axhline(base, color=INK, lw=1, ls=(0, (1, 3)))
        ax.text(x[-1], base, " no event", va="bottom", ha="right", fontsize=9, color=INK)
        ax.axvline(0, color="#C9CFD7", lw=1)
        ax.set_xlabel("Salient event time relative to end of task A (steps)")
        ax.set_ylabel(lab)
    size = st.mean(r["frac_A"] for r in e1["size_triggered"])
    axes[0].axhline(size, color=LATE, lw=1.2, ls="-.")
    axes[0].text(600, size, "size-triggered capture, no event", va="bottom", ha="right", fontsize=8.5, color=LATE)
    axes[0].legend(loc="upper left", fontsize=8.5, bbox_to_anchor=(0, 1.02))
    axes[1].set_ylim(0.4, 1.0)
    fig.suptitle("E1  Behavioural tagging: a later (or earlier) salient event decides what task A keeps",
                 x=0.01, ha="left", fontsize=11.5)
    fig.savefig(os.path.join(FIG, "fig1_window.png")); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=True)
    for ax, names, title in [(axes[0], [("tau_h_75", "τh = 75"), ("full", "τh = 150"), ("tau_h_300", "τh = 300")],
                              "Tag lifetime τh sets how long AFTER learning an event still works"),
                             (axes[1], [("tau_p_40", "τp = 40"), ("full", "τp = 80"), ("tau_p_160", "τp = 160")],
                              "PRP lifetime τp sets how long BEFORE learning an event still works")]:
        shades = ["#9FD8C3", TAG, "#0B5A43"] if "tau_h" in names[0][0] else ["#E8A9D1", PRP, "#6E1A52"]
        for (n, lab), c in zip(names, shades):
            x, m, _ = agg(e1[n], "frac_A", "delta")
            ax.plot(x, m, color=c, lw=2, marker="o", ms=3, label=lab)
        ax.axvline(0, color="#C9CFD7", lw=1)
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_xlabel("Salient event time relative to end of task A (steps)")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Share of task A consolidated")
    fig.savefig(os.path.join(FIG, "fig2_time_constants.png")); plt.close(fig)

    def w(name, d):
        return st.mean(r["frac_A"] for r in e1[name] if r["delta"] == d)
    summary += ["## E1: behavioural-tagging window (5 seeds; consolidated share of task A)", "",
                "| event offset | STC | no tags | PRPs per synapse |", "|---|---|---|---|",
                f"| none | {none_mean(e1['full'],'frac_A','delta'):.2f} | {none_mean(e1['no_tags'],'frac_A','delta'):.2f} | {none_mean(e1['prp_per_synapse'],'frac_A','delta'):.2f} |"]
    for d in [-300, -150, 0, 150, 300, 600]:
        summary.append(f"| {d:+d} | {w('full', d):.2f} | {w('no_tags', d):.2f} | {w('prp_per_synapse', d):.2f} |")
    summary += ["", f"Size-triggered capture without any event: {size:.2f}.", ""]
    x, m, _ = agg(e1["full"], "frac_A", "delta")
    demo["window"] = {"delta": x.tolist(), "stc": m.round(3).tolist(),
                      "no_tags": agg(e1["no_tags"], "frac_A", "delta")[1].round(3).tolist(),
                      "per_synapse": agg(e1["prp_per_synapse"], "frac_A", "delta")[1].round(3).tolist(),
                      "acc": agg(e1["full"], "acc_A", "delta")[1].round(3).tolist(),
                      "acc_none": round(none_mean(e1["full"], "acc_A", "delta"), 3),
                      "acc_after": round(st.mean(r["acc_A_after_A"] for r in e1["full"]), 3)}

# ---------------------------------------------------------------- E2 cross
e2 = load("e2_cross.json")
if e2:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
    for name, col, ls, label in [("weight_tags", GRAY, "--", "Tags bound to the weight"),
                                 ("change_tags", TAG, "-", "Tags bound to the change")]:
        for ax, key in [(axes[0], "frac_A"), (axes[1], "acc_A")]:
            x, m, s = agg(e2[name], key, "gap")
            ax.plot(x, m, ls, color=col, lw=2, marker="o", ms=3.5, label=label)
            ax.fill_between(x, m - s, m + s, color=col, alpha=0.12, lw=0)
    for ax, key in [(axes[0], "frac_A"), (axes[1], "acc_A")]:
        v = st.mean(r[key] for r in e2["no_salience"])
        ax.axhline(v, color=INK, lw=1, ls=(0, (1, 3)))
        ax.text(900, v, "B not salient", ha="right", va="bottom", fontsize=9)
        ax.set_xlabel("Gap between task A and salient task B (steps)")
    axes[0].set_ylabel("Share of task A consolidated")
    axes[1].set_ylabel("Task A accuracy after delay")
    axes[0].legend(fontsize=9)
    fig.suptitle("E2  Cross-task rescue: the salient onset of task B rescues task A; selectivity needs change-bound tags",
                 x=0.01, ha="left", fontsize=11.5)
    fig.savefig(os.path.join(FIG, "fig3_cross_task.png")); plt.close(fig)
    summary += ["## E2: cross-task rescue (5 seeds)", "",
                "| gap | weight-bound tags: share A / acc A | change-bound tags: share A / acc A |", "|---|---|---|"]
    for g in sorted({r["gap"] for r in e2["weight_tags"]}):
        a = [r for r in e2["weight_tags"] if r["gap"] == g]; b = [r for r in e2["change_tags"] if r["gap"] == g]
        summary.append(f"| {g} | {st.mean(r['frac_A'] for r in a):.2f} / {st.mean(r['acc_A'] for r in a):.3f} | "
                       f"{st.mean(r['frac_A'] for r in b):.2f} / {st.mean(r['acc_A'] for r in b):.3f} |")
    ns = e2["no_salience"]
    summary += ["", f"B not salient: share A {st.mean(r['frac_A'] for r in ns):.2f}, acc A {st.mean(r['acc_A'] for r in ns):.3f}, "
                f"acc B {st.mean(r['acc_B'] for r in ns):.3f}. Task B accuracy with salient onset: "
                f"{st.mean(r['acc_B'] for r in e2['weight_tags']):.3f}.", ""]
    demo["cross"] = {"gap": agg(e2["change_tags"], "frac_A", "gap")[0].tolist(),
                     "change": agg(e2["change_tags"], "frac_A", "gap")[1].round(3).tolist(),
                     "weight": agg(e2["weight_tags"], "frac_A", "gap")[1].round(3).tolist()}

# ---------------------------------------------------------------- E3 explain
e3 = load("e3_explain.json")
if e3:
    rows = e3["deletion"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    styles = {"log_A": (TAG, "-", "Consolidation log: task A"), "log_B": (PRP, "-", "Consolidation log: task B"),
              "fisher_A": (TAG, "--", "Fisher importance: task A"), "fisher_B": (PRP, "--", "Fisher importance: task B"),
              "random": (GRAY, ":", "Random")}
    for ax, key, title in [(axes[0], "acc_A", "Accuracy on task A"), (axes[1], "acc_B", "Accuracy on task B")]:
        for rk, (c, ls, lab) in styles.items():
            sub = [r for r in rows if r["ranking"] == rk]
            x, m, s = agg(sub, key, "level")
            ax.plot(x * 100, m, ls, color=c, lw=2, marker="o", ms=3, label=lab)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("Parameters reset to their pre-learning value (%)")
    axes[0].set_ylabel("Test accuracy")
    axes[1].legend(fontsize=8.5, loc="lower left")
    fig.suptitle("E3  Faithfulness: resetting the weights the log names removes that memory, and only that memory",
                 x=0.01, ha="left", fontsize=11.5)
    fig.savefig(os.path.join(FIG, "fig4_faithfulness.png")); plt.close(fig)

    def at(rk, lv, key):
        return st.mean(r[key] for r in rows if r["ranking"] == rk and abs(r["level"] - lv) < 1e-9)
    summary += ["## E3: faithfulness (deletion test, 5 seeds)", "",
                "| ranking | reset 1%: acc A / acc B | reset 5%: acc A / acc B |", "|---|---|---|"]
    for rk in styles:
        summary.append(f"| {rk} | {at(rk,0.01,'acc_A'):.3f} / {at(rk,0.01,'acc_B'):.3f} | {at(rk,0.05,'acc_A'):.3f} / {at(rk,0.05,'acc_B'):.3f} |")
    summary.append(f"| none | {at('random',0.0,'acc_A'):.3f} / {at('random',0.0,'acc_B'):.3f} | |")
    att = e3["attribution"]
    summary += ["", "Memory-source attribution at the end of task B (before the delay):",
                f"- task A: full weights {st.mean(r['acc_A_end_B'] for r in att):.3f}, long-term part only {st.mean(r['late_only_A_end_B'] for r in att):.3f}",
                f"- task B: full weights {st.mean(r['acc_B_end_B'] for r in att):.3f}, long-term part only {st.mean(r['late_only_B_end_B'] for r in att):.3f}", ""]
    prov = [p for p in e3["provenance"] if p["seed"] == 0]
    summary += ["Provenance log, seed 0:", ""]
    for p in prov:
        summary.append(f"- event '{p['event']}' consolidated {p['fraction']:.0%} of source '{p['source']}' "
                       f"(total change {p['amount']:.2f}); strongest hidden units {p['top_hidden_units']}")
    summary.append("")
    demo["faith"] = {rk: agg([r for r in rows if r["ranking"] == rk], "acc_A", "level")[1].round(3).tolist() for rk in styles}
    demo["faithB"] = {rk: agg([r for r in rows if r["ranking"] == rk], "acc_B", "level")[1].round(3).tolist() for rk in styles}
    demo["levels"] = sorted({r["level"] for r in rows})
    demo["provenance"] = [{k: p[k] for k in ("event", "source", "fraction", "top_hidden_units")} for p in prov]

    feat = e3["features"]
    fig, axes = plt.subplots(2, 6, figsize=(10.5, 3.9))
    for row, key, lab in [(0, "kept", "Kept (consolidated)"), (1, "lost", "Faded (not consolidated)")]:
        for i in range(6):
            img = np.array(feat[key][i]); v = np.abs(img).max() or 1
            axes[row, i].imshow(img, cmap="PiYG", vmin=-v, vmax=v)
            axes[row, i].set_xticks([]); axes[row, i].set_yticks([])
            for sp in axes[row, i].spines.values():
                sp.set_visible(False)
            axes[row, i].set_title(f"unit {feat[key+'_units'][i]}", fontsize=8.5)
        axes[row, 0].set_ylabel(lab, fontsize=9.5)
    fig.suptitle("E3  What task A left in the first layer: changes that were kept vs changes that faded",
                 x=0.01, ha="left", fontsize=11.5)
    fig.savefig(os.path.join(FIG, "fig5_features.png")); plt.close(fig)
    demo["features"] = {k: [np.round(np.array(v), 3).tolist() for v in feat[k]] for k in ("kept", "lost")}
    tl = e3["timeline_seed0"]
    demo["timeline"] = {"t": [x["t"] for x in tl], "prp": [round(x["prp"], 4) for x in tl],
                        "tagged": [round(x["tagged_frac"], 4) for x in tl],
                        "capture": [round(x["capture"], 5) for x in tl]}

# ---------------------------------------------------------------- E4 split-MNIST
e4 = load("e4_splitmnist.json")
if e4:
    methods = [("plain", "Plain SGD", GRAY), ("ewc", "EWC", "#C9A227"), ("si", "SI", "#6B7FB8"),
               ("stc_oracle", "STC, event at task onset", "#6FC3A6"),
               ("stc_surprise", "STC, surprise-gated", TAG),
               ("stc_anchor", "STC log as importance", LATE)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    summary += ["## E4: Split-MNIST (5 tasks, 5 seeds, average final accuracy)", "",
                "| method | task-incremental | domain-incremental |", "|---|---|---|"]
    table = {}
    for ax, sc, title in [(axes[0], "task", "Task-incremental (one head per task)"),
                          (axes[1], "domain", "Domain-incremental (shared head)")]:
        ms, ss = [], []
        for m, _, _ in methods:
            v = [r["avg"] for r in e4["eval"] if r["method"] == m and r["scenario"] == sc]
            ms.append(st.mean(v) if v else np.nan); ss.append(st.pstdev(v) if v else 0)
            table.setdefault(m, {})[sc] = (ms[-1], ss[-1])
        ax.bar(range(len(methods)), ms, yerr=ss, color=[c for *_, c in methods], width=0.7, capsize=3)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([l for _, l, _ in methods], rotation=28, ha="right", fontsize=8.5)
        ax.set_title(title, fontsize=10, loc="left")
        lo = np.nanmin(ms)
        ax.set_ylim(max(0, lo - 0.1), 1.0)
    axes[0].set_ylabel("Average accuracy on all 5 tasks")
    fig.suptitle("E4  Split-MNIST: STC alone barely reduces forgetting; its log as importance matches EWC, not SI",
                 x=0.01, ha="left", fontsize=11.5)
    fig.savefig(os.path.join(FIG, "fig6_splitmnist.png")); plt.close(fig)
    for m, lab, _ in methods:
        t, d = table[m]["task"], table[m]["domain"]
        summary.append(f"| {lab} | {t[0]:.3f} ± {t[1]:.3f} | {d[0]:.3f} ± {d[1]:.3f} |")
    hp = {(r["method"], r["scenario"]): r["hp"] for r in e4["eval"]}
    summary += ["", "Selected hyperparameters (tuning seed 100): " +
                "; ".join(f"{m}/{s}: {h}" for (m, s), h in sorted(hp.items(), key=lambda x: (x[0][1], x[0][0]))), ""]
    demo["splitmnist"] = {m: {sc: round(table[m][sc][0], 3) for sc in ("task", "domain")} for m, _, _ in methods}

with open(os.path.join(RES, "summary.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary))
data_js = "window.STC_DATA = " + json.dumps(demo) + ";"
tpl = os.path.join(DEMO, "demo_template.html")
if os.path.exists(tpl):
    html = open(tpl, encoding="utf-8").read().replace("/*DATA*/", data_js)
    with open(os.path.join(DEMO, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
print("\n".join(summary))
