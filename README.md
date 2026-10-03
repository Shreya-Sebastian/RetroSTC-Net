# RetroSTC-Net

Retroactive consolidation for neural networks, based on synaptic tagging and capture.

Can an event that happens *after* learning decide whether a network keeps what it learned?

In the synaptic tagging and capture (STC) hypothesis, weak stimulation of a hippocampal input produces only transient early-phase LTP but sets a short-lived synaptic tag. Strong stimulation of a separate input to the same neurons supplies plasticity-related products (PRPs), which tagged synapses capture, turning the change into lasting late-phase LTP (Frey & Morris, 1997). This also works when the strong stimulation comes after the weak one (Frey & Morris, 1998). At the behavioural level the strong event can be a salient experience such as novelty (Redondo & Morris, 2011): in rats, exploring a novel open field 1 h before, or 15 to 60 min after, weak inhibitory-avoidance training turned a short-term memory into a 24-hour memory (behavioural tagging; Moncada & Viola, 2007).

This project implements a simplified version of that mechanism in small backprop-trained networks, tests whether a behavioural-tagging time window appears, and keeps a record of what was consolidated, which is then tested with deletion experiments.

![Behavioural-tagging window](figures/fig1_window.png)

## Contribution

Each part has precedent:

- Two-component weights with consolidation in a backprop MLP on MNIST, triggered by the size of the transient change, for energy efficiency (Li & van Rossum, 2020).
- A decaying fast and a stable slow weight in a single-layer, reward-trained network, where large fast changes are consolidated when postsynaptic activity is high, framed as tag and capture (Leimer, Herzog & Senn, 2019, preprint).
- Neuromodulator-dependent STC in recurrent spiking networks, where neuromodulation after learning retroactively selects what is consolidated (Luboeinski & Tetzlaff, 2021; Lehr, Luboeinski & Tetzlaff, 2022).
- Logging what gated consolidation keeps, proposed as a route to traceability (Takano, 2026).

To my knowledge, this project is the first to:

- implement, in a backprop-trained network, consolidation decided retroactively by a salience event, combining two-component weights with decaying per-weight tags
- test a behavioural-tagging time window, including rescue of one task by another, in the weights of a gradient-trained network
- implement a consolidation log in such a network and test it with deletion experiments (it separates the two tasks, but for the earlier task it does no better than a simple change-size ranking)

## Method

Each parameter (weights and biases) is split into a consolidated part `L` and a labile part `E`, with effective value `w = L + E`. The optimiser updates `w` as usual; after each step its change `Δw` is added to `E`, and then:

| Biology | Network | Default |
|---|---|---|
| Early-phase LTP decays | `E ← E · exp(−1/τE)` | τE = 1000 steps |
| Synaptic tag | `h ← h · exp(−1/τh) + \|Δw\|` | τh = 150 steps |
| PRPs, stored per unit | `p_unit ← p_unit · exp(−1/τp) + α·s(t)` | τp = 80 steps, α = 1 |
| Capture | `c = clamp(κ · p_unit · [h > θ], 0, 1)`; `L += c·E`, `E −= c·E` | κ = 0.05, θ = 10⁻³ |
| Salient event | `s(t)`: a pulse, or a sudden jump in training loss | |

In the default mode every unit receives the same PRP pulse, so PRPs act as one global signal; where capture happens is decided by the tags. Two variants are used as ablations: PRPs made only at synapses tagged at the time of the event, and a separate tag for each learning episode's change instead of one per weight ("tags per change").

Labile change is tracked per learning episode, using episode labels supplied by the experiment code, so every consolidated change can be traced to the episode that produced it and to the most recent salience event (exact when events do not overlap within the PRP lifetime). Implementation: [`stc/stc.py`](stc/stc.py); tests: [`tests/test_stc.py`](tests/test_stc.py).

## Results

Setup unless stated otherwise: MLP with two hidden layers of 256 units, SGD (learning rate 0.05, batch 64), two binary MNIST tasks from the harder digit pairs (A: 3 vs 5, B: 4 vs 9), 5 seeds. Experiment 1 uses one shared output; experiments 2 and 3 use a separate output per task, so interference between tasks runs through the shared hidden layers. In experiments 1 to 3 each run ends with 4000 steps without learning, after which the labile part has decayed to under 2% of its value (exp(−4) for τE = 1000), so what remains is almost entirely consolidated memory. The main measure is the share of task A's learning that became permanent: the consolidated change projected onto the total change task A made, Σ(consolidated · change) / Σ(change²). Accuracy is reported too, but saturates quickly on these tasks. Full tables: [results/summary.md](results/summary.md).

### 1. Event timing determines what is kept

60 SGD steps on task A, one salient pulse at a variable time (figure above). Because capture needs both a tag and PRPs, the shape of the window follows from the time constants built into the model. This experiment therefore checks that the mechanism works as designed inside gradient training and that the consolidated share translates into retained accuracy; it does not show that a window emerges on its own.

| Event time relative to end of A | none | −300 | −150 | 0 | +150 | +300 | +600 |
|---|---|---|---|---|---|---|---|
| Share of A kept | 0.00 | 0.13 | 0.58 | 0.91 | 0.71 | 0.42 | 0.03 |

- Without an event nothing is consolidated, and accuracy on A falls from 0.95 to 0.61 after the delay. With an event near the end of learning it stays at about 0.95.
- Without tags the window after learning stays open much longer, limited only by the decay of the labile weights: an event at +600 still keeps 51%.
- With PRPs made only at synapses tagged at event time, events before learning have no effect.
- Size-triggered capture (in the style of Li & van Rossum, 2020; capture wherever |E| > 0.002) keeps 62% without any event. It ignores salience, so later events cannot change what it keeps.

![Time constants](figures/fig2_time_constants.png)

The two sides of the window are controlled mainly by different time constants (3 seeds per setting). Tag lifetime τh mainly shifts the after-side (at +300: 0.01 / 0.42 / 0.63 for τh = 75 / 150 / 300; at −150 it changes little: 0.54 / 0.58 / 0.58). PRP lifetime τp mainly shifts the before-side (at −150: 0.11 / 0.58 / 0.91 for τp = 40 / 80 / 160), but also affects the share kept at and just after the end of learning (at 0: 0.81 / 0.91 / 0.93).

### 2. Cross-task rescue

60 steps on task A, an idle gap without training, then 400 steps on task B, with a salience pulse at B's onset.

![Cross-task rescue](figures/fig3_cross_task.png)

| Gap between A and B | 0 | 150 | 300 | 600 | 900 | B not salient |
|---|---|---|---|---|---|---|
| Tags per weight | 0.92 | 0.78 | 0.66 | 0.37 | 0.24 | 0.00 |
| Tags per change | 0.91 | 0.71 | 0.43 | 0.03 | 0.00 | 0.00 |

- B's onset rescues A retroactively. Without salience nothing is consolidated, and after the delay B is at chance (0.52 instead of 0.97).
- With one tag per weight, B's own learning re-tags weights shared with A, so B's event also captures A's residual change long after A's tag has faded (24% at a 900-step gap). Binding each episode's change to its own tag restores a clean window.

### 3. Consolidation log as explanation

Log for one run:

> The "B onset" event consolidated 85% of task A (strongest hidden units 131, 52, 200, 214, 233) and 63% of task B (units 135, 132, 200, 211, 181).

Units are in the first hidden layer, ranked by the consolidated change of their incoming weights.

The log is tested by deletion (gap of 75 steps between A and B): parameters are ranked by the log's attribution to A or B (size of the consolidated change from that task), by empirical Fisher importance for A or B (computed on the final model from 1,000 training images), by the total change each task made to each parameter, consolidated or not (change size), or at random. The top-ranked parameters are reset to their values before task A was trained.

![Deletion test](figures/fig4_faithfulness.png)

| Reset 5% of parameters, ranked by | Accuracy A | Accuracy B |
|---|---|---|
| nothing reset | 0.872 | 0.971 |
| log, task A | 0.797 | 0.963 |
| log, task B | 0.951 | 0.691 |
| Fisher, task A | 0.902 | 0.950 |
| Fisher, task B | 0.946 | 0.929 |
| change size, task A | 0.796 | 0.964 |
| change size, task B | 0.948 | 0.764 |
| random | 0.876 | 0.971 |

- The log dissociates the two memories: resetting A's attributed weights lowers A and spares B, and vice versa. Fisher-ranked resets do not do this: resetting the top Fisher weights for A even raises accuracy on A (0.87 to 0.90).
- Ranking weights simply by how much each task changed them is a stronger baseline than Fisher. For task A it gives the same result as the log (0.796 vs 0.797). For task B the log does better than this baseline (accuracy B 0.691 vs 0.764). So the log's attribution to A mostly reflects which task changed which weights; what consolidation adds is visible only for B.
- Resetting B's weights raises accuracy on A (0.87 to 0.95), which indicates that B interferes with A.
- Evaluating only the consolidated weights at the end of task B gives higher accuracy on A (0.873) than the full weights (0.846), consistent with A being held in consolidated memory and B's labile changes interfering with it.

![Features](figures/fig5_features.png)

First-layer weight changes from task A: on the left, the consolidated change of the units with the most consolidation; on the right, the total change of the units whose change was mostly not consolidated.

### 4. Forgetting on Split-MNIST

Five binary digit tasks in sequence (0/1, 2/3, 4/5, 6/7, 8/9), 500 steps each, no delay at the end. "Event at each task onset" places a pulse at the start of tasks 2 to 5. In the surprise-gated variants, events come from a detector that fires when the training loss exceeds its running mean by 3 running standard deviations, with a 150-step refractory period. Its settings were fixed in advance. The trigger itself uses no task labels (the task-incremental setting still uses a separate output head per task, and the log is kept per task for bookkeeping). "STC log as importance" adds an EWC-style penalty that pulls each parameter towards its consolidated value `L`, weighted by its consolidated change (normalised to a maximum of 1). EWC uses the empirical Fisher at the end of each task; SI uses the path integral with ξ = 10⁻³. Tuned per method: the penalty strength for EWC, SI and STC log as importance, and τE for the other STC variants; all other STC settings are the defaults above. In practice the detector fired 5 to 9 times per run, mostly within tasks, and caught a task change within 20 steps in 0 to 3 of the 4 changes. Hyperparameters were chosen on a separate seed (100) but scored on the same MNIST test set. Values are means over 5 seeds; standard deviations are in [results/summary.md](results/summary.md).

![Split-MNIST](figures/fig6_splitmnist.png)

| Method | Task-incremental | Domain-incremental |
|---|---|---|
| Plain SGD | 0.980 | 0.616 |
| EWC | 0.987 | 0.637 |
| Synaptic Intelligence | 0.987 | 0.657 |
| STC, event at each task onset | 0.980 | 0.619 |
| STC, surprise-gated | 0.980 | 0.621 |
| STC log as importance, surprise-gated | 0.980 | 0.636 |

- STC alone does not measurably reduce forgetting (domain-incremental: 0.619 and 0.621 vs 0.616, within seed variation): it decides what becomes permanent, but consolidated weights are not protected from later learning.
- Using the consolidation log as importance in an EWC-style penalty reaches EWC level (0.636 vs 0.637). Its importance comes from capture events after learning rather than being fixed at the end of each task, but because the detector does not reliably mark task changes, this result does not show that surprise picks the right moments. It does not reach Synaptic Intelligence (0.657).
- With one output head per task, all methods are near ceiling.

## Relevance

- In many applications importance becomes clear only later: fraud labels, clinical outcomes, user corrections. Retroactive consolidation is designed for that setting; whether it beats importance fixed at training time there is not tested here.
- For models that keep learning after deployment, a consolidation log records what was made permanent and what triggered it, so that record can be inspected rather than assumed.

## Limitations

- Small MLPs on MNIST only; results demonstrate the mechanism, not performance at scale.
- Time is measured in training steps; time constants were set by hand.
- The deletion test resets weights to pre-learning values without retraining (unlike ROAR, Hooker et al., 2019); other interventions may rank methods differently. Against a change-magnitude baseline the log adds nothing for task A (see experiment 3).
- The surprise detector reacts to loss noise as well as to task changes.
- Hyperparameters for experiment 4 were selected on test accuracy of a separate seed, not on a validation split.
- Time-constant sweeps use 3 seeds; other results use 5.
- The biological mapping is simplified: backprop is not a biological learning rule, PRPs are one number that every unit receives equally rather than proteins made by active neurons, and a hidden unit is not a neuron.

Next steps: a benchmark where importance is revealed only later (delayed rewards or labels), protection of consolidated weights, and extension to CNNs and LoRA fine-tuning of language models.

## Usage

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # macOS/Linux: .venv/bin/python
.venv/Scripts/python tests/test_stc.py
.venv/Scripts/python experiments/e1_window.py      # ~15 min on a laptop CPU
.venv/Scripts/python experiments/e2_cross.py       # ~6 min
.venv/Scripts/python experiments/e3_explain.py     # ~1 min
.venv/Scripts/python experiments/e4_splitmnist.py  # ~16 min
.venv/Scripts/python make_figures.py               # figures, results/summary.md, demo/index.html
```

MNIST is downloaded to `data/` on first run. `demo/index.html` contains an interactive version of the results.

## References

- Frey, U. & Morris, R. G. M. (1997). Synaptic tagging and long-term potentiation. *Nature* 385:533–536.
- Frey, U. & Morris, R. G. M. (1998). Weak before strong: dissociating synaptic tagging and plasticity-factor accounts of late-LTP. *Neuropharmacology* 37:545–552.
- Redondo, R. L. & Morris, R. G. M. (2011). Making memories last: the synaptic tagging and capture hypothesis. *Nat. Rev. Neurosci.* 12:17–30.
- Moncada, D. & Viola, H. (2007). Induction of long-term memory by exposure to novelty requires protein synthesis: evidence for a behavioral tagging. *J. Neurosci.* 27:7476–7481.
- Leimer, P., Herzog, M. & Senn, W. (2019). Synaptic weight decay with selective consolidation enables fast learning without catastrophic forgetting. *bioRxiv* 613265 (preprint).
- Li, H. L. & van Rossum, M. C. W. (2020). Energy efficient synaptic plasticity. *eLife* 9:e50804.
- Lehr, A. B., Luboeinski, J. & Tetzlaff, C. (2022). Neuromodulator-dependent synaptic tagging and capture retroactively controls neural coding in spiking neural networks. *Sci. Rep.* 12:17772.
- Luboeinski, J. & Tetzlaff, C. (2021). Memory consolidation and improvement by synaptic tagging and capture in recurrent neural networks. *Commun. Biol.* 4:275.
- Takano, R. (2026). From retrospective decoding to prospective logging: traceability as a byproduct of gated consolidation. *Zenodo* preprint, doi:10.5281/zenodo.22031993.
- Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS* 114:3521–3526.
- Zenke, F., Poole, B. & Ganguli, S. (2017). Continual learning through synaptic intelligence. *ICML*, PMLR 70:3987–3995.
- van de Ven, G. M., Tuytelaars, T. & Tolias, A. S. (2022). Three types of incremental learning. *Nat. Mach. Intell.* 4:1185–1197.
- Hooker, S., Erhan, D., Kindermans, P.-J. & Kim, B. (2019). A benchmark for interpretability methods in deep neural networks. *NeurIPS.*

## Licence

MIT, see [LICENSE](LICENSE).
