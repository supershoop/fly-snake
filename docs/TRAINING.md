# Snake survival training

The previous readout learned to move toward food unless the next cell was
occupied. An open cell leading into a closed loop looked exactly like an open
cell with an escape route. Repeating that training could not teach the readout
to distinguish the two.

The five sensory inputs retain their names and binary values. Danger now also
includes a move that disconnects the head from the moving tail. The encoder
checks the hypothetical body after the move, including growth when eating,
and treats other living snakes as obstacles. It does not change the game or
override actions. This is engineered board analysis, not a claim about fly
perception or planning.

Training uses whole-game food scores to optimize the 24 action targets, starting
from the original teacher. Candidate targets explore safe alternatives on
training seeds only. The linear readout then learns those targets from actual
simulated descending-neuron spike counts. At inference it receives only those
counts. No lookup table, board features, action mask, or fallback controller is
used to choose a move. Synaptic weights remain fixed.

The response bank contains both reset and carried-over brain activity: every
four trials the brain resets, and stimuli are shuffled between each 100 ms
window. The old bank used only reset responses, unlike live play. Cache checks
include neuron IDs/order, window, dt, wiring, seed, trial count, and weight scale.

## Results

The shipped training run uses 32 trials per sensory pattern (24 train / 8
held out), training game seeds 100–111, evaluation seeds 20000–20063, and a
1,600-move limit. The readout matches the learned targets on all 192 held-out
brain responses. Complete configuration and per-game scores are in
[`models/readout-real-training.json`](../models/readout-real-training.json).

These are **sampled response-bank estimates**, not continuously simulated games:

| Configuration | Mean food score | Collision deaths / 64 | Idle timeouts / 64 |
| --- | ---: | ---: | ---: |
| Original sensing and readout | 11.08 | 64 | 0 |
| Escape sensing, original readout | 14.30 | 64 | 0 |
| Escape sensing, retrained readout | 54.34 | 0 | 45 |

The remaining 19 retrained games were alive at the move limit; they are not
counted as wins. The original model is preserved as `readout-real-legacy.npz`.
The shuffled-wiring model has not been retrained as part of this change.

A separate **continuous-brain smoke comparison** ran four new seeds
(30000–30003), with one uninterrupted brain per game, through 275 recorded
moves. The original averaged **12.0** food, with **4/4 collisions**. The retrained
model averaged **24.5**, with **1/4 collisions** and **3/4 still alive** at the cap.
Those unfinished games are censored; this is a small bounded check, not a full
game benchmark or a zero-collision claim. See
[`models/readout-real-live-evaluation.json`](../models/readout-real-live-evaluation.json).
The actual server also chose the escape route in the enclosed-food regression
scenario using simulated spikes and the saved readout.

Long snakes can still circle safely until the 150-move idle timeout. The
observations remain coarse, and stochastic brain responses can still cause
mistakes. Zero collisions in this estimate is not a guarantee of perfect play.
Shared-arena opponents can also move into a route after it was observed.

## Run and reproduce

Select **Trained readout** to use `models/readout-real.npz`. Restart an already
running brain server after replacing that file, since it caches loaded policies.
**Learn live** still starts blank; it benefits from the new sensing but does not
load the pretrained weights or the offline target table.

With the Python environment and MaleCNS data described in `AGENTS.md`:

```sh
# Reproduce the shipped training and bank estimate (CPU or GPU).
.venv/Scripts/python scripts/train_readout.py --trials 32 --evaluation bank --threads 4 --report models/readout-real-training.json

# Fresh continuous-brain comparison on separate game seeds; slow on CPU.
.venv/Scripts/python scripts/evaluate_survival.py --games 4 --max-moves 275

# Regression tests; these require Python dependencies, but no connectome data.
.venv/Scripts/python -m unittest discover -s tests -p "test_*.py" -v
```

On macOS/Linux use `.venv/bin/python`. Training writes a model only when its
evaluated score exceeds the named baseline without increasing collisions. Use
`--output readout-candidate` to keep a separate candidate. `--evaluation live`
runs the brain throughout the evaluation; `--evaluation bank` prints and saves
an explicit estimate label. Training and evaluation game seeds must be disjoint.
The server uses 100 ms windows, so incompatible window settings are rejected.

## CPU playback speed

Single-brain CPU playback now propagates spikes through the outgoing columns
of firing neurons instead of multiplying the full connectome by mostly zeros.
All nonzero contributions remain; the 0.5 ms integration step, 100 ms decision
window, synaptic weights, and trained readout are unchanged. Swarm/arena batches
and GPU runs retain the original sparse multiplication. `Brain(cpu_sparse=False)`
selects the reference CPU path for comparisons.

On this machine, four consecutive 100 ms inputs took 9.49 s total with the
original CPU kernel versus 3.70 s with the optimized path (about 2.6x faster).
Every neuron's spike count and every chosen action matched exactly in that
check. Timings vary with activity and hardware; floating-point accumulation
can differ between numerical backends. Tests cover propagation, inhibitory
weights, continuous state, lesions, and resizing. The server selects four CPU
threads for one fly and up to eight for multiple flies.
