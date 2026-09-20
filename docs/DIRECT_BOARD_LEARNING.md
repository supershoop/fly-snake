# Direct-board input and internal synapse training

This is a separate, user-authorized experiment. It starts with the original
MaleCNS connectome, feeds the complete solo Snake board into existing visual
neurons, and searches for useful changes to existing internal synapses. The
web demo's default encoder, saved models, and decision policies are unchanged.

## What the input contains

The 12 by 12 board is placed in a 23 by 23 field centered on the head and
rotated so the head faces up. That field is large enough to contain every
board square at every head position. This symmetry transform retains the
whole relative layout; the snake's movement commands are also relative.

Three planes represent food, occupied/outside-board space, and ordered body
segments. The raw body plane gives segment index divided by 144, so it
preserves the head, tail, length and body order. Head position is always the
center. At stimulation time its otherwise constant body-order input carries
the normalized starvation timer. Walls are distinguished from the body by
having occupancy but no body-order value. Empty cells are implicit zeros.

Each plane has 529 dedicated neurons, for **1,587 unique input neurons**:

- Food: LC10 neurons.
- Occupancy/boundary: LC4, LPLC2 and LC12 neurons. Nearby field positions use
  LC4 first, then LPLC2, then LC12.
- Body order and starvation timer: Tm3 neurons.

Positions are assigned deterministically within the matching left/right
hemisphere. These are **engineered receptive fields, not measured retinotopy
or a reconstruction of natural fly vision**. Fixed peak-normalized Gaussian
fields (sigma 3 cells for food, 0.8 for occupancy) spread stimuli over local
populations, with raw cell signals retained via a maximum and drives bounded
to 0..1. Finite stochastic spike trains do not transmit these arrays losslessly.

No next-move collision check, tail-path search, nearest-food direction, expert
action, or action mask is used by this encoder. The original 24-situation
encoder is used only as a comparator in the input probe, never in these
training decisions. All activity is simulated.

## What learns

The trainer starts all log gains at zero, corresponding to the original
connectome; it does not load the previous trained brain or external readout.
Diagonal CMA-ES changes **60 shared gains on 8,082 existing connections**:
sensory input neurons to direct steering relays, and incoming connections to
DNa02/DNa01. Every gain stays between 0.25 and 4 times its original strength.
No connections are added/deleted, and no transmitter signs change.
The eligible set comprises 5,505 food-input connections, 646 occupancy-input
connections, 194 body-order-input connections, and 1,737 connections entering
the steering neurons. The remaining connectome strengths stay fixed.

Every move runs the full 165,122-neuron, 6,235,682-edge LIF simulation for
100 ms at 0.5 ms time steps. State carries over between moves and resets only
between games. No response bank, reduced brain, or teacher controls actions.

The fixed decoder sums left DNa02/DNa01 spikes minus right spikes: above +2
means left, below -2 means right, otherwise straight. The encoder and decoder
are not trained. Only internal connection gains are optimized.

Fitness is mean food + 0.05 times the game's cumulative reward + 0.001 times
survived moves. Game rewards are food +1, death -1, closer/farther +/-0.1.
The small survival term helps break zero-food ties but can favor circling;
food, collisions and starvation must therefore be reported separately.

CMA-ES is an evolutionary parameter optimizer, not a claimed biological fly
learning rule. Reference: [CMA-ES author documentation](https://cma-es.github.io/).

## Reproduce

```powershell
.venv/Scripts/python -m pip install -r requirements-training.txt
.venv/Scripts/python scripts/probe_board_encoder.py --repeats 16 --output outputs/board-probe.json
.venv/Scripts/python scripts/verify_brain_kernel.py --direct-board --report outputs/board-kernel-verification.json
.venv/Scripts/python scripts/train_board_brain.py --run outputs/board-brain-new --generations 24 --workers 2
.venv/Scripts/python scripts/train_board_brain.py --run outputs/board-brain-new --test --test-games 32 --test-moves 250 --test-seed-index 100 --workers 2
```

The default pilot uses population 8, two matched training games per candidate,
24 generations, 100-move training caps, six validation games capped at 120
moves, and validation at generations 1/4/8/12/16/20/24. Training seeds rotate
each generation. All candidates in a generation share game and neural-noise
seeds. Validation selects the best **trained** candidate; the original brain
is an independent comparator and remains available even if training fails.

The final test uses 32 untouched paired game/noise seeds and a 250-move cap,
plus food-only and no-sensory-input controls with the trained synapses. The
food-only control checks whether additional board information helps the saved
model; encoding that information does not establish that the brain uses it.
The original brain with the old 24-pattern input is also evaluated on the
same game seeds as a reference. Its input-neuron count differs, so identical
noise seeds do not imply identical sensory noise between the two encoders.
Testing locks the run
before observing outcomes, preventing further training on that run. Short
games and small seed sets make this a feasibility pilot, not a definitive
comparison of learning algorithms.

Use `--workers 2` to evaluate independent CPU candidate brains concurrently;
each has separate neural state, synaptic weights, and input-noise RNG. Resume
unfinished training with the identical scientific flags plus `--resume`, optionally
increasing `--generations`. Ask/tell history is replayed and checked to restore
the CMA optimizer; resume requires the same CMA version and compute device.
Use a fresh `--run` and `--seed-offset` (a multiple of one million) for another
research run. `--device cuda` supports the existing GPU simulator, but has not
been benchmarked or validated for this experiment on this CPU laptop.

Saved synapse checkpoints validate anatomy, site identity, decoder and encoder
mapping before use. They are incompatible with the older five-channel synaptic
model and are not automatically loaded into the web app.

## Input and implementation checks

Six constructed layouts were each tested on the original brain with 16 paired
neural-noise seeds. Three changes preserve the old four-value/24-pattern input:

| Change from near-left food board | Changed input neurons | Mean changed descending neurons | Mean change in left-minus-right steering spikes |
|---|---:|---:|---:|
| Food farther left | 506 | 33.25 | +2.1875 |
| Different body shape | 49 | 35.31 | 0.0 |
| Closer left wall, same relative food | 117 | 90.06 | -0.1875 |

This establishes distinct simulated responses, not an ability to plan routes
or reliably decode every board. In particular, body shape changed descending
activity but barely affected the four fixed steering neurons before training.
See `models/brain-board-probe.json` and the accompanying raw response arrays.

The compiled CPU kernel matched all 165,122 neurons' spike counts exactly on
24 continuous moves with new inputs, including altered and restored synapses.
Full neural state agreed within 1e-5. The measured kernel time was 13.94 seconds
for the original versus 2.90 seconds compiled (about 4.81x in that check, not
a guaranteed end-to-end training speed). See `models/brain-board-kernel-verification.json`.

A two-generation implementation check produced identical candidate proposals,
game records, selected winner, and move counts for serial split/resume and
uninterrupted two-worker execution. Short smoke checks use the first test
seeds; the scientific pilot therefore reserves test indices 100..131
(8,300,100..8,300,131), selected before viewing their outcomes.

## Training results

The completed CPU pilot ran **24 generations, 384 scored training games and
48 validation games**, recording **16,260 whole-brain moves**. Generation 20
was selected on validation (original 0.3333 versus trained 2.6667 mean food).
All **8,082 eligible connections** changed; the input encoder and movement
decoder stayed fixed. Recorded training segments took 1,446.7 seconds (24.1
minutes), excluding initialization, probes, checks and discarded partial
generations when switching worker counts. Completed-generation history and
software versions are in `models/brain-board-training.json`.

The frozen checkpoint was then tested on **32 new paired game seeds**,
8,300,100..8,300,131, with a 250-move cap and the same fixed motor decoder:

| Condition | Mean food | Mean moves | Collisions | Starvation | Alive at cap |
|---|---:|---:|---:|---:|---:|
| Original brain, full-board input | 0.21875 | 7.8125 | 32/32 | 0 | 0 |
| Trained brain, full-board input | 2.0625 | 29.0625 | 32/32 | 0 | 0 |
| Trained brain, food-only input | 2.5625 | 30.875 | 32/32 | 0 | 0 |
| Trained brain, no sensory input | 0.09375 | 6.6875 | 32/32 | 0 | 0 |
| Original brain, old 24-pattern input | 2.59375 | 25.15625 | 32/32 | 0 | 0 |

The complete five-condition test simulated 3,187 moves in 291.7 seconds.
All rows are actual continuous whole-brain rollouts, not response-bank estimates.

Paired food-score comparisons (20,000 bootstrap samples, seed 91741):

- Training gain with full-board input: **+1.84375 food**, 95% interval
  **[+1.28125, +2.40625]**; 25 wins, 5 ties, 2 losses.
- Trained full board versus original 24-pattern input: **-0.53125**, interval
  **[-1.09375, +0.0625]**; 7 wins, 8 ties, 17 losses.
- Trained full board versus the same trained model with food-only input:
  **-0.5**, interval **[-1.125, +0.09375]**; 5 wins, 15 ties, 12 losses.

**Interpretation:** internal synapse training improved performance with the new
encoder on these unseen games. It did not establish that full-board input is
better than the old engineered observations or food-only stimulation. Those
two comparisons favor the simpler inputs numerically, but their intervals
include zero. Every evaluated game still ended in a collision. The additional
board detail reaches the network; this pilot does not demonstrate useful body
planning, optimal moves, or reliable Snake play. The old encoder already
computes collision and tail-route warnings, whereas this new encoder does not.

These intervals describe variation across game seeds for one trained checkpoint,
not variation across independent training runs. This is a small, restricted
60-parameter pilot, not a test of the full learning capacity of the connectome.
The next experiment should examine whether body/boundary signals can reliably
change appropriate steering responses, and compare improved mappings or wider
plastic pathways before committing to thousands of generations.

## Saved artifacts and using the model

- `models/brain-board-direct.npz`: selected generation-20 internal synapses.
- `models/brain-board-training.json`: configuration, original start, training
  history, validation and software versions.
- `models/brain-board-test.json`: all 160 final game records and paired analyses.
- `models/brain-board-probe.json` and `.npz`: input probe and raw DN responses.
- `models/brain-board-kernel-verification.json`: full-simulator equivalence check.

Checkpoint SHA-256:
`7a9cd82db482e8a570a420733cdec26fcdf262e6be986755ccf12b7f763b2c74`.
The previous readout and five-channel synaptic model files are unchanged.
The current web app still uses its existing input modes; this model is separate.

To evaluate the saved model on an additional seed from Python, at the repository
root (this does not train or modify its saved weights):

```python
from flybrain.board_experiment import BoardRunner

runner = BoardRunner()
parameters, metadata = runner.load("models/brain-board-direct.npz")
result = runner.evaluate(parameters, seeds=[9_400_000], limit=250)
print(result)
```

Validation also passed all 51 Python unit tests, strict model/encoder identity
checks, sign and gain-bound checks, final-test seed separation, and the serial
versus four-worker evaluation reproducibility check. A tested training run is
locked; further research should use a fresh run and fresh test seeds.
