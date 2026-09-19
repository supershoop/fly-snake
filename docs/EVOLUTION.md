# Reward-driven Snake evolution

This project can learn through thousands of generations of games. The new
trainer evolves the **linear descending-neuron readout**, starting from random
parameters. The connectome's synaptic weights remain fixed. The result is a
standard `Policy` file that receives simulated neuron spike counts at inference.

The completed regularized run used 2,000 generations, 24 candidates per
generation, and 227,784 actual training games. It learned strongly relative to
its random starting point, but **did not beat the existing model on the fresh
response-bank test**. The existing demo model has therefore been preserved.

## Research and method choice

[Code Bullet's SnakeFusion](https://github.com/Code-Bullet/SnakeFusion) uses
genetic algorithms and neural networks. This is neuroevolution: score policies
by playing games, select stronger policies, mutate them, and repeat. It differs
from gradient-based reinforcement learning such as
[PPO](https://arxiv.org/abs/1707.06347), which optimizes a policy objective from
sampled experience. Both use rewards from interaction, rather than requiring a
label telling the agent the correct action at every step.

[Deep Neuroevolution](https://arxiv.org/abs/1712.06567) demonstrates that a
mutation-and-selection genetic algorithm can train neural policies on RL tasks.
[Evolution Strategies](https://arxiv.org/abs/1703.03864) is related work on
scalable black-box policy optimization. Our trainer is a small elitist genetic
algorithm, not an implementation or reproduction of either paper's benchmark.
It uses no crossover, no teacher labels, and no action-safety filter.

The practical choice here is based on this machine and this project: the
machine has CPU-only PyTorch, and full connectome simulation is expensive.
Existing simulated response banks and a small linear readout make fast offline
evolution feasible. This is not evidence that evolution universally beats PPO.

## What one generation does

1. Construct 24 candidate readouts. The first generation is random; later
   generations retain four elites, mutate their descendants, and occasionally
   introduce a new random candidate.
2. Let every candidate play eight games, up to 600 moves each. Candidates in a
   generation use identical game seeds and response-sampling RNG streams.
3. Rank by mean food collected. Ties prefer fewer collisions, fewer starvation
   deaths, then more moves. There is no reward for pretending a capped game was
   won and no heuristic action mask. The built-in 150-move starvation limit stays.
4. Keep four distinct behaviors when possible. Mutation changes a mixture of
   small and large subsets of coefficients at several scales. All candidates
   remain ordinary linear readouts.
5. Rotate the eight training game seeds every 50 generations. Every 50
   generations, evaluate the training elites on separate validation games and
   response samples; save a checkpoint whenever validation improves.

The optimizer searches 72 coefficients: three actions times 24 fixed response
directions. Those directions are derived from mean `log1p` descending-neuron
responses in **training samples only**, without action labels. A ridge-regularized
inverse suppresses directions that amplify noise. The directions are multiplied
into the final weights, giving the usual 3 × 1,314 weights plus three biases.
No lookup table or extra board information is used by the saved live policy.

The 24-state encoder is still engineered: it supplies food direction and threat
signals including tail connectivity. This experiment does not establish that a
biological fly can learn Snake or plan routes.

## Efficiency without changing the game

`flybrain/evolution_game.py` compiles solo Snake and tail-connectivity sensing
with Numba. Differential tests compare its observations, random food sequences,
scores, collision/starvation endings, and move counts against the canonical
Python game. A timing check after compilation ran 64 games with a 600-move cap
in approximately 0.054 seconds on this machine; timing varies with policy and
other workloads. This timing covers game rollouts, not brain simulation.

Candidates that choose identical actions for every training-bank sample reuse
the same deterministic rollout result. The cache is cleared when game seeds or
evaluation settings change. The regularized run requested 48,000 candidate
evaluations, actually evaluated 28,473 distinct behaviors within their seed
batches, and reused 19,527 results. Its 227,784-game count excludes validation
games and cached duplicate evaluations. Training and validation took 126 seconds,
excluding development, dependency installation, fresh-bank collection, and the
continuous-brain check.

The run used Python 3.12, NumPy 2.5.3, CPU PyTorch 2.14.0 and Numba 0.67.0.

Each rollout samples actual previously simulated DN responses. That saves
rerunning 165,122 neurons for every training move, but it is an **approximation**:
sampled responses do not reproduce action-dependent neural history in a
continuously running brain. Full-game scores from this trainer must be labeled
response-bank estimates.

## Data separation and results

Training used `bank-real-transition.npz`, with 32 responses per situation:
trials 0–15 trained the population and basis; trials 16–23 selected the winner.
These boundaries align with the bank's reset-every-four-trials blocks. Training
game seeds were 40000–40319; validation seeds were 50000–50031. The regularized
run did not use trials 24–31 for optimization or scoring.

| Generation | Best validation mean food (1,600-move cap) |
| --- | ---: |
| 1 | 3.72 |
| 50 | 41.91 |
| 350 | 59.78 |
| 650 | 62.47 |
| 900 | 64.88 |
| 1,500 | 65.63 |
| 2,000 | 65.63 |

The selected winner came from generation 1,500. The last 500 generations did
not improve validation. These selection scores are not independent test scores.

An initial 2,000-generation pilot used weak regularization (ridge 0.001). Its
training scores improved but validation plateaued at 5.19 food. Its old-bank
test mean was 2.43. This failed pilot is retained in
`models/readout-evolved-2000.json`; it is not the selected model. Regularization
was increased to 100 based on the validation failure. The revised model's best
validation mean was 65.625, selected during the 2,000-generation run.

Because the pilot had already consumed the old held-out responses, final testing
used a **new response bank**, generated with neural seed 170071, 16 responses per
situation, and game seeds 81000–81127. The selected model was frozen before those
test results were viewed. The same tail-escape encoder, move cap, game seeds, and
response draws were used for all three policies.

**Fresh response-bank estimates: 128 games, 1,600-move cap.**

| Policy | Mean food | Collision deaths | Starvation deaths | Still alive at cap |
| --- | ---: | ---: | ---: | ---: |
| Random initial readout | 0.0078 | 0 | 128 | 0 |
| Existing `readout-real` | 36.9375 | 97 | 31 | 0 |
| Evolved, ridge 100 | 32.8672 | 125 | 3 | 0 |

The evolved-minus-existing paired difference is **−4.0703 food**. A paired
bootstrap over these game seeds gives a 95% interval of **[−7.3828, −0.7813]**.
This interval does not cover variation from different training runs or fresh
neural response banks. There was one regularized training run, not a multi-seed
algorithm benchmark. The gap between validation and fresh-bank scores is further
evidence that reusing a small neural-response bank can overstate performance.
The existing model was trained with heuristic-derived targets; the evolved
model started random without teacher actions. This comparison checks deployment
quality and does not isolate which learning algorithm is intrinsically better.

**Continuous-brain check: four paired game seeds, 250-move cap.**

| Policy | Mean food | Per-game food | Collisions | Starved | Still alive at cap |
| --- | ---: | --- | ---: | ---: | ---: |
| Existing `readout-real` | 21.5 | 26, 24, 26, 10 | 0 | 1 | 3 |
| Evolved, ridge 100 | 18.0 | 25, 9, 23, 15 | 2 | 0 | 2 |

These used game seeds 90000–90003 and matched neural-noise seeds 1729–1732.
The brain ran continuously within each game. The candidate was better on one
of the four seeds, but the existing model had the higher mean and fewer
collisions. This is a small capped comparison, not a statistical guarantee or
an estimate of completed-game performance. Evaluation took about 24 minutes
on the CPU while other local workloads were running. Both the independent bank
test and this check support preserving the existing demo model.

Machine-readable evidence, configuration, seeds, model hashes, and per-game
scores are in:

- `models/readout-evolved-ridge100.json`: training and validation.
- `models/readout-evolved-test.json`: fresh-bank test.
- `models/readout-evolved-history.json`: the validation-checkpoint learning curve.
- `models/readout-evolved-initial.npz`: the random starting readout used in the comparison.
- `models/readout-evolved-live.json`: continuous-brain check; only a report with
  `complete: true` represents the completed paired comparison.
- `outputs/evolution-ridge100/history.json`: all 2,000 generation summaries.
- `outputs/evolution-ridge100/winner-*.npz`: successive validation winners.

## Reproduce or continue

Install the usual project dependencies and the optional training compiler:

```sh
.venv/Scripts/python -m pip install -r requirements-training.txt

# New experiment, from random initialization; leaves the demo model untouched.
.venv/Scripts/python scripts/train_evolution.py --generations 2000 --ridge 100 --validation-only --run outputs/evolution-new --output models/readout-evolved-new.npz

# Resume an interrupted run with the same options, adding --resume.
.venv/Scripts/python scripts/train_evolution.py --generations 2000 --ridge 100 --validation-only --run outputs/evolution-new --output models/readout-evolved-new.npz --resume

# Differential and existing regression tests.
.venv/Scripts/python -m unittest discover -s tests -p "test_*.py"
```

Run directories cannot be silently overwritten. Checkpoints save the population,
mutation RNG, selected winner, configuration and input hashes. A split/resumed
run was verified to produce exactly the same population and winner as an
uninterrupted run. `--validation-only` allows model development without scoring
the old held-out partition. Once a normal run writes `report.json`, or a separate
fresh-bank evaluation writes `final-test.json` in the run directory, it refuses
to resume that finalized run.

To recreate the separate fresh test bank (the file is intentionally gitignored):

```sh
.venv/Scripts/python -c "from pathlib import Path; import torch; from flybrain.connectome import load_connectome; from flybrain.channels import build_channels; from flybrain.response_bank import response_bank; torch.set_num_threads(4); c=load_connectome(); response_bank(c,build_channels(c),Path('data/bank-real-evolution-test.npz'),trials=16,seed=170071,device='cpu')"
.venv/Scripts/python scripts/evaluate_evolution_bank.py --model models/readout-evolved-ridge100.npz --training-report models/readout-evolved-ridge100.json --initial models/readout-evolved-initial.npz
.venv/Scripts/python scripts/evaluate_evolution.py --model models/readout-evolved-ridge100.npz --games 4 --max-moves 250 --game-seed 90000
```

The continuous check compares both policies using the same sensing, new game
seeds and matched Poisson-noise seeds. It resets the brain only between games,
not between moves. A snake alive after 250 moves is an unfinished game, not a
win. Use more seeds and longer games before making a strong live-performance
claim. CPU evaluation is much slower than the compiled bank rollouts.

On macOS/Linux use `.venv/bin/python`. The candidate can be loaded with
`Policy.load('readout-evolved-ridge100')`; it has the same format as the existing
readout. No saved connectome weights or default demo model were changed.

## What would improve the next experiment

The evidence points to response diversity as the next issue, not simply more
generations. A stronger follow-up would collect training responses across more
neural seeds and actual game trajectories, reserve whole independent neural
seeds for validation, and compare several training seeds. A GPU would make that
and continuous-brain fine-tuning more practical. Keep a new final test set
untouched while making those choices. The coarse 24-state observation also
limits what any memoryless readout can learn, regardless of generation count.
