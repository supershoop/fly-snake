# Learning inside the simulated fly brain

This experiment keeps the requested signal loop:

**Snake observation → sensory-neuron stimulation → whole fly connectome →
descending-neuron spikes → fixed game controls.**

The learning algorithm changes connection strengths inside the simulated brain.
It does not choose moves, train a replacement neural network, or change the
movement decoder. This differs from the earlier readout-evolution experiment.

## What changes, and what remains anatomical

The simulator contains 165,122 neurons and 6,235,682 signed connections from the
MaleCNS data after the project's existing ≥5-synapse filter. It remains a leaky
integrate-and-fire approximation with the project's 0.4 weight scaling.

Only the 1,737 existing connections into four DNa02/DNa01 steering neurons are
eligible to change. Source families AOTU, PVLP, and other sources share gains
according to target neuron and excitatory/inhibitory sign. There are 22 gain
groups; the saved first-pilot winner changed 751 connections. Every gain is
bounded to 0.25–4 times its original strength. Connections are never added, deleted,
or sign-flipped. All other synapses retain their original model weights.

The fixed decoder sums left DNa02/DNa01 spikes minus right spikes. A difference
above 2 means left, below −2 means right, and otherwise means straight. Its
threshold and coefficients are never optimized. It cannot see food, walls,
reward, or the game state. Silencing those four steering neurons therefore
removes its steering signal.

Inputs are the existing LC10 food, LC4 danger, and LPLC2 danger-ahead channels.
The existing encoder includes a computed route to the moving tail. That is
engineered spatial preprocessing, not evidence that a fly can plan routes.

## Training procedure

1. Start from the original connection strengths, with all gains equal to one.
2. Propose two opposite random changes to bounded log-gains; also evaluate the
   current settings. Each proposal uses the same game and input-noise seeds.
3. Run actual Snake episodes. Every move simulates 100 ms of the whole brain at
   0.5 ms time steps. Membrane voltage, conductance, refractory state, and delayed
   signals carry between moves. Only episode boundaries reset brain state.
4. Keep the proposal with the greatest mean food score. Ties use accumulated
   game reward, then fewer collisions. Existing game rewards are food +1,
   death −1, and moving closer/farther ±0.1. No teacher actions or action masks
   are used. Training seeds rotate every four generations.
5. Select a saved winner using separate validation games. Freeze it before
   comparing with the original brain on previously unused paired test seeds.

This is reward-driven synaptic parameter search, inspired by the use of
[evolution strategies for reinforcement tasks](https://arxiv.org/abs/1703.03864).
It is **not implemented as dopamine-driven STDP**, and it is not a reconstruction
of biological fly learning. [Reward-modulated synaptic plasticity in spiking
networks](https://pubmed.ncbi.nlm.nih.gov/17444757/) is a research basis for a
possible later learning rule, rather than a claim about this implementation.

The existing mushroom-body feasibility checks found that game stimuli did not
activate Kenyon cells and directly driven MBONs did not activate DNa02/DNa01 in
this model. That motivated starting at existing steering synapses. It does not
show that a biological fly's mushroom body cannot affect movement.

## Run or resume

Install the normal simulator dependencies from AGENTS.md, plus optional Numba:

```powershell
.venv/Scripts/python -m pip install -r requirements-training.txt
.venv/Scripts/python scripts/train_brain.py --run outputs/my-brain --generations 30
.venv/Scripts/python scripts/train_brain.py --run outputs/my-brain --generations 100 --resume
.venv/Scripts/python scripts/train_brain.py --run outputs/my-brain --test --test-games 16 --test-moves 250
```

Resuming must repeat any nondefault training/validation settings from the first
command. The run saves parameters, mutation RNG, configuration, anatomical hash,
per-game results, and the selected winner. The test writes a lock before running:
once test results have been exposed, that run cannot continue training.

The first pilot command is:

```powershell
.venv/Scripts/python scripts/train_brain.py --run outputs/synaptic-pilot --generations 12 --train-games 2 --max-moves 100 --validation-games 4 --validation-moves 150 --validate-every 4
```

The completed pilot is now frozen after testing. To continue research on the
GPU laptop, start a **new run** from its saved brain with fresh seed ranges:

```powershell
.venv/Scripts/python scripts/train_brain.py --run outputs/synaptic-gpu-1 --initial models/brain-synaptic.npz --seed-offset 1000000 --generations 100 --train-games 8 --max-moves 200 --validation-games 16 --validation-moves 400 --device cuda
```

Use a new multiple of 1,000,000 for `--seed-offset` in each subsequent research
run. This separates training, validation, and final-test seeds from earlier
experiments. To resume, omit `--initial`, add `--resume`, and repeat the other
settings. For final testing, the seed offset is read from the saved run.

For another laptop, copy the repository, connectome data, and the complete run
directory. Create a new local Python environment; do not copy `.venv`. Install
a PyTorch build compatible with that laptop's GPU. Use `--device cuda --resume`
with the same run and settings. Changing device re-evaluates the baseline and
saved winner because CPU and CUDA noise streams and arithmetic can differ.
CUDA portability is implemented but has not been measured on this CPU-only PC.

For the offered RTX 5050/4070 laptops, start with the 4070, which matches the
project's documented tuning hardware. Both have 8 GB VRAM; the 4070 has 4,608
CUDA cores versus 2,560 on the 5050, while the 5050 uses newer GDDR7 memory.
Power limits and this sparse workload make an actual timing check necessary;
these specifications alone do not establish a speed ratio.
[NVIDIA laptop specifications](https://www.nvidia.com/en-us/geforce/laptops/compare/).
The 5050's Blackwell architecture needs a compatible recent PyTorch/CUDA build;
do not use this project's older CUDA 12.6 installation command for it.
[Current PyTorch CUDA guidance](https://pytorch.org/blog/pytorch-2-12-release-blog/).

The CPU uses an optional compiled full LIF kernel. It skips no neurons,
connections, or time steps. `--torch-kernel` uses the original PyTorch loop.
`scripts/verify_brain_kernel.py` compares all neuron outputs and state, including
changed/restored synapses. A 24-move whole-connectome check produced identical
spike counts, with state within 1e-5, and measured 3.54× speedup under that load.

## Use the saved brain in the web app

The app's **Use trained brain connections** button loads
`models/brain-synaptic.npz`. It starts fresh boards, uses the hardwired decoder,
and displays the changed-synapse provenance. The checkpoint is frozen during
play. The original brain/readouts remain available; selecting a trained/learning
readout or shuffled wiring restores the original brain first.

The separate **Live learning** panel continues to train an external readout.
It is not the synaptic trainer. For a fair manual comparison, restore the
original brain and choose **Nothing trained** to use the same hardwired decoder.

## First pilot results

The CPU pilot completed 12 generations, 72 candidate training episodes, and
16 validation episodes (including the original baseline): **3,092 actual
whole-brain moves in 900.2 seconds**. Generation 8 was selected. Validation
mean food increased from 3.0 to 4.0 across four games. The pilot checked
validation at generations 4, 8, and 12; the reusable trainer also checks at
generation 1 so short runs have a consistent validation schedule when resumed.

After selection, the frozen model and original brain each played 16 separate
test seeds (300000–300015), with matching neural-noise seeds and a 250-move
cap. All games ended by collision before that cap; none were censored.

| Continuous-brain test | Original synapses | Trained synapses |
| --- | ---: | ---: |
| Mean food | 2.5000 | 2.9375 |
| Collisions | 16/16 | 16/16 |
| Still alive at cap | 0/16 | 0/16 |

The paired difference is +0.4375 food per game (17.5% in this sample): 3 wins,
12 ties, and 1 loss. A paired bootstrap interval over these 16 seeds is
**−0.0625 to +1.1875** (95%, 20,000 resamples, seed 741), which includes zero.
This is a small, inconclusive indication of improvement, not established
generalization. It also excludes uncertainty across independently trained
models. The test ran 928 whole-brain moves in 283.2 seconds.

The existing trained external readout remains a stronger controller in prior
experiments, but those different runs are not a paired comparison with this
pilot. The purpose here was to make the learned parameters reside inside the
connectome and verify the complete signal loop with a frozen motor readout.

Artifacts:

- `models/brain-synaptic.npz`: the selected brain connection gains and anatomical identity.
- `models/brain-synaptic-training.json`: configuration, progress and validation scores.
- `models/brain-synaptic-test.json`: every held-out game, frozen-model hash and paired comparison.
- `models/brain-kernel-verification.json`: exact spike-count comparison with the original simulator.

42 Python tests and 4 frontend tests passed; the production build passed.
Checks cover synapse identity/sign/bounds, both CPU sparse caches, simulator
state/delays/lesions, model switching, and fixed-policy enforcement. A split
training run reproduced an uninterrupted run's parameters, random state,
history and winner; starting training after the final test was rejected.
The browser loaded the trained brain and restored the original successfully,
including a model switch while paused.

More generations do not guarantee improvement. Any score is simulated
performance, not a measurement of a living fly learning Snake.
