# Fly Snake

A leaky integrate-and-fire simulation of the whole MaleCNS v1.0 fruit-fly connectome plays Snake.
Game state -> stimulation of real sensory neuron types -> unmodified connectome -> a linear readout of the
real descending neurons picks left / straight / right. **Synaptic weights are never trained.**

## Run it
```sh
# one-time: Python 3.12 venv (uv), CUDA PyTorch, data download (~565 MB into data/, gitignored)
uv venv --python 3.12 && uv pip install torch --index-url https://download.pytorch.org/whl/cu126
uv pip install pandas pyarrow numpy scipy fastapi "uvicorn[standard]"
B=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome; mkdir -p data; cd data
curl -LO $B/body-annotations-male-cns-v1.0-minconf-0.5.feather -O $B/body-neurotransmitters-male-cns-v1.0.feather \
     -O $B/connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather

.venv/Scripts/python scripts/train_readout.py             # bank + readout + live-in-the-loop scores (~8 min on a 4070)
.venv/Scripts/python scripts/train_readout.py --shuffled  # scrambled-wiring control
.venv/Scripts/python -m uvicorn flybrain.server:app --port 8000   # brain server (GPU)
npm ci && npm run dev                                     # web page; connects to ws://localhost:8000/ws
```
No GPU? Everything runs on CPU, slower. The web page alone needs no Python: set `VITE_BRAIN_WS=ws://<gpu-laptop-ip>:8000/ws`
and start uvicorn with `--host 0.0.0.0` on the GPU machine.

## Map
| File | What |
|---|---|
| `flybrain/connectome.py` | MaleCNS feather files -> signed sparse edges (>=5 synapses; ACh +, GABA/Glu/histamine -). `Connectome.select(type=..., side=...)` |
| `flybrain/brain.py` | Batched torch LIF (Shiu et al. 2024 parameters), state `[N, B]`, `run(ms, stim_index, stim_level, record_index)` -> spike counts. `shuffled=True` = control |
| `flybrain/channels.py` | Senses: food = LC10 L/R, threat = LC4 L/R, threat ahead = LPLC2. Readout = all 1,314 descending neurons |
| `flybrain/snake.py` | Snake with relative actions, egocentric encoder (24 situations), heuristic teacher |
| `flybrain/readout.py` | `Policy` (linear, the only trained part), `HardwiredPolicy` (DNa02/DNa01 left-minus-right, nothing trained) |
| `flybrain/server.py` | FastAPI WebSocket live loop, modes real / hardwired / shuffled, manual stimulation |
| `scripts/bench_brain.py`, `probe_channels.py`, `train_readout.py` | sanity + speed; which senses steer; bank/train/evaluate |
| `src/lib/live.ts`, `src/App.tsx`, `src/components/Environment.tsx` | live adapter, layout, snake view. `BrainScene.tsx` takes `{time, values:[bodyId, 0..1][]}` |

## Findings so far (keep these honest in the pitch)
- `MALECNS_WEIGHT_SCALE = 0.4`: MaleCNS has ~2x the synapse counts FlyWire has, so Shiu's 0.275 mV/synapse gives runaway
  activity (any input -> 20k neurons). 0.35-0.45 is sparse and stimulus-specific. dt 0.5 ms matches dt 0.1 ms.
- Reproduces the Shiu headline on a different (male) connectome: sugar GRNs (LB3) -> MN9 fires; bitter (LB1) -> 0 Hz.
- Left LC10 -> DNa02 left ~250 Hz vs right 0 Hz (pursuit steering). Left LC4/LPLC2 -> giant fiber DNp01 ~390 Hz (escape),
  LC4 -> contralateral DNa01 (turn away). All emerge from wiring alone.
- 32 games, live sim in the loop: trained readout **17.4** mean score (max 33) · hardwired, nothing trained **2.75** · random **0.03**.
- Everything the viewer shows is *simulated / predicted* activity, never measured. Say so.

## Conventions
- Neurons are addressed by MaleCNS `bodyId` across the Python/JS boundary, by simulator row index inside Python.
- One owner per file during the hackathon; small commits to `main`.
- Template licence: keep `src/components/Attribution.tsx` visible and the credit in the README.

## Open work (suggested split)
1. **Brain/perf + science checks** - dt/scale sensitivity table, CPU fallback, response-bank replay mode for a demo without GPU.
2. **Neuro story** - richer senses (graded distance -> drive level, olfactory food), drive the Flybody panel from VNC motor neurons
   (`superclass == vnc_motor`, already simulated), figure for the pitch from `probe_channels.py`.
3. **Learning/eval** - ES fine-tune of the readout on score, bigger board, shuffled/hardwired/random comparison plot.
4. **Demo/frontend** - polish layout, DN raster, highlight stimulated + descending neurons in `BrainScene`, fallback video, slides.
