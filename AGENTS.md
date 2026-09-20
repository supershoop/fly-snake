# Fly Snake

A leaky integrate-and-fire simulation of the whole MaleCNS v1.0 fruit-fly connectome plays Snake.
Game state -> stimulation of real sensory neuron types -> unmodified connectome -> a linear readout of the
real descending neurons picks left / straight / right. **Synaptic weights are never trained.**

This file is the shared brief for every teammate and every coding agent (Claude Code reads it through `CLAUDE.md`).

## Run it
```sh
# one-time: Python 3.12 venv (uv), CUDA PyTorch, data download (~565 MB into data/, gitignored)
uv venv --python 3.12 && uv pip install torch --index-url https://download.pytorch.org/whl/cu126
uv pip install pandas pyarrow numpy scipy fastapi "uvicorn[standard]"
B=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome; mkdir -p data; cd data
curl -LO $B/body-annotations-male-cns-v1.0-minconf-0.5.feather -O $B/body-neurotransmitters-male-cns-v1.0.feather \
     -O $B/connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather; cd ..

.venv/Scripts/python -m uvicorn flybrain.server:app --port 8000 --host 0.0.0.0   # brain server (GPU; CPU works, slower)
npm ci && npm run dev                                                             # web page -> ws://localhost:8000/ws
```
(macOS/Linux: `.venv/bin/python`.) Trained readouts are committed in `models/`, so the demo runs without retraining.
First start builds `data/connectome-cache.npz` (~1 min). Tuned on an 8 GB RTX 4070; 16 brains use ~3 GB.

Other commands: `scripts/train_readout.py [--shuffled]` (response bank + readout + live-in-the-loop scores, ~8 min),
`scripts/probe_channels.py` (which senses steer), `scripts/bench_brain.py` (sanity + speed),
`scripts/live_learning_test.py` (offline test of on-stage learning, no GPU),
`FLY_RECORD=frames.jsonl` on the server records a session; `scripts/replay_server.py frames.jsonl` replays it with no GPU.
Web page against someone else's server: `VITE_BRAIN_WS=ws://<their-ip>:8000/ws npm run dev`.

## Map
| File | What |
|---|---|
| `flybrain/connectome.py` | MaleCNS feather files -> signed sparse edges (>=5 synapses; ACh +, GABA/Glu/histamine -). `Connectome.select(type=..., side=...)` |
| `flybrain/brain.py` | Batched torch LIF (Shiu et al. 2024 parameters), state `[N, B]` = B independent brains. `run(ms, stim_index, stim_level, record_index)` -> spike counts. `set_lesion(mask [N] or [N,B])`, `resize(batch)`, `shuffled=True` = control |
| `flybrain/channels.py` | Senses: food = LC10 L/R, threat = LC4 L/R, threat ahead = LPLC2. Readout = all 1,314 descending neurons |
| `flybrain/snake.py` | `Arena`: any number of fly/human snakes on one board, relative actions, rewards, egocentric encoder (24 situations), heuristic `teacher` |
| `flybrain/readout.py` | `Policy` (linear, fitted offline), `OnlineLearner` (same readout, learns live from reward), `HardwiredPolicy` (DNa02/DNa01 left-minus-right, nothing trained) |
| `flybrain/server.py` | FastAPI WebSocket live loop: layouts, policies, per-fly lesions, sensor input, feedback, human control |
| `src/lib/live.ts` | Frame types + WebSocket hook. `src/App.tsx` layout/controls, `src/components/Environment.tsx` boards, `BrainScene.tsx` takes `{time, values:[bodyId, 0..1][]}` |

## WebSocket protocol (`ws://host:8000/ws`, JSON)
Client -> server, any combination of keys in one message (applied between moves):
| Key | Meaning |
|---|---|
| `{"layout": "solo"\|"swarm"\|"versus"\|"arena"}` | 1 fly · 16 flies on 16 boards · fly vs human on one board · 8 flies on one board |
| `{"policy": "trained"\|"hardwired"\|"learning"}`, `{"wiring": "real"\|"shuffled"}` | who picks the move; scrambled-wiring control |
| `{"learning": "reset"}` | blank readout for live learning (rewards: food +1, death -1, closer/farther +-0.1) |
| `{"feedback": 1\|-1, "fly": i or omitted for all}` | human reward / punishment added to the last move's reward (learning policy only) |
| `{"lesion": {"fly": i\|null, "types": ["DNa02", "LC10.*"]}}` | silence neuron types (regex, full match on annotation `type`); `null` = every fly; `[]` heals |
| `{"sensor": {"danger_ahead": 0.8}}` | hardware input: drive 0..1 **added** to the game's senses, goes stale after 0.6 s, so resend at >= 5 Hz; ultrasonic clients should use the send-only `/ws/hardware` endpoint |
| `{"stimulate": {"food_L": 1}\|null}` | manual override of all senses; game holds still while set |
| `{"human": "up"\|"down"\|"left"\|"right"}`, `{"select": i}`, `{"paused": bool}` | human snake; which fly's brain is shown; pause |

Server -> client, one frame per move (see `LiveFrame` in `src/lib/live.ts`): `arenas[]` (boards, foods, snakes), `flies[]`
(per fly: `channels`, `action`, `probabilities`, `reward`, `steer` = Hz of DNa02/DNa01/DNp01 L/R, `lesion`), `selected`,
`values` (selected fly's brain activity by bodyId), `learning {moves, games, scores[]}`, `activeNeurons`, `sensor`, `manual`.
Channel names: `food_L, food_R, danger_L, danger_R, danger_ahead`. **If you change the protocol, update this table and `live.ts`.**

## Findings so far (keep these honest in the pitch)
- `MALECNS_WEIGHT_SCALE = 0.4`: MaleCNS has ~2x the synapse counts FlyWire has, so Shiu's 0.275 mV/synapse gives runaway
  activity (any input -> 20k neurons). 0.35-0.45 is sparse and stimulus-specific. dt 0.5 ms matches dt 0.1 ms.
- Reproduces the Shiu headline on a different (male) connectome: sugar GRNs (type `LB3*`) -> MN9 fires; bitter (`LB1*`) -> 0 Hz.
- Left LC10 -> DNa02 left ~250 Hz vs right 0 Hz (pursuit steering). Left LC4/LPLC2 -> giant fiber DNp01 ~390 Hz (escape),
  LC4 -> contralateral DNa01 (turn away). All emerge from wiring alone.
- 32 games, live sim in the loop: trained readout **17.4** mean score (max 33) · hardwired, nothing trained **2.75** · random **0.03**.
- Scrambled wiring (`scripts/scrambled_check.py`, 16 games): with a readout trained on it, the scrambled network scores
  **14.3** vs **18.8** for the real wiring. So a trained readout can play through almost any network that keeps left and right
  inputs separable - **scrambled-vs-real with the trained readout is NOT evidence that the wiring matters.** The evidence is the
  nothing-trained policy: real wiring 2.0-2.75, scrambled 0.00, and the lesion table below. (An earlier 0.00 for the scrambled
  trained readout was a bug: the evaluation used a different shuffle than the training bank. Fixed: `SHUFFLE_SEED`.)
  Carry-over between moves is harmless: real wiring 18.8 with carry-over vs 20.1 with the brain reset every move.
- Live learning, 16 real brains sharing one blank readout: last-20 average 3.2 at 47 s, 5.1 at 94 s, 8.2 at 141 s (2.1 moves/s).
  Offline estimate (`live_learning_test.py`) reaches ~19. One fly alone learns too slowly; rates above 0.01 collapse.
- Path search in the connectome (signed 2-hop paths): LC10 has **no direct synapses** onto DNa02; the food signal crosses
  ~6 cells in the anterior optic tubercle (AOTU025, AOTU012, AOTU015 - the known pursuit pathway). LC4 -> giant fiber DNp01 is
  direct (3,782 synapses, a textbook circuit). LC4 -> contralateral DNa01 runs through PVLP141 and PVLP137.
  `scripts/lesion_scores.py` silences these and scores play; the web page's Lesion lab does it live (`src/components/LesionLab.tsx`).
- Lesion table (`lesion_scores.py`, 8 games each, live sim). **Nothing-trained policy:** intact 2.00 · 12 AOTU relay cells
  silenced 0.38 · 2 DNa02 cells silenced 0.00 (dies in 6 moves) · 12 random neurons 2.00 · 2,000 random neurons 2.25.
  **Trained readout:** 19.75 intact vs 16-20 for every lesion including the random controls (12 random: 16.1), i.e. no lesion
  effect distinguishable from noise at 8 games - the readout reads many descending neurons and compensates. One outlier
  (AOTU025 alone: 11.0) needs more games before anyone interprets it.
- With DNa02 silenced the trained readout still steers (it uses other descending neurons); the hardwired policy cannot.
- Known weakness to answer: the game shows the brain only 24 distinct situations and the readout copies a rule-based teacher,
  so "the readout plays, the brain relabels" is a fair criticism. Lesions, the untrained mode and real vision are the answers.
- Everything the viewer shows is *simulated / predicted* activity, never measured. Say so.

## Conventions
- Neurons are addressed by MaleCNS `bodyId` across the Python/JS boundary, by simulator row index inside Python.
- Work on `main`, small commits, pull often. Shared files (`server.py`, `App.tsx`, `live.ts`): keep edits small and additive;
  put new work in new files where possible (a new component, a new script, a new module).
- Template licence: keep `src/components/Attribution.tsx` visible and the credit in the README.
- No faked results in the demo. Fallbacks are labelled: fixed seed, "load pre-trained readout", recorded video.

## Tracks (pick one each; ~17 h left at the time of writing)
| # | Track | What you build | Mostly touches | Effort | Risk | Needs |
|---|---|---|---|---|---|---|
| 1 | **Live training show** (core demo) | Polish "Learn live": score-over-time chart, make it faster (50 ms window or dt 1 ms -> needs a new bank + check scores), dopamine flash (stimulate `PAM*` on reward, `PPL1*` on punishment - they exist in the annotations, class `DAN`), learn-by-demonstration fallback, "load pre-trained" button | `readout.py`, new chart component | 3-4 h | low | GPU |
| 2 | **Lesion lab + controls** | Lesion UI beyond presets (search any type), `scripts/lesion_scores.py` = 32 games per lesion -> table, find the neurons *between* LC10 and DNa02 (path search in `connectome`) and lesion those, resolve the scrambled-wiring question (reset brain every move in both conditions and compare) | new script, new component | 3-4 h | low | GPU |
| 3 | **Hardware** (Pi Zero 2 W, ultrasonic sensor, RFID shield) | Pi runs a small Python WebSocket client on WiFi: distance + approach speed -> `{"sensor": {"danger_ahead": x}}` at 10 Hz so a judge's hand makes the giant fiber fire and the snake dodge. RFID tags as cards: "sugar"/"bitter" tag -> stimulate `LB3*`/`LB1*` and show MN9 (needs a small server addition), or tags = reward / punish / lesion cards. HC-SR04 echo is 5 V: use a voltage divider into the Pi's 3.3 V GPIO | new `hardware/` folder, tiny `server.py` addition | 2-4 h | low-medium | Pi; no GPU (use `replay_server.py` or a teammate's server) |
| 4 | **Arena + human vs fly + frontend** | Make versus/arena fun: countdown, win condition, scores, fly colours, leaderboard of lesioned flies in swarm, brain panel highlights for stimulated + descending neurons, spike audio (giant fiber clicks), overall layout polish, fallback video | `src/`, small `snake.py` rules | 4-5 h | low | any GPU or replay |
| 5 | *Stretch:* **Real vision** | Paint the board onto the eye: stimulate medulla columns by `assignedOlHex1/2` retinotopy instead of LC10/LC4 directly. First a 1 h feasibility check: which types have hex coordinates; does a left-eye patch give left-biased LC10/LC4/DNa02? Go/no-go after that. Raw photoreceptors will likely fail (graded, non-spiking; motion detection needs timing the LIF lacks) | new `flybrain/vision.py`, `channels.py` | 1 h check, then 4-8 h | high | GPU |
| 6 | *Stretch:* **Learning inside the brain** | Mushroom body: reward -> PAM dopamine, punishment -> PPL1, plasticity only at Kenyon cell -> MBON synapses. 30-45 min check first: do Kenyon cells respond to our stimuli at all, and does MBON activity reach the steering neurons? | new module | check, then 4-6 h | high | GPU |

Suggested default for four people: 1, 2, 3, 4 - and whoever finishes first runs the feasibility check for 5 or 6.
Pitch + slides: owner of track 2 (they hold the evidence), with everyone's numbers.
