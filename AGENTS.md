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

Other commands: `scripts/train_readout.py [--shuffled]` (long-game target training + transition response bank + live scores),
`scripts/train_readout.py --trials 32 --evaluation bank` (CPU-friendly training with explicitly labelled bank estimates),
`scripts/evaluate_survival.py --games 4 --max-moves 400` (saved readout vs original, continuous brain),
`scripts/train_evolution.py --generations 2000 --validation-only --run outputs/evolution-new` (reward-driven readout evolution, CPU-friendly bank estimates; optional `requirements-training.txt`; see `docs/EVOLUTION.md`),
`scripts/evaluate_evolution.py --model models/readout-evolved-ridge100.npz --games 4 --max-moves 250 --game-seed 90000` (same-sensing continuous-brain comparison with the current model),
`scripts/probe_channels.py` (which senses steer), `scripts/bench_brain.py` (sanity + speed),
`scripts/live_learning_test.py` (offline test of on-stage learning, no GPU),
`FLY_RECORD=frames.jsonl` on the server records a session; `scripts/replay_server.py frames.jsonl` replays it with no GPU.
Web page against someone else's server: `VITE_BRAIN_WS=ws://<their-ip>:8000/ws npm run dev`.

## Map
| File | What |
|---|---|
| `flybrain/connectome.py` | MaleCNS feather files -> signed sparse edges (>=5 synapses; ACh +, GABA/Glu/histamine -). `Connectome.select(type=..., side=...)` |
| `flybrain/brain.py` | Batched torch LIF (Shiu et al. 2024 parameters), state `[N, B]` = B independent brains. `run(ms, stim_index, stim_level, record_index)` -> spike counts. `set_lesion(mask [N] or [N,B])`, `resize(batch)`, `shuffled=True` = control |
| `flybrain/cpu_synapses.py` | Faster single-brain CPU propagation: multiply only firing presynaptic columns; full weights and time steps preserved. Batched/GPU simulation retains its original kernel. |
| `flybrain/channels.py` | Senses: food = LC10 L/R, threat = LC4 L/R, threat ahead = LPLC2. Readout = all 1,314 descending neurons |
| `flybrain/snake.py` | `Arena`: any number of fly/human snakes on one board, relative actions, rewards, egocentric encoder (24 situations); legacy sensing available via `lookahead=False` |
| `flybrain/navigation.py` | Threat observations include loss of a route to the moving tail after a move, accounting for growth; never overrides actions |
| `flybrain/training.py`, `flybrain/response_bank.py` | Learn target preferences from long-game food scores; collect real DN responses with state carried between inputs |
| `flybrain/evolution.py`, `flybrain/evolution_game.py`, `scripts/train_evolution.py` | Mutation/selection of random readouts from game rewards, compiled canonical solo rollouts, separate validation and resumable checkpoints; scores are bank estimates |
| `flybrain/readout.py` | `Policy` (linear, fitted offline), `OnlineLearner` (same readout, learns live from reward), `HardwiredPolicy` (DNa02/DNa01 left-minus-right, nothing trained) |
| `flybrain/feedback.py`, `src/components/LiveTraining.tsx` | Reward/punishment for a displayed move; delayed feedback trains saved decisions, with receipts and stale-decision rejection |
| `flybrain/vision.py`, `src/components/FlyView.tsx` | Retinotopic encoder: each LC10/LC4/LPLC2 cell's viewing direction from the eye-map position of its columnar inputs; board painted as a 1-D horizon onto those cells. `VisionDisplay` = signal-path overlay data (nodes/edges/rates) for the brain view, `VisionUntrained` = the branch's original untrained rule |
| `flybrain/server.py` | FastAPI WebSocket live loop: layouts, policies, per-fly lesions, sensor input, feedback, human control |
| `src/lib/live.ts` | Frame types + WebSocket hook. `src/App.tsx` layout/controls, `src/components/Environment.tsx` boards, `BrainScene.tsx` takes `{time, values:[bodyId, 0..1][]}` |

## WebSocket protocol (`ws://host:8000/ws`, JSON)
`{"hello": true}` returns `{"hello": {"types": [...], "feedbackUrls": ["http://<LAN-IP>:8000/feedback/", ...]}}`.
Training shows a QR code for this same server's phone controller. Loopback hosts are replaced with LAN addresses;
`FLY_FEEDBACK_URL` overrides the advertised URL for a public HTTPS reverse proxy. Run the server with `--host 0.0.0.0`
and connect phones to the same Wi-Fi/hotspot for local demos. No frontend server access is needed on phones.
On campus networks with device isolation, `scripts/audience_gateway.py` exposes only phone assets and
`/feedback/ws` on loopback port 8002 for an HTTPS tunnel. It forwards to the existing brain's feedback socket;
it does not start another experiment or expose host `/ws` controls. `VITE_FEEDBACK_URL` overrides the QR URL
at frontend build/dev startup without restarting the brain. Temporary tunnel URLs must be refreshed on restart.
The feedback-only socket at `/feedback/ws` accepts exactly `{id: string, feedback: number, fly: number|null, move: number}`.
It sends `{frame: {move, policy, manual, paused, selected, arenas, flies, feedback: {positive, negative}}}` and a private
`{id, receipt}` for each request, using the existing applied/rejected receipt shape. Feedback is applied between moves
to the shared readout; stale moves, other mode commands, paused/manual play and non-learning modes are rejected.

Private operator control (`flybrain/operator_control.py`, `flybrain/operator_ui/index.html`) is enabled by
`FLY_OPERATOR_KEY`. `/operator/ws` first requires `{"key": "<secret>"}`, then accepts `{"status": true}` or
`{"preset": "crash"|"untrained"|"legacy"|"evolved"|"best"|"release"}`. It returns `{state: {presets, active,
supported, mode, paused, move, reason}}` and `{result: {ok, state?, reason?}}`. Only the authenticated operator
gets selection state; public frames/hello and UI do not link to this page or announce switches. Gateway exposure
is opt-in via `--operator-ws ws://127.0.0.1:<brain-port>/operator/ws`. The page accepts a key in a URL fragment
for a private bookmark, then removes the fragment and authenticates over WebSocket. No key is committed.
Five readout presets require original real wiring and Trained/Training mode; they do not promise a perfect
brain. The crash preset is a deliberately straight-biased decoder. The others load saved readouts. Selection
applies between moves, keeps brain simulation and boards running, expires old feedback, and trains a copy
in Training. Release restores the cached host policy. Explicit host model/mode choices clear the override.

Client -> server, any combination of keys in one message (applied between moves):
| Key | Meaning |
|---|---|
| `{"layout": "solo"\|"swarm"\|"versus"\|"arena"}` | 1 fly · 16 flies on 16 boards · fly vs human on one board · 8 flies on one board |
| `{"policy": "trained"\|"instinct"\|"hardwired"\|"learning"}`, `{"wiring": "real"\|"shuffled"}` | who picks the move; scrambled-wiring control |
| `{"learning": "reset"\|"pretrained"}` | switch to live learning from a blank readout or a copy of the trained readout for the current wiring; saved models are unchanged (automatic rewards: food +1, death -1, closer/farther +-0.1) |
| `{"feedback": value, "fly": i\|null, "move": moveId}` | reward/punishment in [-1, 1], excluding zero, for a displayed decision (learning policy only); omitted/null fly targets all eligible flies; omitted move uses latest saved decision. The last 64 decisions are retained; experiment/model changes invalidate them. Receipts report applied or rejected feedback. |
| `{"lesion": {"fly": i\|null, "types": ["DNa02", "LC10.*"]}}` | silence neuron types (regex, full match on annotation `type`); `null` = every fly; `[]` heals |
| `{"encoder": "channels"\|"retina"}` | how the game reaches the brain: 5 on/off channels (default), or the connectome-derived retinotopic eye (`flybrain/vision.py`; trained policy = `models/readout-vision.npz`, instinct and hardwired also work) |
| `{"events": bool}` | taste on eating / pain on dying (default on), see Findings |
| `{"deathHold": seconds}` | sent by the page: how long its death animation lasts; the game holds that long after the displayed fly dies (0-5 s) |
| `{"sensor": {"danger_ahead": 0.8}}` | external input: drive 0..1 **added** to the game's senses, goes stale after 0.6 s, so resend at >= 5 Hz (built for the dropped hardware track; still works) |
| `{"stimulate": {"food_L": 1}\|null}` | manual override of all senses; game holds still while set |
| `{"human": "up"\|"down"\|"left"\|"right"}`, `{"select": i}`, `{"paused": bool}` | human snake; which fly's brain is shown; pause |

Server -> client, one frame per move (see `LiveFrame` in `src/lib/live.ts`): `arenas[]` (boards, foods, snakes), `flies[]`
(per fly: `channels`, `action`, `probabilities`, `reward`, `feedbackEligible`, `steer` = Hz of DNa02/DNa01/DNp01 L/R, `lesion`), `selected`,
`move` (monotonically increasing decision ID), `values` (selected fly's brain activity by bodyId),
`learning {moves, games, scores[], feedback: {positive, negative, last}}`, `activeNeurons`, `sensor`, `manual`,
`encoder`, `events`, `silenced` / `silencedTotal` / `silencedByFly` (bodyIds of silenced, drawn cells), `vision {pathway: {node: Hz}, view}`;
per fly also `event` (`"taste"`, `"pain"` or null = what it feels during this window). A frame with `eventOnly: true` repeats the
game state with fresh brain activity: the pain burst shown while the death hold runs. The one-off `{"hello": true}` reply carries
`types` (lesion search), `feedbackUrls` and `vision` (static: pathway nodes/edges by bodyId, eye columns, retina cells; `VisionStatic` in `live.ts`).
Feedback `last` is null, `{status: "applied", value, fly, move, targets[]}`, or `{status: "rejected", reason}`.
Live readout changes last for the server session. Feedback controls require an unpaused game without manual sensory override.
Channel names: `food_L, food_R, danger_L, danger_R, danger_ahead`. Game danger channels indicate immediate collision
or a move cutting off the path to the snake's moving tail. This is an engineered spatial observation, not measured fly perception.
**If you change the protocol, update this table and `live.ts`.**

## Findings so far (keep these honest in the pitch)
- `MALECNS_WEIGHT_SCALE = 0.4`: MaleCNS has ~2x the synapse counts FlyWire has, so Shiu's 0.275 mV/synapse gives runaway
  activity (any input -> 20k neurons). 0.35-0.45 is sparse and stimulus-specific. dt 0.5 ms matches dt 0.1 ms.
- Reproduces the Shiu headline on a different (male) connectome: sugar GRNs (type `LB3*`) -> MN9 fires; bitter (`LB1*`) -> 0 Hz.
- Left LC10 -> DNa02 left ~250 Hz vs right 0 Hz (pursuit steering). Left LC4/LPLC2 -> giant fiber DNp01 ~390 Hz (escape),
  LC4 -> contralateral DNa01 (turn away). All emerge from wiring alone.
- Historical baseline with adjacent-cell sensing, 32 games, live sim in the loop: trained readout **17.4** mean score (max 33) · hardwired **2.75** · random **0.03**.
- Survival retraining: 64 held-out games, 1,600-move limit, **sampled response-bank estimates**: original **11.08**, new senses alone **14.30**, retrained **54.34** mean food score; collisions **64 / 64 / 0**. Retrained games still include 45 starvation timeouts. These are not live-brain scores. See `docs/TRAINING.md` and `models/readout-real-training.json`.
- Separate continuous-brain smoke check, four new seeds, 275 recorded moves: original mean **12.0**, **4/4 collisions**; retrained mean **24.5**, **1/4 collisions**, three games still alive at the cap. This is a small censored comparison, not a guarantee. See `models/readout-real-live-evaluation.json`.
- Reward-driven evolution from random readouts: 2,000 generations, 227,784 actual training games. On a fresh neural-response bank and 128 new games, evolved **32.87** vs current **36.94** mean food (bank estimates). A separate continuous-brain check, four paired seeds with a 250-move cap, gave evolved **18.0** vs current **21.5**, with **2 vs 0 collisions** and **2 vs 3 still alive**. The experimental readout did not replace the default. See `docs/EVOLUTION.md` and `models/readout-evolved-*.json`.
- Scrambled wiring (`scripts/scrambled_check.py`, 16 games): with a readout trained on it, the scrambled network scores
  **14.3** vs **18.8** for the real wiring. So a trained readout can play through almost any network that keeps left and right
  inputs separable - **scrambled-vs-real with the trained readout is NOT evidence that the wiring matters.** The evidence is the
  nothing-trained policy: real wiring 2.0-2.75, scrambled 0.00, and the lesion table below. (An earlier 0.00 for the scrambled
  trained readout was a bug: the evaluation used a different shuffle than the training bank. Fixed: `SHUFFLE_SEED`.)
  Re-measured fairly (`scripts/untrained_control.py`, 32 games, nothing trained): real wiring **2.28**, every game eats, steering
  neurons fire ~35 spikes/move; three independent scrambles **0.00 / 0.00 / 0.09**, steering neurons <1 spike/move, dead in ~6 moves.
- Learning from reward alone, offline on the response banks (`live_learning_test.py --bank bank-real|bank-shuffled`, 16 flies,
  1,200 moves each, identical rule): real wiring reaches **17-20** at every learning rate tried (0.001-0.01); scrambled wiring
  plateaus at **~5** at every rate and never reaches an average of 8. With 5 seeds at rate 0.005 (`--seeds 5`): real wiring
  19.9, 22.5, 21.2, 20.7 and one stalled run at **5.2**; scrambled 5.3, 5.3, 5.8, 5.8, 6.0. So say "usually learns" (4 of 5), and
  for the stage use a rehearsed fixed seed or the pre-trained fallback. Caveats: one scrambled network, offline only.
  **Re-measured with the current look-ahead senses, 10 seeds each (same command plus `--seeds 10`): real wiring 22.0 +- 2.7, all
  10 runs pass an average of 15 (the one-in-five stall seen with the old adjacent-cell senses is gone); scrambled wiring 5.1 +- 0.8,
  0 of 10 pass an average of 8.** An entropy bonus (`OnlineLearner(entropy=0.03)`, `--entropy`) tightens the real-wiring spread to
  +- 1.8 and does not rescue the scrambled brain (5.6); it is off by default. Still one scrambled network and offline only.
  On the page: train with "Training", then "Training - scrambled" - the learning chart keeps both curves.
  Do NOT demo "readout trained on the real brain, run on the scrambled brain": a decoder fails on any network it was not trained
  on, so that break says nothing about the wiring.
  Carry-over between moves is harmless: real wiring 18.8 with carry-over vs 20.1 with the brain reset every move.
- Live learning, 16 real brains sharing one blank readout: last-20 average 3.2 at 47 s, 5.1 at 94 s, 8.2 at 141 s (2.1 moves/s).
  Offline estimate (`live_learning_test.py`) reaches ~19. One fly alone learns too slowly; rates above 0.01 collapse.
- Path search in the connectome (signed 2-hop paths): LC10 has **no direct synapses** onto DNa02; the food signal crosses
  ~6 cells in the anterior optic tubercle (AOTU025, AOTU012, AOTU015 - the known pursuit pathway). LC4 -> giant fiber DNp01 is
  direct (3,782 synapses, a textbook circuit). LC4 -> contralateral DNa01 runs through PVLP141 and PVLP137.
  `scripts/lesion_scores.py` silences these and scores play; the web page's Lesion lab does it live (`src/components/LesionLab.tsx`).
- **Normal mode = `InstinctPolicy` (nothing trained).** Diagnosis (`scripts/instinct_analysis.py`, offline on the response bank):
  the old rule (DNa02 + DNa01 left-minus-right) dies because pursuit always wins - food on a blocked side drives DNa02 to ~240 Hz
  while the turn-away neuron DNa01 stays at 0-10 Hz, so it turns into the obstacle 100% of the time. The brain does register
  the threat: the giant fiber DNp01 fires ~370 Hz on the blocked side vs ~78 Hz. Instinct = the same pursuit steering plus two
  giant-fiber overrides: *veto* (never turn toward the side whose giant fiber fires >100 Hz harder) and *dodge* (both giant fibers
  >150 Hz and nothing pulling sideways -> turn away from the louder one). Thresholds are hand-set, not fitted - say so.
  Offline, 200 games: old rule 4.3 food / 39 moves; instinct 6.8 / 64; teacher 22.5 / 210.
  **Double dissociation**, live, 16 real brains, ~3 min: intact ~75 moves per life, last games 3-18 food; DNa02 silenced ~276 moves
  per life but 0-1 food (wanders, still dodges); DNp01 silenced ~25 moves per life, ~2 food (still seeks food, crashes 3x as often).
  DNa01 silenced: no effect. Remaining weakness: a threat straight ahead saturates both giant fibers (~380 Hz), hiding which side
  is worse; lowering threat intensity keeps them graded but still symmetric (`scripts/threat_intensity_probe.py`) - better senses,
  not a cleverer rule, are the fix.
- Page: modes are Trained / Normal / Scrambled / Training (Scrambled = scrambled wiring + the same instinct rule, the fair control).
  Click boards to pick flies (multi-select); the lesion lab appears for the picked flies and their silenced cells are drawn as orange
  rings in the brain view (`silencedByFly` in the frame). Live learning shows only in Training. No pause, no manual stimulation.
- **Lesion table, instinct rule, current senses** (`scripts/lesion_scores.py --games 16`, live brain, food +- s.e.m. / moves survived).
  Nothing trained: intact 5.9 +- 1.0 / 63 · AOTU relays (12 cells) **0.8 +- 0.2** / 132 · DNa02 (2 cells) **1.1 +- 0.3** / 170 ·
  DNa02 + DNa01 0.4 / 160 · giant fiber DNp01 (2 cells) 2.2 +- 0.1 / **25** · PVLP relays 5.7 / 59 · AOTU025 alone 6.9 / 63 ·
  controls: 12 random neurons 5.5 / 61, 2,000 random neurons 6.4 / 68. Double dissociation: no steering -> no food but long survival
  (it wanders and still dodges); no giant fiber -> still seeks food, dies 2.5x sooner. Trained readout (Hang's current `readout-real`):
  intact 24.9 +- 1.8, controls 25.8 and 23.8, **AOTU relays 15.0 +- 2.0** - so the trained readout does lean on the real pursuit
  pathway; every other lesion is within noise (DNa02 24.9, DNp01 25.8, PVLP 20.3 +- 2.5, AOTU025 alone 25.1 - the old 11.0 outlier was noise).
  The older table below used the previous untrained rule and previous senses.
- Lesion table (`lesion_scores.py`, 8 games each, live sim). **Nothing-trained policy:** intact 2.00 · 12 AOTU relay cells
  silenced 0.38 · 2 DNa02 cells silenced 0.00 (dies in 6 moves) · 12 random neurons 2.00 · 2,000 random neurons 2.25.
  **Trained readout:** 19.75 intact vs 16-20 for every lesion including the random controls (12 random: 16.1), i.e. no lesion
  effect distinguishable from noise at 8 games - the readout reads many descending neurons and compensates. One outlier
  (AOTU025 alone: 11.0) needs more games before anyone interprets it.
- Learning INSIDE the brain (`scripts/mushroom_body_check.py`): our visual stimuli activate **0 of 4,064 Kenyon cells**; a smell
  (olfactory sensory neurons) activates ~2,500 and drives 60 MBONs at ~115 Hz. But driving 49 MBONs directly at ~70 Hz gives
  **0 Hz in DNa02 and DNa01** - mushroom-body output does not reach steering in this model (anatomical LAL relays exist, mixed
  sign, too weak). So plasticity at the 33,496 KC->MBON connections could not change how the snake steers. "Learn live" trains the
  readout only - say so. Still possible as a side demo: odour conditioning with PAM/PPL1 dopamine and a KC->MBON rule, visible
  as a changed MBON response, not as changed steering.
- Vision (`scripts/vision_feasibility.py`): painting pixels onto the eye's columns **fails** - the whole left eye lit gives LC10
  < 1 Hz and no steering; even a 3x optic-lobe gain gives 5 Hz and the wrong size tuning. Expected: the optic lobe is mostly
  graded, non-spiking cells. So `flybrain/vision.py` enters at the detector layer with connectome-derived viewing directions.
  Eye-map axes from soma positions: hex1-hex2 large = front, hex1+hex2 large = dorsal; lamina and medulla show opposite signs,
  i.e. the data reproduce the optic chiasm flip. `scripts/vision_tuning.py`: food 60-90 deg to one side -> ipsilateral DNa02
  93-207 Hz (nearer = more), food dead ahead -> silent although most cells are lit there; giant fiber 17-60 Hz for an obstacle
  4 cells away, 130-290 Hz at 1.5 cells, biased to its side. None of that is trained.
- Retina encoder scores (`scripts/train_vision.py`, 32 games, measured on the vision branch before the instinct rule existed):
  trained readout **9.2** (87% teacher match), its untrained rule **1.8**. Lower than the channel encoder but continuous input,
  not 24 situations. Caveats: only 142 left vs 228 right LC10 cells get a viewing direction; the -15..150 deg field per eye is assumed.
- **Taste and pain** (`scripts/event_probe.py`, server `EVENT_SOURCES`). Eating stimulates the sugar taste neurons (`LB3*`, taste pegs);
  dying stimulates the heat/humidity receptors (`HRN_*`, `TRN_*`). Live: ordinary move ~1,750 active neurons, MN9 8 Hz, PPL1 0 Hz;
  taste move MN9 **79 Hz**; pain move ~**6,700** active neurons, PPL1 (punishment dopamine) **83 Hz**. Two things we had to do, both
  worth saying out loud: (1) visual input suppresses feeding in this model - sugar alone drives MN9 43-110 Hz, with LC10 on ~5 Hz,
  with a threat 0 Hz - so a feeding fly pauses for one move with no visual input; (2) the pain burst is **self-sustaining**:
  ~7,000 neurons and PPL1 ~144 Hz ring on indefinitely after 100 ms of input, so the dead fly's brain is reset to rest (a new
  life starts quiet). Reward dopamine (PAM) does NOT respond to sugar here: say "tastes the food", never "feels rewarded".
- With DNa02 silenced the trained readout still steers (it uses other descending neurons); the hardwired policy cannot.
- Known weakness to answer: the game still shows only 24 distinct situations. Target preferences start from a heuristic and are
  optimized on full-game scores; the encoder now computes tail connectivity. This is engineered spatial preprocessing, not evidence
  that a biological fly plans routes. "The readout plays, the brain relabels" remains a fair criticism.
- Everything the viewer shows is *simulated / predicted* activity, never measured. Say so.

## Keeping the laptop alive (read before long GPU runs)
The demo laptop hard-crashed once, most likely from heat: hours of GPU load, a 16-brain swarm plus evidence runs, RAM exhausted.
- **One heavy GPU job at a time.** Stop the brain server before running an evidence script, and the other way round. A brain
  process needs 3-4 GB of RAM; the laptop has 15.6 GB shared with browsers and agent sessions.
- **The simulation idles while every page is hidden** (tab minimised or in the background; the page sends `{"visible": bool}`).
  Losing focus alone does not pause it, so switching to slides during a demo is safe. Clients that never report count as watching.
- **Thermal guard** (`flybrain/thermal.py`): reads the GPU temperature every 5 s, logs it to `outputs/gpu-temps.csv`, adds
  `thermal {gpu, state}` to every frame. At 80 C it leaves 0.35 s gaps between moves ("slow"), at 87 C it holds until 78 C
  ("cooling", banner on the page). `FLY_THERMAL_GUARD=slow` never holds, `=off` only logs; thresholds via `FLY_THERMAL_SLOW`,
  `FLY_THERMAL_PAUSE`, `FLY_THERMAL_RESUME`. Only the GPU is watched. One fly runs at ~45 C; the swarm is the hot layout.
- Lesion lab offers three lesions only: steering DNa02, giant fiber DNp01, food relays AOTU025/012/015. The any-type search is gone
  from the page (the `{"lesion": ...}` message still accepts any type).

## Conventions
- Neurons are addressed by MaleCNS `bodyId` across the Python/JS boundary, by simulator row index inside Python.
- Work on `main`, small commits, pull often. Shared files (`server.py`, `App.tsx`, `live.ts`): keep edits small and additive;
  put new work in new files where possible (a new component, a new script, a new module).
- Template licence: keep `src/components/Attribution.tsx` visible and the credit in the README.
- No faked results in the demo. Fallbacks are labelled: fixed seed, "load pre-trained readout", recorded video.

## Status (replaces the old track list)
Done: instinct policy (Normal), lesion lab with pick-to-lesion + silenced-cell markers, signal-path overlay, retinotopic vision
encoder + fly's-eye view, taste/pain events, live learning + phone audience feedback (QR), human vs fly with buffered input,
death scene + hold, evolved readouts (see `docs/EVOLUTION.md`).
Dropped: the hardware track (Pi, ultrasonic sensor, RFID). Its replacement is the phone reward/punish buttons. The `sensor`
message and the unmerged branch `origin/hardware-track-pi-websocket` remain if anyone wants them.
No-go after feasibility checks: painting pixels onto the eye; mushroom-body learning that changes steering.
Open: see the project owner's list; the evidence scripts are `lesion_scores.py`, `untrained_control.py`, `scrambled_check.py`,
`live_learning_test.py`, `instinct_analysis.py`.
