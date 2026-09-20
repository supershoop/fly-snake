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
| `flybrain/server.py` | FastAPI WebSocket live loop: layouts, policies, per-fly lesions, sensor input, feedback, human control |
| `src/lib/live.ts` | Frame types + WebSocket hook. `src/App.tsx` layout/controls, `src/components/Environment.tsx` boards, `BrainScene.tsx` takes `{time, values:[bodyId, 0..1][]}` |

## WebSocket protocol (`ws://host:8000/ws`, JSON)
Client -> server, any combination of keys in one message (applied between moves):
| Key | Meaning |
|---|---|
| `{"layout": "solo"\|"swarm"\|"versus"\|"arena"}` | 1 fly · 16 flies on 16 boards · fly vs human on one board · 8 flies on one board |
| `{"policy": "trained"\|"instinct"\|"hardwired"\|"learning"}`, `{"wiring": "real"\|"shuffled"}` | who picks the move; scrambled-wiring control |
| `{"learning": "reset"\|"pretrained"}` | switch to live learning from a blank readout or a copy of the trained readout for the current wiring; saved models are unchanged (automatic rewards: food +1, death -1, closer/farther +-0.1) |
| `{"feedback": value, "fly": i\|null, "move": moveId}` | reward/punishment in [-1, 1], excluding zero, for a displayed decision (learning policy only); omitted/null fly targets all eligible flies; omitted move uses latest saved decision. The last 64 decisions are retained; experiment/model changes invalidate them. Receipts report applied or rejected feedback. |
| `{"lesion": {"fly": i\|null, "types": ["DNa02", "LC10.*"]}}` | silence neuron types (regex, full match on annotation `type`); `null` = every fly; `[]` heals |
| `{"sensor": {"danger_ahead": 0.8}}` | hardware input: drive 0..1 **added** to the game's senses, goes stale after 0.6 s, so resend at >= 5 Hz |
| `{"stimulate": {"food_L": 1}\|null}` | manual override of all senses; game holds still while set |
| `{"human": "up"\|"down"\|"left"\|"right"}`, `{"select": i}`, `{"paused": bool}` | human snake; which fly's brain is shown; pause |

Server -> client, one frame per move (see `LiveFrame` in `src/lib/live.ts`): `arenas[]` (boards, foods, snakes), `flies[]`
(per fly: `channels`, `action`, `probabilities`, `reward`, `feedbackEligible`, `steer` = Hz of DNa02/DNa01/DNp01 L/R, `lesion`), `selected`,
`move` (monotonically increasing decision ID), `values` (selected fly's brain activity by bodyId),
`learning {moves, games, scores[], feedback: {positive, negative, last}}`, `activeNeurons`, `sensor`, `manual`.
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
- With DNa02 silenced the trained readout still steers (it uses other descending neurons); the hardwired policy cannot.
- Known weakness to answer: the game still shows only 24 distinct situations. Target preferences start from a heuristic and are
  optimized on full-game scores; the encoder now computes tail connectivity. This is engineered spatial preprocessing, not evidence
  that a biological fly plans routes. "The readout plays, the brain relabels" remains a fair criticism.
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
| 3 | **Hardware** (Pi Zero 2 W, ultrasonic sensor, RFID shield) | See **Hardware track** at the end of this file. Pi runs a small Python WebSocket client on WiFi: distance + approach speed -> `{"sensor": {"danger_ahead": x}}` at 10 Hz so a judge's hand makes the giant fiber fire and the snake dodge. RFID tags as cards: "sugar"/"bitter" tag -> stimulate `LB3*`/`LB1*` and show MN9 (needs a small server addition), or tags = reward / punish / lesion cards. HC-SR04 echo is 5 V: use a voltage divider into the Pi's 3.3 V GPIO | new `hardware/` folder, tiny `server.py` addition | 2-4 h | low-medium | Pi; no GPU (use `replay_server.py` or a teammate's server) |
| 4 | **Arena + human vs fly + frontend** | Make versus/arena fun: countdown, win condition, scores, fly colours, leaderboard of lesioned flies in swarm, brain panel highlights for stimulated + descending neurons, spike audio (giant fiber clicks), overall layout polish, fallback video | `src/`, small `snake.py` rules | 4-5 h | low | any GPU or replay |
| 5 | *Stretch:* **Real vision** | Paint the board onto the eye: stimulate medulla columns by `assignedOlHex1/2` retinotopy instead of LC10/LC4 directly. First a 1 h feasibility check: which types have hex coordinates; does a left-eye patch give left-biased LC10/LC4/DNa02? Go/no-go after that. Raw photoreceptors will likely fail (graded, non-spiking; motion detection needs timing the LIF lacks) | new `flybrain/vision.py`, `channels.py` | 1 h check, then 4-8 h | high | GPU |
| 6 | *Stretch:* **Learning inside the brain** | Mushroom body: reward -> PAM dopamine, punishment -> PPL1, plasticity only at Kenyon cell -> MBON synapses. 30-45 min check first: do Kenyon cells respond to our stimuli at all, and does MBON activity reach the steering neurons? | new module | check, then 4-6 h | high | GPU |

Suggested default for four people: 1, 2, 3, 4 - and whoever finishes first runs the feasibility check for 5 or 6.
Pitch + slides: owner of track 2 (they hold the evidence), with everyone's numbers.

## Hardware track (owner: Owen) - everything decided or learned so far
**Parts on hand:** Raspberry Pi Zero 2 W, an ultrasonic distance sensor, an RFID evaluation shield. Exact sensor and shield
models have not been checked - read the markings before wiring. No Arduino is confirmed (only an Arduino IDE folder on one laptop).

**Architecture.** The Pi Zero 2 W runs Python and has WiFi, so it is a WebSocket client of the brain server directly -
no Arduino, no serial bridge. Start the server with `--host 0.0.0.0`, put the Pi on the same network, connect to
`ws://<gpu-laptop-ip>:8000/ws`. Hackathon/venue WiFi often blocks device-to-device traffic: a phone hotspot is the fallback.

**Input: hand = looming threat (the main demo).** Send `{"sensor": {"danger_ahead": x}}` with x in 0..1.
- The value is *added* to the game's own senses for every fly, clipped to 1, and expires after 0.6 s: resend at >= 5 Hz (10 Hz is good).
- Any channel name works: `food_L, food_R, danger_L, danger_R, danger_ahead`. Two sensors could drive `danger_L` / `danger_R`.
- Verified against the live server (scripted client, not real hardware): `danger_ahead: 1` -> giant fiber DNp01 ~400 Hz on both sides.
- Looming means *approaching*, so encode closeness plus approach speed, e.g.
  `x = clip(max(0, (60 - cm) / 50) + max(0, -d_cm_per_s) / 100, 0, 1)` - tune on the day.
- `{"stimulate": {...}}` is different: it *replaces* all senses and freezes the game. Use `sensor` for hardware.
- Latency: one brain window is 100 ms and the solo server runs ~6 moves/s, so hand-to-dodge is roughly 0.2-0.4 s.

**Output: what the brain is doing, for servos / LEDs / sound.** Every frame (one per move) has, per fly,
`flies[i].steer` = firing rates in Hz for `DNa02_L/R`, `DNa01_L/R` (steering) and `DNp01_L/R` (giant fiber, escape), plus
`action` (0 left, 1 straight, 2 right), `reward`, and the board in `arenas[]`. `frame.selected` is the fly shown on screen.
A frame is ~20-25 kB of JSON (it also carries brain activity for the viewer); the Pi Zero 2 W parses that fine at 6 Hz.

**Wiring caution.** A classic HC-SR04 runs on 5 V and its ECHO pin outputs 5 V; the Pi's GPIO is 3.3 V only. Put a voltage
divider on ECHO (1 kOhm from ECHO to the GPIO pin, 2 kOhm from that pin to ground). TRIG can be driven straight from 3.3 V.
3.3 V-tolerant variants (HC-SR04P, RCWL-1601) need no divider. RFID shields built for Arduino are usually 5 V logic too:
check before connecting its UART/I2C/SPI lines to the Pi, and level-shift if needed.

**RFID ideas, cheapest first.**
1. Cards that need no server change: "reward" / "punish" cards -> `{"feedback": 1}` / `{"feedback": -1}` (only acts in the
   "Learn live" policy); "lesion" cards -> `{"lesion": {"fly": null, "types": ["DNa02"]}}`, a "heal" card -> `"types": []`;
   mode cards -> `{"policy": "hardwired"}`, `{"layout": "swarm"}`.
2. Sugar / bitter cards (the published result, physically): sugar taste neurons are annotation type `LB3*`, bitter `LB1*`,
   the feeding motor neuron is `MN9`. In the model LB3 drives MN9 and LB1 gives 0 Hz. This needs a small server addition that
   does not exist yet: a message that stimulates arbitrary neuron types (today only the five channels can be stimulated) and
   `MN9` added to the rates reported in `steer`. Then: sugar card -> MN9 fires -> a servo extends a proboscis; bitter card -> nothing.

**Other output ideas discussed (none built):** proboscis servo on MN9; giant-fiber or DNa02 spikes as clicks through a speaker
(the Pi Zero has no headphone jack - use PWM audio, I2S or USB); snake on a NeoPixel grid / brain activity on an LED strip;
a joystick for the human snake in the `versus` layout (`{"human": "up"}`); and the ambitious one, a two-wheel robot whose left /
right motor speeds follow `DNa02_L` / `DNa02_R`, with light or distance sensors feeding `food_*` and `danger_*`.

**Developing without the GPU laptop.** Point at a teammate's running server (several laptops have GPUs). `scripts/replay_server.py`
replays recorded frames over the same socket, which is enough for output devices, but it ignores incoming messages, so sensor input
cannot be tested against it. It has not been run yet.

**Starting point for the Pi (untested sketch).** `pip install websockets gpiozero`
```python
import asyncio, json, time, websockets
from gpiozero import DistanceSensor                     # echo through the voltage divider!
sensor = DistanceSensor(echo=24, trigger=23, max_distance=2)

async def main():
    async with websockets.connect("ws://<gpu-laptop-ip>:8000/ws", max_size=None) as ws:
        async def read():                                # drain frames; use frame["flies"][0]["steer"] for outputs
            async for raw in ws:
                steer = json.loads(raw)["flies"][0]["steer"]
        asyncio.create_task(read())
        last_cm, last_t = sensor.distance * 100, time.monotonic()
        while True:
            cm, now = sensor.distance * 100, time.monotonic()
            speed = (cm - last_cm) / (now - last_t)      # negative = approaching
            x = min(1.0, max(0.0, (60 - cm) / 50) + max(0.0, -speed) / 100)
            await ws.send(json.dumps({"sensor": {"danger_ahead": round(x, 2)}}))
            last_cm, last_t = cm, now
            await asyncio.sleep(0.1)
asyncio.run(main())
```

**Say it honestly in the pitch:** the hand is not a visual stimulus to the fly; the sensor value is injected into the fly's
looming-detector neurons (LPLC2 / LC4), and everything downstream - the giant fiber firing, the dodge - is the connectome's.
