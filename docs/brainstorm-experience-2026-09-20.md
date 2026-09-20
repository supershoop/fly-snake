# Fly Snake: experience and demonstration ideas

Brainstorm prepared 20 September 2026 from the current working tree. This is a design proposal, not a report of newly run experiments. Effort estimates assume one developer familiar with this repository and exclude new scientific training. No application code, models, services, or existing files were changed for this report.

## Recommendation

Make the project a place where visitors can **predict, intervene, and inspect a consequence**. The most distinctive experience is not a snake getting a larger score; it is a visitor making a precise prediction about a small circuit, changing that circuit, and seeing both the expected effect and its limits.

The first useful sequence is a visible explanation of one decision, a matched lesion comparison, and an inspectable record of the result. These improvements reuse the existing simulator and successful causal demonstrations. They do not depend on a new training breakthrough.

## Three different goals

| Goal | What counts as progress | Best first investments | What does not establish success |
|---|---|---|---|
| A compelling demonstration | A visitor can predict an intervention, understand the signal chain, and accurately describe the limits. | Decision microscope, short guided challenge, physical sensory instrument, replay. | A high score without understanding, or a lucky dramatic episode. |
| A stronger scientific result | A precisely stated claim survives matched controls, declared metrics, independent evaluation, and reproducible artifacts. | Paired intervention runner, complete trial records, random-cell/shuffled controls, terminal outcomes. | A beautiful animation, a within-run training curve, or silently changing the decoder during a comparison. |
| Better learned behavior | A declared trainable component improves performance on held-out games and remains useful across independent training runs or conditions. | Train/freeze/evaluate phases, checkpoint identity, automatic-only versus coached controls, explicit learning target. | More parameter changes, more training generations, or a best-seen score on familiar seeds. |

These goals can support one another, but they are not interchangeable. The input puzzle is an excellent demonstration even if it exposes a weakness. A null lesion result can be useful science. A stronger learned decoder may make circuit lesions less visually dramatic.

## What is already built

Do not spend another sprint proposing these as new features:

- Solo, 16-board swarm, human versus fly, and eight-fly arena layouts.
- Real/scrambled wiring and trained, instinct, learning, and hardwired policies, though not every policy has its own visible button.
- Per-fly lesions, group presets, arbitrary type search, multiselect, healing, and orange cell markers.
- Positive/negative audience feedback from phones, QR links, narrow feedback gateway, and per-request receipts.
- A saved internal-synapse experiment with a fixed decoder and a visible toggle. The separate full-board pilot is not integrated into the web app.
- Recorded-frame output and a replay server; this is a development fallback, not yet a complete audience-facing replay experience.
- An animated keyboard-playing fly. This is a display driven by game events, not a simulated physical fly body.
- A `NeuralReadout.tsx` component containing named descending-neuron meters, action probabilities, and sensory input controls. **It is currently not mounted by `App.tsx`.** Its explanatory pieces can be reused without restoring the removed manual-stimulation controls.

## Five strongest proposals

### 1. A microscope for one move

**Experience.** Below the board, show one readable chain: “food left; threat left” → LC10/LC4 input → DNa02/DNp01 rates → the decoder's actual operation → “straight.” A visitor can pin the last completed decision and read it while the live simulation continues. The live board and pinned decision must be visually distinct.

**Why this matters.** The current three-panel display shows a board, a large point cloud, and an animated fly, but the causal middle is mostly implicit. Making a single decision inspectable turns impressive activity into an understandable mechanism.

**Implementation foothold.** Extract display-only parts of `src/components/NeuralReadout.tsx` into a new decision inspector; use `flies[].channels`, `steer`, `action`, `probabilities`, and `move`. Add a small explanation payload generated beside the actual policy computation in `flybrain/readout.py`, rather than maintaining a second imitation of the policy in JavaScript.

- Instinct mode: show pursuit drive, threshold, whether veto fired, and whether dodge fired. Call these hand-set rules.
- Hardwired mode: show the steering difference and threshold. One-hot outputs are decisions, not calibrated confidence.
- Trained mode: optionally show the largest exact contributions to the selected-versus-runner-up logit difference, including bias and the actual `log1p(spike_count)` transform. Label this “readout contribution,” not the neuron's intention or a causal explanation of the whole brain. This policy chooses argmax; its softmax scores are preferences, not actual probabilities that it will select each move.
- Learning mode: distinguish the most probable move from the move actually sampled. Here the softmax is the distribution sampled by the learner, not a verified probability of surviving.

**Critical detail.** `Experiment.tick()` computes input before `arena.step()` but emits the board after the move. The inspector needs an explicitly associated pre-move board, action, and outcome. Do not paint the last frame's channels onto the returned board as though they describe that geometry. Frame history alone also needs reset/respawn boundaries.

`Arena.render_state()` exposes mutable food/body lists. A pre-move dictionary must copy those nested lists; merely retaining its returned dictionary before stepping will not preserve the earlier state.

**Effort.** Half a day for a compact read-only strip; 1–2 days for a correct pinned pre/move/post inspector and policy explanations.

**First experiment.** Show five unfamiliar visitors three genuine captured decisions. Ask them to identify the sensory input, the chosen action, and whether the learning is in the readout or synapses. A useful first target is four of five visitors answering all three without coaching.

**Stop rule.** If the explanation requires an entire screen of firing rates, simplify it. Never invent reasons that were not computed, imply that the fly “saw” the rendered board, or use a selected neuron's correlation as proof that it caused the move.

### 2. A matched intervention challenge, including recovery

**Experience.** “Which changes first: eating or avoiding obstacles?” Visitors make a prediction, then see intact and altered flies on matched starting boards. Run pursuit-relay silencing, giant-fiber silencing, and a size-matched random lesion as separate trials. Finish by restoring the cells. Report food and moves lived together.

**Why this matters.** The project already has striking lesion results, but the present UI asks viewers to mentally compare independently seeded boards with different accumulated histories. An experiment-shaped interaction makes the evidence easier to assess. Restoration makes the reversible intervention concrete.

**Implementation foothold.** Build a small experiment runner around `Arena`, the existing per-column `Brain.set_lesion` mask, and the loop pattern in `scripts/lesion_scores.py`. Add a trial component rather than enlarging the general control panel. Record trial ID, condition, board seed, neural seed/state policy, decoder, encoder, episode limit, completed outcomes, and failures. Show the current trial and accumulated paired differences.

**Design discipline.** Choose the seed list, episode count, and metrics before playing. Match starting boards and initial-state policy; do not call independent noisy brains “identical clones.” Matching seeds does not imply that all future board states remain identical after different actions. An actual fork at a decision requires game state, simulator state, random state, and learner state to be copied. Recovery is a new condition; neural carry-over needs either a declared reset or a declared observation interval.

**The most defensible first comparison.** Start with the fixed, nothing-trained decoder and AOTU relay versus random-cell lesions. DNa02 is wonderfully understandable, but removing the decoder's own input cells is less surprising than changing an upstream pathway. The current Normal mode uses `InstinctPolicy`; state exactly which rule each trial uses.

**Separate synaptic comparison.** An “original versus trained connections” experiment must hold the fixed hardwired decoder constant. The existing toggle loads hardwired mode when enabled and restores the trained readout when disabled; that toggle alone is not a fair learning comparison. It also changes global weights, so do not assume two different synaptic conditions fit into the existing single-weight batched `Brain` without additional machinery.

**Effort.** 2–4 days for a reproducible small trial runner with a useful UI; a clearly labeled replay of a pre-run trial is a smaller first deliverable.

**First experiment.** Use a preselected modest set of paired games with no audience feedback. Confirm that condition labels, seeds, outcomes, and decoder identity survive export and replay. Then test whether visitors understand that “lives longer” can coexist with “eats less.”

**Stop rule.** Do not keep rerunning until an attractive difference appears. If the effect is inconsistent, present that result or remove the theatrical claim. A live episode is an illustration, not an estimate of a condition's mean performance.

### 3. Instant replay and an honest failure autopsy

**Experience.** On a crash or starvation event, offer “Inspect what happened.” Show the previous few moves, the exact input, output, and terminal reason. Let the viewer scrub a local replay while the main experiment keeps running. End with one falsifiable question: “Did food pursuit dominate?” or “Were two different boards encoded the same way?”

**Why this matters.** Failure is informative and often more compelling than another long successful snake. A visitor currently sees a death animation and respawn, which loses the chance to understand it. Replay also provides a dependable demo artifact when the GPU or network is unavailable.

**Implementation foothold.** Add a bounded client ring buffer of decision summaries and a backend pre-move snapshot for events. Reuse `FLY_RECORD`, `scripts/replay_server.py`, and the source/provenance ideas already present in `src/lib/replay.ts`. Add `endReason` to the snake/frame contract: `Body.end_reason` already exists but `render_state()` omits it. The phone controller currently describes every dead snake as a collision; it should use the actual reason too.

**What to show.** Food collected; moves; collision/starvation/filled/capped status; sensory channel values; externally added sensor values; relevant readout evidence; and the exact version of the controller. An optional game-rule overlay can mark which alternative moves were immediately legal, explicitly labeled as a game-state check rather than a replay of alternative neural decisions.

**Replay truthfulness.** A recording must say “Recorded simulation” throughout, disable ineffective controls, and carry its original checkpoint/encoder identity. The present replay server emits ordinary live-shaped frames, so the current header would call it “Live simulation.” Add source metadata before offering this to a public audience. A frozen local view should say “Inspecting move 418; live experiment continues.”

**Effort.** 1–2 days for short local replay and terminal reasons; 2–3 more for portable clips with complete provenance.

**First experiment.** Capture one collision, one starvation, a mode switch, a restart, and a disconnect. Check that a viewer never confuses a pinned board with current play, and that no stale feedback is offered for an expired decision.

**Stop rule.** Do not infer starvation or collision from `reward == -1`. Do not claim that an unchosen legal move would have won the game. Keep speculative explanations visually separate from recorded facts.

### 4. “What can this fly actually see?” — the input puzzle

**Experience.** Present two visibly different boards and ask whether this encoder gives the brain different inputs. Reveal the five input channels and one of the 24 default patterns. Invite the visitor to edit food or body placement until they find a difference that is invisible to the encoder. A second view highlights exactly what the engineered threat calculation contributes.

**Why this matters.** The project's most important limitation becomes an interactive discovery rather than a caveat buried below the fold. It gives visitors the conceptual tools to distinguish anatomical data, engineered sensing, neural computation, and learned decoding.

**Implementation foothold.** Build a small board editor and a backend or shared evaluation endpoint that calls the canonical `Arena.state()/encode()` and `navigation.dangers()`. Begin with a small set of verified board pairs exported by a Python script to avoid porting the threat logic incorrectly into JavaScript. The full-board pilot can later appear as a separate, accurately labeled comparison with its own encoder and measured results.

**Critical detail.** Equal current inputs do not imply equal current outputs in a continuously running brain with different histories or noise. Say “same input this move.” To demonstrate truly identical one-step computations, start from the same copied neural state and random state. Do not let the board editor silently become a planner that chooses the fly's action.

**Effort.** 1 day for six verified paired examples; 2–4 days for an editable version with validated board geometry.

**First experiment.** Use examples with the same immediate obstacle pattern but different distant geometry. Verify every pair with the actual encoder. Ask a visitor to explain one thing the brain receives and one thing it does not receive.

**Stop rule.** Reject invalid boards and avoid cherry-picking a pair whose “correct action” is asserted without a defined horizon. The first version can demonstrate information loss without making an optimal-action claim at all.

### 5. Audience learning with named phases and inspectable credit

**Experience.** Replace unrestricted button mashing as the main story with a short sequence: observe a frozen initial model; train it with audience feedback; freeze that learned model; evaluate it on new games. During training, visitors can pin one displayed decision before rewarding it. Show which decision received credit and an aggregate trace of audience feedback.

**Why this matters.** QR feedback and delayed-decision receipts already exist. The missing feature is an understandable experiment: what changed, when, and whether the change helped beyond the moves everyone just watched. A private prediction-only interaction can also teach visitors without changing the learner.

**Implementation foothold.** Extend `HumanFeedback`'s saved decisions with lightweight display snapshots. Use the existing request ID/receipt flow in `public/feedback/controller.js`. Add explicit experiment epochs and a snapshot/export of the current readout, preserving the saved models. Render phase boundaries and cumulative human/automatic feedback separately. Keep any new phone operation as a narrow audience action; do not expose host commands through the feedback gateway.

**Credit and evaluation discipline.** All 16 flies currently share a single learner. Rewarding half the flies and calling the others “untreated controls” is invalid because the weight update affects all of them. A proper audience-versus-automatic-only comparison needs independent readouts/cohorts or separate matched runs. During a frozen evaluation, no audience or automatic learning updates occur, and unseen seeds are chosen before inspection. Fresh score distributions are more useful than one best game.

The existing Trained button selects the cached saved policy; it does not freeze the current live learner. A real freeze step must copy the current learner's weights and bias, preserve its identity, and declare whether evaluation uses argmax or sampling. Otherwise both the parameters and action-selection rule can change under an apparent “freeze.”

**Two useful first versions.** (A) Pin and reward a recent decision with the existing 64-decision history and expiry semantics. (B) A clearly labeled “prediction only” quiz in which visitors forecast the next action; it does not train the readout. These are independently useful before implementing a full comparison.

**Effort.** 1–2 days for pinned feedback and phase markers; 3–5 days for trustworthy train/freeze/evaluate and independent comparison cohorts.

**First experiment.** Rehearse with a few phones and deliberately delay feedback by several moves. Confirm the receipt refers to the pinned move, model changes invalidate it, and participants can distinguish “my vote was applied” from “my vote improved performance.”

**Stop rule.** If audience learning collapses or the evaluation shows no improvement, display that result. Do not rescue the curve with an unmarked checkpoint switch. Existing private operator controls can remain private, but scientific comparisons need an exportable marker for controller replacement and invalidated evaluation phases without disclosing the secret.

## Eighteen more concrete ideas

These are proposals, not additional claims of demonstrated performance.

| # | Idea and visitor payoff | Smallest useful implementation | Effort / limiting condition |
|---|---|---|---|
| 6 | **A physical escape instrument.** Move a hand; see raw distance, injected threat drive, giant-fiber response, and selected action together. | A tiny sensor client using the already supported `sensor` message; a visible input/response strip using `frame.sensor` and `steer`. Start with the actual available sensor only after identifying its model. | 1–2 days plus hardware setup. Existing safety guidance requires checking voltage compatibility. Strong DNp01 firing is not a promise of a successful dodge; input clipping and symmetric threats can hide the effect. |
| 7 | **A 90-second guided exhibit.** “See the input; remove a pathway; restore it; try a control.” | A presenter checklist or UI tour that selects documented settings and shows the next explanatory prompt; every change visibly marks the current condition. | Half a day for a script, 1–2 days for UI. Never automate a desired result or choose only flattering episodes. |
| 8 | **Two-dimensional performance.** Show food collected against moves lived; distinguish useful pursuit from long wandering. | A scatter plot of completed episodes with lesion condition, cap, and terminal reason. | 1 day after episode events exist. Avoid a single leaderboard that rewards starvation or unfinished games as victories. |
| 9 | **A compact model passport.** One line identifies encoder, wiring, decoder, learning target, source, checkpoint, and phase. | A compact persistent label with expandable provenance and model hash. | 1 day. Keep it readable on the main page; detailed version information belongs in the expansion/export. |
| 10 | **A shareable experiment postcard.** “We silenced these 12 cells; here is what we observed.” | Export a small HTML/PNG result plus machine-readable JSON with conditions, seeds, N, cap, uncertainty, source labels, and links to reproduction commands. | 1–2 days after trial metadata. Include the full predeclared trial, not only a winning episode; preserve attribution. |
| 11 | **A quiet, accessible bench.** Equal functionality without the animated fly, rapid changes, color distinctions, or precise mouse clicks. | Static organism illustration; explicit text action/condition; keyboard board selection; focusable cell/type controls; slower local inspection. | 1–2 days. Existing reduced-motion CSS suppresses some transitions but does not stop the Three.js fly animation; do not assume CSS alone solves it. |
| 12 | **Projection mode.** A judge across the room can follow one question and one result. | Hide secondary controls behind an operator drawer; enlarge one board, a compact causal strip, phase label, and two metrics. | Half a day to 1 day. Preserve a visible simulation label and reachable attribution. |
| 13 | **Circuit spotlight.** Select food pursuit or escape and inspect named cell groups, rates, and measured connections. | A schematic beside the soma atlas, generated from actual edge/type data; highlight group members by body ID. | 2–3 days. Schematic edges are not measured neurite paths, and anatomical paths alone do not prove active signal flow. |
| 14 | **An evidence shelf.** Visitors can compare a live trace with the larger saved experiment behind a claim. | Cards for continuous-brain tests, response-bank estimates, and internal-synapse pilots, each with N and caveats. | 1 day using existing JSON artifacts. Never compare unlike rows as though they were one benchmark; show the inconclusive and negative results. |
| 15 | **A fair human challenge.** Play a defined round, then inspect how your observation differs from the fly's input. | Countdown, fixed round length or episode count, visible outcome, and optional human view showing only the encoded signals. | 2–3 days. Full-board human vision versus five-channel fly input is a deliberately unequal task; label it. Engineering a restricted human interface is a separate experiment. |
| 16 | **Learning as a trace, not a slogan.** Display completed-game score versus experience with phase changes and feedback dose. | A rolling plot beside the current last-20 mean, with sample counts, terminal reasons, and explicit resets. | 1 day. Do not pool unrelated modes or checkpoint replacements into one apparent learning curve. |
| 17 | **Show where learning happened.** A before/after readout weight panel contrasts external readout updates with bounded internal-synapse changes. | Snapshot model summaries and visualize deltas with a fixed scale and target label. | 1–2 days. Parameter movement is not proof of behavioral improvement; internal synaptic changes remain engineered simulated plasticity. |
| 18 | **Rate sonification.** A quiet left/right sound makes pursuit and escape perceivable without watching every meter. | Optional gain-controlled stereo tones driven by DNa02/DNp01 rates with a mute control and textual equivalent. | Half a day. The current output has 100 ms counts/rates, not precise spike times; label sound as a rate mapping, not recorded neural spikes. |
| 19 | **An experiment notebook.** Capture a visitor's prediction before a lesion and their observation after it. | A local session card with hypothesis, condition, outcome, and a suggested next control selected from a short authored list. | 1 day. No need for generated scientific explanations; distinguish observation from interpretation. |
| 20 | **A visible sensor fault demo.** Unplug or stop the hand sensor and watch its drive expire while the game keeps going. | Age indicator, input status, and a deliberate test of the existing 0.6 s expiry behavior. | Half a day after the hardware client. Demonstrates robustness and the boundary between external input and brain dynamics. |
| 21 | **A museum of null results.** “We tried more board detail; did it help?” | A small interactive comparison of the direct-board pilot's five actual conditions and an explanation of what remains unknown. | 1 day from existing artifacts. This pilot does not establish an advantage over the old 24-pattern input or food-only input. |
| 22 | **A separate smell-learning station.** Pair an odor with an engineered reward rule and inspect MBON response changes. | A distinct, clearly scoped experiment after the mushroom-body feasibility work, with its own outputs and controls. | Research stretch, several days or more. Current evidence says the proposed mushroom-body output does not drive Snake steering in this model; do not pitch it as a Snake-learning upgrade. |
| 23 | **A trustworthy live indicator.** The display distinguishes an open connection from a brain that is actually producing new frames. | Track last new move and last frame age in `useLiveBrain`; show “simulation waiting” or “stream stalled” rather than letting an old frame remain marked live indefinitely. | Half a day. Treat declared pause, death animation hold, initialization, and stale transport differently; the phone controller already has a freshness check that the host can learn from. |

## Suggested order

**If there is only one day:** mount a compact, display-only neural decision strip; correct the animation wording; expose terminal reasons; provide a carefully labeled recorded example. This immediately improves comprehension without requiring more training.

**If there is one week:** add the paired intervention runner and pre/move/post replay, then a portable experiment postcard. Use a small guided story that includes a control and restoration.

**After that:** audience phase-based learning, the input puzzle, physical sensor instrumentation, and optional sonification. Add visual polish only after visitors can correctly explain what computes the senses, what runs in the connectome, and what chooses the action.

## Small copy and truthfulness improvements worth bundling

- `App.tsx` currently says “live fruit fly reaction” below an event-driven animation. “Animated reaction to the simulated move” is more accurate.
- “Customize a fly's wiring” currently often means silencing cells, which does not rewire connectivity. “Change which neurons can fire” describes the lesion interaction more precisely.
- “Normal” is a hand-set instinct decoder; explain that without suggesting it is the fly's biological motor policy.
- The lesion note says the trained readout “works around” missing cells. A frozen readout need not be adapting: it already uses many other descending neurons. Describe that distinction.
- Do not infer “brain learning improved Snake” from a visible switch whose decoder also changed. The existing synaptic pilot is inconclusive on held-out improvement.
- Do not promote the anatomy point cloud into an image of synaptic wiring. It depicts measured somata, with simulated activity values overlaid.
- Preserve visible source labels when screenshots, clips, and exported figures leave the app. A footnote in the original page will not travel with a cropped image.

## Source map

This report was grounded in `src/App.tsx`, `src/components/{Environment,ExperimentControls,NeuralReadout,LesionLab,LiveTraining,AudienceTraining,BrainLearning,BrainScene,FlyScene}.tsx`, `src/lib/{live,replay}.ts`, `public/feedback/{index.html,controller.js}`, `flybrain/{server,readout,snake,navigation,feedback,audience}.py`, `scripts/{lesion_scores,replay_server,evaluate_survival}.py`, and `docs/{SYNAPTIC_LEARNING,DIRECT_BOARD_LEARNING,MODEL-INTEGRATION}.md` in the current working tree. No web sources were needed for these repository-specific design proposals. Existing edits from other work were left untouched.
