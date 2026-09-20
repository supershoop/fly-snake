# Fly Snake: what to build next

Prepared during the requested brainstorming session on 20 September 2026. This is a proposal portfolio grounded in the current working tree, saved results, a small read-only data/encoder audit, and primary research. Three parallel reviews covered science, experience, and engineering. No training, new brain simulations, model replacement, deployment, or application changes were performed.

**My recommendation: make Fly Snake an interactive circuit experiment, and use a small diagnostic suite to decide the next learning investment.** The project already has the ingredients for an unusually good experience: recognizable behavior, real anatomical identities, simulated activity, reversible interventions, and honest negative results. The biggest opportunity is making the relationship between these things visible and testable.

For a demo, build an explanation of one move and a predict–lesion–restore challenge. For better internally learned behavior, first establish that obstacle/body information can reliably change the fixed steering output. For a stronger scientific claim, compare against controls that preserve ordinary motor activity and coarse wiring properties. These are complementary workstreams, but success in one does not establish success in the others.

The longer notes contain implementation footholds, literature links, costs, experiments, and stop rules:

- [Scientific experiments](brainstorm-science-2026-09-20.md)
- [Experience, interaction, and exhibit concepts](brainstorm-experience-2026-09-20.md)
- [Nine distinctive demo concepts](brainstorm-bold-demos-2026-09-20.md)
- [Engineering, performance, and experimental infrastructure](brainstorm-engineering-2026-09-20.md)
- [Alternative tasks and learning controls](brainstorm-engineering-unconventional-2026-09-20.md)
- [A concrete first-afternoon protocol: five conditions, 1,920 windows](brainstorm-first-afternoon-2026-09-20.md)
- [Numbers derived during this session](brainstorm-evidence-2026-09-20.json)

| If the immediate goal is… | Start with… | First useful result |
|---|---|---|
| A more convincing audience demo | One-vote impact receipt, or the one-move inspector | A visitor can see and explain an actual computation/change |
| Better internal learning | Small matched-board assay plus sensory/motor gain transplantation | Evidence about where to spend the next training budget |
| A stronger scientific claim | Upstream pursuit-pathway lesion, matched controls, relay rescue | A specific causal result inside the simulation |
| A dependable exhibit | Freshness/task health, bounded publication, labeled recording | A session that stays understandable through interruptions |
| A distinctive long-term direction | One circuit in pursuit, threat, and Snake tasks | A reusable experiment workbench with explicit adapters |

## What changed my priorities

The existing direct-board result is a real improvement over its own original baseline: mean food rises from 0.21875 to 2.0625 on the saved 32-game paired test. But food-only input to the same trained model scores 2.5625, and the old 24-pattern/original-brain reference scores 2.59375. All conditions collide in every test game. The comparisons with the simpler inputs have intervals including zero. This supports a useful pilot, not useful full-board planning or reliable improvement over simpler inputs. See [the experiment report](DIRECT_BOARD_LEARNING.md).

Several more specific observations suggest productive experiments:

| Observation | Evidence | What to investigate; what it does not establish |
|---|---|---|
| The original full-board model mostly moves straight | 237 of 250 recorded test actions are straight, or 94.8%; 20/32 games contain no turns | Training may partly restore motor responsiveness. This does not show that its turns ignore the board. |
| The trained model and its food-only ablation have very similar aggregate action frequencies | Full: 41.4% left, 12.9% straight, 45.7% right. Food-only: 41.2%, 13.5%, 45.3% | Compare state-dependent decisions and matched turn-frequency controls. Aggregate frequencies do not establish identical policies. |
| Occupancy receives a large external stimulation budget | The 23×23 field has 385 outside-board cells before adding the body. In the existing near-left-food fixture, 432/529 occupancy drives equal 1 after blur | Test input dose, boundary representation, and rate-matched irrelevant input. This is not proof of downstream saturation. |
| Input strength is highly uneven across planes | In that fixture, summed drive is food 56.500, occupancy 442.801, body order 0.776. Expected external events per 100 ms are about 847.5, 6,642.0, and 11.6 | These are injected-event expectations from the encoder and 150 Hz stimulus rate, **not measured neural spike counts**. Different cell types have different influence. |
| Fine body rank may be hard to transmit in one window | Adjacent ranks differ by 0.75/144 drive, or 0.078125 expected external events per neuron per 100 ms | Test coarse tail/age representations. Population activity and temporal integration could still carry information; the arithmetic is not an impossibility proof. |
| A mirrored food probe has unequal threshold crossing | Original near-left food: 4/16 left turns. Original near-right food: 0/16 right turns; its steering differences stay between −2 and 0, below the magnitude needed to turn | Test mirrored scenes, mapping, gain, and motor margins. Sixteen repetitions of these boards do not establish a universal anatomical asymmetry. |
| Annotated optic-column coordinates exist in usable upstream populations | Both hex fields are populated for 1,762/1,773 traced Mi1 and 1,767/1,777 traced Tm1 neurons; none for the current LC10/LC4/LPLC2/LC12/Tm3 pools | A bounded column-based input probe is possible. Coordinates are not a validated natural-vision model or a guarantee that these inputs reach steering. |

The derived audit uses the current `BoardEncoder` and saved JSON/NPZ artifacts. Placeholder IDs were used only when computing input planes because that arithmetic is independent of neuron identity; no synthetic neuronal response was generated. The annotation counts use exact type names except explicitly marked LC10/T4/T5 families. Detailed values are in the companion evidence JSON.

## Twelve bets worth considering first

Effort estimates below mean work by someone familiar with this repository. They are rough implementation estimates, not promised elapsed times, and experimental compute is additional unless stated.

### 1. Explain one move completely

Show a compact chain: board **before** the move → actual injected senses → relevant descending activity → the actual decoder calculation → chosen action → outcome. Let a visitor pin a decision while the game continues. This makes the whole activity cloud interpretable.

The existing `NeuralReadout.tsx` contains useful display pieces but is not mounted by `App.tsx`. Reuse those pieces. For a linear readout, show exact contributions to the selected-versus-runner-up logit; for instinct, show whether pursuit, veto, or dodge determined the action. Call these decoder explanations. A contribution to a logit does not prove a neuron is causally necessary. The trained policy chooses argmax; only the live learner samples the softmax distribution. Label the trained bars as preferences rather than literal selection frequencies.

**Important implementation detail:** current frames contain the board after `Arena.step()`, while channels and spikes describe the preceding decision. An explanation must explicitly store the pre-move state, not overlay stale senses on the next board. `render_state()` includes references to mutable body/food lists, so the saved snapshot must copy them. Estimated work: a half-day strip, 1–2 days for a correct inspector. Success: unfamiliar viewers can identify what was sensed, what chose the action, and where learning happens.

### 2. Make the lesion lab a prediction game

Ask the visitor to predict whether a lesion will reduce eating, shorten survival, or do neither. Run intact, targeted lesion, and matched random controls from a declared set of paired starting conditions; then restore the circuit. Report food and survival together, because a fly that wanders safely can live longer while eating less.

Begin with an upstream pursuit relay versus matched controls, then the DNa02/DNp01 double dissociation. Removing a neuron's signal from the decoder is intuitive, but disrupting an upstream relay and recovering the response is a stronger mechanistic story. Keep a live episode as an illustration and accumulated trials as evidence. Estimated work: 2–4 days for a reusable runner and interface; a clearly labeled recorded trial is a smaller first version.

Use the frozen `InstinctPolicy` for the pursuit/escape behavioral demonstration: its hand-set rules explicitly read both pathways. `HardwiredPolicy` does not read DNp01, so it cannot substitute for that comparison. The output-cell dissociation partly follows this decoder design; an upstream intervention provides more independent mechanistic evidence.

### 3. Build a sensory-to-steering exam before longer training

Use paired valid boards with the same head, heading, food, body length, and timer, but an obstacle on opposite sides. Add moving-tail entry, mirror scenes, food range, and short route traps as separate categories. Record changes at input, relay, broad descending, and fixed-motor stages. Run from reset and from a few declared continuous histories.

This answers whether the bottleneck is weak input, anatomical routing, a narrow set of trainable pathways, or threshold crossing. A game engine can label immediate safety for evaluation; those labels must never enter the experimental controller. Estimated work: several hours for a small panel, 1–2 days for reusable diagnostics. Do not turn a diagnostic panel repeatedly used for design into the final test set.

Use a few handmade cases for debugging, then sample valid states from different policies, body lengths, and difficulties. Keep mirrors and near-duplicate layouts in the same data split. Evaluate valid action sets rather than demanding one arbitrary turn when several moves are safe. A negative local sensitivity screen means no effect was detected under those conditions; coordinated nonlinear changes could still work.

### 4. Find out what the saved synaptic model actually learned

The direct-board checkpoint, `models/brain-board-direct.npz`, already labels sensory and motor parameter groups. Recombine original and trained gains into four frozen variants: neither changed, sensory only changed, motor only changed, both changed. Then revert food, occupancy, and body-order groups one at a time. This checkpoint is separate from the web app's `brain-synaptic.npz`.

If motor gains alone recover most behavior, that is a valuable, narrower result about responsiveness. If obstacle/body groups selectively improve the paired-board exam, it is evidence that those channels have acquired useful influence. Nonlinear interactions mean the effects need not add. Estimated work: a few hours plus evaluation. This is more informative than interpreting the number of changed connections as the amount learned.

### 5. Redesign the full-board input around an explicit signal budget

Try three separately versioned variants: lower occupancy drive; encode the board boundary rather than filling the entire outside area; replace fine body rank with a tail marker or coarse age bins. Keep food input fixed initially. Compare each with a within-type/hemisphere permutation that preserves the same input values but disrupts geometry.

A fixed permutation is still information-preserving in principle. With the frozen model it tests dependence on the original input-to-neuron alignment; it does not, by itself, identify an abstract geometry representation. Re-randomizing each move also changes temporal statistics. Declare the version and use the paired-board behavior to test useful geometry directly.

Raw body order is mathematically precise but transmitted noisily by the current Poisson stimulation interface. A clear tail marker could be more usable than a tiny rank increment. It remains engineered sensing, and boundary/age transforms must be described as such. Do not add a collision planner or action mask to the direct-board experiment. Estimated work: 0.5–2 days per small variant; continue only if the paired-board exam improves before longer games.

### 6. Add controls that can turn

The no-input trained model is straight on all 214 recorded moves. Beating it establishes the value of sensory stimulation under this setup, but it is an easy behavioral baseline. Add a separate, plainly labeled non-brain controller with matched left/straight/right frequencies, and another preserving short action-transition statistics. Fit these statistics on development rollouts, freeze them, then evaluate fresh boards.

Also compare 2–4 bounded gains on existing incoming DNa01/DNa02 connections with the 60-group training experiment, keeping the same fixed decoder, training/validation seeds, and actual whole-brain window budget. Do not change the motor thresholds in this control. For the 24-pattern demo, surface the direct observation-to-action target table as a baseline; much of that machinery already exists in `training.py`. These comparisons ask whether detailed anatomy/learning adds something beyond broad movement statistics and engineered sensing. Estimated work: roughly a day for a clean evaluation ladder.

Equal episodes and equal windows cannot generally both hold when one controller survives longer. Choose one primary budget, match model-selection opportunities, and report realized episodes, windows, and wall time. Lower-dimensional search being easier is part of the practical comparison, not a flaw to eliminate.

### 7. Introduce a short failure replay

After a death, preserve the last several decisions with the true terminal reason, input, decoder, and model identity. Ask one concrete question: did it pursue food into danger, fail to distinguish the situation, or receive too little steering drive?

Use actual `end_reason`; reward −1 alone cannot distinguish collision and starvation. A legal alternative move is not proof that the alternative would win. Recorded simulation must remain labeled as recorded and disable controls that cannot affect it. The current replay server emits live-shaped frames, so source metadata is an early prerequisite. Estimated work: 1–2 days for the useful core.

### 8. Turn an encoder limitation into a memorable exhibit

Show two boards that look different to a human but produce the same default five-channel input. Ask whether the brain can distinguish them, then reveal the shared 24-pattern code. Let the visitor discover which changes are invisible and which trigger the engineered tail-route warning.

For identical-output demonstrations, hold the starting neural state and external random events equal. Continuous histories can otherwise produce different responses to identical current input. A second round lets the visitor suggest a missing observation and inspect what the direct-board encoder supplies. Estimated work: 0.5–1.5 days. This teaches the project's most important limitation through interaction rather than a buried disclaimer.

### 9. Show exactly what one audience vote changed; then test coaching

The smallest strong addition is a receipt showing the action distribution for the saved neural response immediately before and after that one feedback update. Evaluate both sides at application time. Comparing the original displayed probabilities with current probabilities would wrongly include intervening learning. This makes a real parameter change visible without claiming improved play or simulating imaginary future trajectories. Estimated work: 1–2 days; see [the detailed concept](brainstorm-bold-demos-2026-09-20.md#4-one-vote-two-possible-responses).

Use the saved features directly—they already contain `log1p(counts)`—and attach the result to the request-specific receipt. A long-pinned decision may have expired from the existing 64-decision history. Request IDs currently correlate replies but do not deduplicate repeated updates, so do not silently retry a feedback request. For experiments, declare a per-decision vote budget.

The larger experiment is a meaningful comparison: matched learners with automatic reward only, genuine coaching, and a sham or time-shifted feedback control. Freeze the resulting readouts and test them on new boards without coaches.

These must be separate learners; the current swarm shares one readout, so assigning different treatment labels to its flies would contaminate the comparison. Add a human-readable receipt showing the exact pre-move board that a vote trained. Match **applied** feedback count, signed magnitude, and timing/age distributions, not just submitted taps: stale or ineligible requests are rejected. For a fair classroom competition, normalize feedback budgets rather than rewarding the fastest tapper. Estimated work: 2–4 days for the experiment. A successful crowd demo is not itself evidence that feedback improves generalization.

The existing Trained button loads the saved trained policy; it does **not** freeze the current live learner. A train→freeze→evaluate experiment needs a snapshot of the current weights/bias and an explicit choice of evaluation action rule.

### 10. Give every displayed result an experiment identity

Use a compact visible status and an inspectable record: live/recorded; encoder; fixed/trained/learning decoder; original/modified synapses; wiring; lesions; model hash; phase; seeds; terminal/capped outcomes. Reuse the strong checkpoint validation from the new synaptic code for older readouts and response banks.

Model switches should be recorded as phase changes in research mode. The private operator tool can keep credentials and controls private while the saved evidence records that the underlying controller changed. An apparent learning curve should never silently combine different checkpoints. This is a proposed research-mode improvement to existing deliberate operator behavior. Estimated work: 1–2 days for a shared manifest and phase ledger; no database needed.

### 11. Spend engineering time on smooth sessions and measured CPU speed

The repository already has a compiled CPU kernel with saved equivalence checks. Profile complete solo/swarm ticks, not just matrix multiplication, and test whether optional live solo use is worthwhile. The compiled backend currently supports only one CPU brain, so transitions to swarm need explicit handling.

Keep simulation progress independent of a slow viewer with bounded latest-frame queues. Preserve complete decision/outcome records on the server and reliable terminal, phase, and receipt events; supersede only presentation frames. Bound incoming controls, expire sensor data from receipt time, retain the background task's health, and reject malformed commands without stopping the loop. These are concrete areas identified in code, not failures induced during this session. Estimated work: 0.5 day for profiling, 1–2 days for a small resilience pass. This can improve every future demo without changing scientific behavior.

### 12. Give the circuit a world it is suited to

Keep Snake as the recognizable main exhibit, but add a simple open-field pursuit task and a looming-escape task sharing the same declared sensory/motor interface. A controller can then show target pursuit and threat response before confronting body avoidance and long-horizon traps. Treat transfer into Snake as an explicit experiment.

A two-dimensional dot world is a small first step. Full NeuroMechFly integration is a later project: its documentation provides vision, olfaction, and hierarchical controller interfaces, but connecting a brain to that body still requires engineered sensory and motor adapters. [Official framework documentation](https://flygym.readthedocs.io/latest/index.html). Estimated work: 1–3 days for a minimal 2D task; substantially more for an embodied integration. Do not equate the existing decorative fly mesh with a physics simulation.

## A wider idea portfolio

These are deliberately varied options, not a recommendation to build all of them. “First step” is a way to reduce uncertainty before committing.

| ID | Idea | First useful step | Main value |
|---|---|---|---|
| 13 | **A circuit escape room:** diagnose a hidden upstream lesion from several controlled stimuli | Start with three known lesion classes and a visible reveal | Engagement and causal reasoning |
| 14 | **Predict the next move before the decoder is revealed** | Use real captured decisions and collect predictions | Teaches sensing and policy differences |
| 15 | **One input, many neural histories** | Present an identical final input after different cue sequences | Establishes whether task-relevant memory exists |
| 16 | **A controlled memory challenge** | Cue, blank delay, then choice; test delay curves against reset | A separate temporal computation benchmark |
| 17 | **Learning reversal** | After acquisition, reverse a declared reward contingency and test frozen checkpoints | Shows adaptation rather than a rising score alone |
| 18 | **Selective mushroom-body re-probe** | Stimulate individual MBON types/doses and record all DNs | Tests whether broad opponent activation hid a pathway |
| 19 | **Two-odor conditioning with an MBON endpoint** | Separate odor-like input patterns, unpaired controls, fixed output measurement | Internal learning without promising Snake steering |
| 20 | **Annotated-column visual input** | Map small patches to Mi1/Tm1 hex annotations and check lateralized propagation | A bounded alternative to arbitrary position assignment |
| 21 | **Directional escape outputs** | Probe existing DNp02/DNp11 pathways for the forward/backward escape dimensions supported by the source study | Better task/output matching; no assumption that these provide left/right Snake steering |
| 22 | **Heading-versus-goal steering assay** | Probe PFL3→DNa02, then supported population patterns | Bounded central-complex route toward navigation |
| 23 | **Pulse-shaped looming** | Independently vary angular-size-like and expansion-speed-like injected features | Tests temporal threat integration |
| 24 | **Minimal sufficient circuit** | Silence outside a candidate circuit on frozen sensory sequences | Measures which anatomy this task actually needs |
| 25 | **Pathway edge lesions and relay rescue** | Disrupt a sensory→relay edge, replay an appropriate relay signal downstream of the lesion but upstream of the decoder | Stronger mechanism than bright-neuron correlation |
| 26 | **Anatomically constrained wiring nulls** | Preserve hemisphere/type/transmitter properties at separate steps | Locates the structural property doing useful work |
| 27 | **Gain placement control** | Reassign gains across compatible groups while matching group size/base strength/activity where feasible; first use the cleaner frozen group-transplant test | Exploratory test of learned placement; within-group permutation is a no-op and unmatched reassignment changes total drive |
| 28 | **Yoked sensory playback** | Feed one game a recorded sensory sequence from a different game, clearly labeled as a control | Tests alignment with the current world, beyond realistic activity |
| 29 | **Causal sensitivity of trainable groups** | Small paired gain perturbations on representative neural states | Finds silent, redundant, or saturated search dimensions |
| 30 | **A tiny motor-gain learning baseline** | Train bilateral output excitability with the same budget | Measures the value of richer plastic pathways |
| 31 | **A task curriculum with transfer** | Pursuit → single obstacle → conflicting cues → body avoidance | Separates skill acquisition from full-game reward noise |
| 32 | **Robustness to input noise and missing channels** | Freeze model, vary dropout/dose on new seeds | Practical sensor tolerance and uncertainty |
| 33 | **Recovery after an imposed turn error** | Apply one declared perturbation, measure recovery to food/safety | Closed-loop correction beyond initial-state success |
| 34 | **Independent training replicates** | Several modest fresh runs before one massive run | Reliability of the learning method |
| 35 | **Crossed game and neural-noise seeds** | Replay several boards with several independent neural streams | Separates environmental difficulty from neural variability |
| 36 | **Neural-bank calibration on encountered histories** | Compare saved-bank predictions with real continuous traces | Quantifies approximation error |
| 37 | **A physiology regression battery** | Preserve sugar/bitter, pursuit, and threat probes while changing parameters | Prevents task optimization from erasing supported behavior |
| 38 | **Parameter sensitivity of the headline claim** | Repeat short causal probes at plausible gain/dose settings | Shows whether an effect is brittle |
| 39 | **Evaluation that resumes after interruption** | Permanent training lock plus immutable per-seed completion ledger | Makes long frozen tests practical |
| 40 | **Reusable exact state snapshots** | Include board, neural state, delayed signals, RNGs, and learner state | Enables trustworthy counterfactuals and debugging |
| 41 | **Deterministic external-event tapes for diagnostics** | Key stimuli by body ID and simulation time | More precise input/kernel comparisons |
| 42 | **A preflight command for exhibitions** | Check data/model identity, backend, frame health, phone route, and replay availability | Reduces setup surprises |
| 43 | **A portable recorded experiment bundle** | Store validated frames, manifest, and read-only controls | Reliable teaching/demo without a running brain |
| 44 | **Shareable result cards with evidence** | Export comparison, uncertainty, seed list, and provenance | Communicates a real finding beyond a screenshot |
| 45 | **A viewer-authored challenge board** | Validate body geometry, declare neural initial-state policy, export challenge | Fun robustness testing with reproducible failures |
| 46 | **Behavior profiles instead of one score** | Food, collisions, starvation, turn bias, loops, and response latency | Distinguishes pursuing, avoiding, and circling |
| 47 | **Accessible sensory and output cues** | Shapes/text plus optional bilateral sonification; reduced motion | Makes the experiment readable beyond color/3D |
| 48 | **A calibrated hand-to-neuron station** | Show distance, approach speed, received age, injected drive, and DN response | Makes hardware input measurable and legible |
| 49 | **A sugar/bitter physical output** | MN9 simulation drives a labeled proboscis servo after a small assay | A compact sensorimotor demonstration grounded in existing work |
| 50 | **A classroom experiment pack** | Three hypothesis cards, expected controls, blank result sheets, honest interpretations | A reusable educational product |
| 51 | **A compact artifact protocol** | Common manifests for model, input mapping, activity, action, intervention, and outcome | Turns a one-off demo into a reusable workbench |
| 52 | **A mode vocabulary people understand** | Show “senses / brain / controller / what learns” instead of relying on “Normal” | Prevents interface labels from overstating biology |

Ideas 18–23 are hypotheses requiring targeted feasibility checks. The science and engineering notes link the relevant biological work and explain why results from other connectomes or models do not automatically transfer here.

## How I would spend the next two days

Run two small workstreams if two people are available; otherwise do them sequentially.

**Experiment workstream:** assemble the paired-board exam; measure current original/trained behavior; perform frozen sensory/motor gain transplantation; compare occupancy dose and a rate-matched permutation. End with a decision about where information stops affecting action. Keep the current encoder/decoder reference intact and give each variant a new identity.

**Experience workstream:** show a single correct pre/move/post decision with source and learning labels. Use an accurately labeled existing recording and a prediction/reveal card for the lesion comparison. Add terminal reason if it fits; defer a new live paired-trial runner and portable replay UI. End with a five-minute demonstration a new visitor can explain back. The complete versions of all three features will take longer than two days.

The deliverables are a diagnostic result and an understandable interaction. More trained generations are a decision informed by this result and the existing learning curve, not an automatic milestone. The [first-afternoon protocol](brainstorm-first-afternoon-2026-09-20.md) makes the initial assay concrete: five frozen gain/input conditions, six validated geometric situations, 16 noise seeds, and four continuous windows per trial. It uses 1,920 neural windows, performs no training, and preserves all existing model files and test locks.

## Why the preferred diagnosis could be wrong

The pilot already improved over its own baseline, and 24 generations with two training games per candidate do not exhaust a noisy 60-parameter search. More independent small runs could be cheaper and more informative than a sophisticated diagnostic platform. If development/validation curves are still improving, useful obstacle responses repeatedly appear, and gains replicate on fresh seeds, modest additional training is reasonable alongside the small assay.

Input-event imbalance is not functional influence: different cell types can amplify or inhibit differently. A fixed-model ablation also differs from training a food-only model from scratch. Local gain perturbations may miss coordinated threshold crossings, and reset-state probes may miss useful history. These are reasons to keep the first diagnostic small, include on-policy states, and check promising effects in closed-loop play—not reasons to assume either the encoder or optimizer has already been disproved.

If the obstacle panel stays negative, the strongest lower-risk scientific branch is a specific native pursuit-pathway study with matched lesions, relay rescue, and parameter robustness in a simple pursuit assay. If internal associative learning is the priority, the two-odor KC→MBON experiment is a separate bounded research bet.

## A one-week plan with decision points

| When | Deliverable | Decision |
|---|---|---|
| Days 1–2 | Decision inspector, paired-board exam, input-budget audit, frozen group transplantation | Does relevant geometry reach and control the fixed motor signal? |
| Day 3 | One targeted encoder/pathway variant, equal-budget food-only and motor-gain controls | Is improvement specific to useful information? |
| Day 4 | Matched lesion/restore trial, initial phase/provenance records | Can another person reproduce and interpret the demonstration? |
| Day 5 | Small independent training replicates if the mechanism passed; otherwise a second bounded feasibility assay | Is the effect repeatable, and which claim can be supported? |
| Days 6–7 | Frozen evaluation, failure gallery, load/replay rehearsal, evidence card | What can be shown confidently, and what remains exploratory? |

This is a prioritization sketch, not a commitment that all listed engineering work fits one developer-week. Choose the demo branch or the research branch if capacity is limited.

## Things I would defer

- **Thousands more generations on the current direct-board setup.** The next uncertainty is functional access to steering, not simply whether the optimizer has run long enough.
- **An immediate raw-photoreceptor overhaul.** Annotated upstream column inputs and short propagation probes are a cheaper feasibility step. Dynamic vision and graded cellular responses are substantive modeling work.
- **A giant new decoder presented as a brain improvement.** A better decoder is useful engineering, but it answers a different question from learning through existing internal synapses with the fixed decoder.
- **A full physics-body integration before a simple 2D world works.** Prove the sensory/motor contract and neural behavior first.
- **A hidden model switch as evidence of learning.** Separate an operator's practical fallback from a recorded scientific comparison through phases and provenance.
- **Dopamine animation presented as synaptic learning.** The current live learner changes an external readout; dopamine-cell stimulation alone does not implement a plasticity rule.
- **Higher confidence-looking percentages without calibration.** The trained softmax supplies normalized preferences before argmax selection; the live learner samples from its distribution; hardwired output is one-hot. None is a validated probability of a good decision.
- **Heavy infrastructure.** Plain manifests, a bounded in-process event loop, a small trial runner, and a few independent scripts are enough for the next stage.

## The story this supports

“This is a simulated nervous system built from a real fly wiring diagram. We give selected sensory neurons an engineered view of a game. You can inspect the resulting signals, change a circuit, and test what changes in behavior. Some modes learn an external controller; a separate experiment changes bounded strengths of existing internal connections. The experiments tell us both what this model can do and where our interface or assumptions are limiting it.”

That premise makes room for entertaining gameplay, genuine causal questions, and failed experiments without needing to imply a living fly understands Snake. It is consistent with the research use of connectome models to generate testable sensorimotor hypotheses, rather than treating anatomy alone as a complete functional explanation. [Shiu et al., primary study](https://www.nature.com/articles/s41586-024-07763-9), [Pospisil et al., effectome study](https://www.nature.com/articles/s41586-024-07982-0).
