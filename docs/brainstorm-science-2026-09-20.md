# Scientific improvements for Fly Snake

Brainstorm, 20 September 2026. These are proposed experiments, not new results. The recommendations come from reading the current simulator, encoders, trainers, evaluation scripts, saved-model documentation, and the primary sources linked below. No training or simulator experiment was run for this note. Effort estimates are engineering estimates for someone familiar with the repository, not measured runtimes.

The highest-value next step is to identify which parts of the signal chain may be limiting behavior. Occupancy-driven saturation remains a hypothesis, and no general inability to transmit information to the motor output has been established. Diagnostics can guide the next substantial training investment while modest independent replications remain worthwhile. The strongest scientific story may be a small number of precise causal findings rather than a higher Snake score.

## What the existing evidence does and does not establish

- The direct-board pilot establishes improvement for one internally trained checkpoint over its original full-board baseline on new game seeds. It does not establish reliable improvement across independently trained checkpoints, useful body planning, or an advantage over food-only input. Every final-test game collided.
- The direct-board encoder has 8,082 trainable existing edges, but their distribution is unequal: 5,505 food-input edges, 646 occupancy-input edges, 194 body-order-input edges, and 1,737 final steering-input edges. Counting neurons or edges does not establish equal functional influence.
- The 23 by 23 occupancy field contains 385 outside-board pixels before adding the snake, so at least 72.8% of its pixels are fully driven before smoothing. The companion root audit in `docs/brainstorm-evidence-2026-09-20.json` examines the input budget more closely. These are external stimulation levels/events, not measured neuronal spike rates.
- A better action histogram can show that a controller has started turning; it cannot by itself establish that the turns depend on useful board geometry.
- The current no-input board control starts each episode from rest with no spontaneous current. With zero sensory drive, no spikes initiate, so the fixed decoder chooses straight. This is an informative silence baseline but a weak test of whether board-specific information is used; unrelated but rate-matched input is stronger.
- Changing synapses entering DNa01/DNa02 is internal parameter learning. Functionally, those changes can still act like an anatomically constrained motor readout. That is a testable distinction, not an objection to the experiment.

## Five experiments to prioritize

### 1. Counterfactual board pairs: can body geometry change the decision appropriately?

**Question.** With heading, relative food position, length and timer held fixed, can a changed body configuration or wall distance alter the fixed steering signal in the useful direction? Body-shape pairs can keep the absolute head fixed; boundary-distance pairs must translate the head/food configuration together within the fixed board.

**Why this matters.** The existing probe established changes in general descending-neuron activity while body shape barely affected the four steering neurons. A full-game score mixes sensory representation, motor activity, visited states, random food placements and survival. A small diagnostic panel can locate the failure before more training.

**Minimum test.** Construct 24–48 valid, reachable paired situations, split into immediate left/right obstruction, occupied versus moving-tail entry, and a short route trap. Include mirrored copies. Fork the same full neural state and sensory RNG state for each pair; also run a reset-state version. Measure external drive, sensory spikes, selected relay spikes, all descending responses, the fixed signed steering sum, and chosen action. The evaluator may annotate safe/preferred moves, but those labels never enter the controller. Start with original and saved trained models, 8–16 noise repeats, and reserve new configurations for confirmation.

**Controls.** Compare full input, food-only, and occupancy/body values spatially permuted within neuron type and hemisphere. Preserve drive totals in the permutation control; simply deleting a plane changes the dynamical regime. Use a separate fixed direct policy on exactly the same observations as an engineering reference.

**Proceed/stop.** Define a practical target before looking, for example at least 75% correct paired response direction on immediate-obstruction cases with a paired interval above chance. This is a proposed project gate, not a biological standard. If the signal is absent at relays, improve input routing; present at relays but absent at the motor scalar, improve the eligible pathway; present at the scalar but below threshold, study the interface. Do not launch long training while all three cases remain mixed together.

**Cost and failure modes.** Approximately half to one day of implementation; hundreds to a few thousand short neural windows. Artificial board pairs can be invalid or out of distribution, and a single action is insufficient to prove planning. Keep short-horizon safety and long-horizon route choice separate.

### 2. A hierarchy of wiring controls that identifies what the anatomy contributes

**Question.** Is the useful property exact named pathways, hemisphere organization, transmitter balance, degree structure, or merely a random feature expansion?

**Current-code nuance.** `Brain(shuffled=True)` permutes the existing postsynaptic-index list. Before duplicate-edge coalescing, this preserves each neuron's raw incoming/outgoing edge-incidence counts and every source's signed outgoing weights. It is inaccurate to describe this as a control that preserves only global edge count. It does change input sign/weight composition and anatomical organization, and duplicate source–target pairs can reduce unique degrees when coalesced.

**Minimum test.** Add a small ladder: existing shuffle; directed edge swaps that prevent duplicates/self-loop changes; swaps constrained by presynaptic transmitter class and source/target hemisphere; a control preserving broad cell classes; and a background shuffle leaving the proposed sensory-to-steering circuit intact. Audit exactly what each preserves, including unique degree, incoming excitatory/inhibitory weight, stimulus-to-output reachability, and activity under a fixed calibration panel. Use several independently randomized graphs, not one graph with many games.

**Fair comparisons.** Keep the fixed untrained decoder for native-circuit questions. For learnability questions, train each graph's own readout from an equivalent fresh initialization and budget, then use held-out games. Do not transfer the real-brain-trained readout onto a new topology. If a separate activity-calibrated control is used, calibrate against neutral sensory probes, not Snake reward, and report both calibrated and uncalibrated results.

**Proceed/stop.** A result supports only the property destroyed at that step. If a background-shuffled network with the native pathway preserved performs similarly, claim a useful native circuit rather than whole-connectome superiority. If hemisphere-preserving random networks also work, the contribution may be coarse lateralization rather than detailed microcircuitry. Both outcomes are informative.

**Cost and failure modes.** One to two days plus evaluation. Constrained rewiring can mix poorly or preserve the entire feature of interest; report fraction of edges actually changed and graph diagnostics. A recent [flyvis preprint](https://arxiv.org/abs/2604.04033) illustrates initialization/null-model confounds, but studies moving-edge decoding for only 5–10 updates, with five degree-preserving nulls and about 0.17 accepted swaps per edge. It is methodological motivation, not evidence that this Snake result will disappear.

### 3. Turn the lesion lab into a necessity, sufficiency and rescue experiment

**Question.** Does a specific anatomical pathway mediate the sensory-to-steering transformation, rather than merely being correlated with activity or important to global excitability?

**Minimum test.** Pick one claim, such as LC10 → AOTU relay → DNa02 pursuit. First replay identical short sensory sequences with and without selected pathway edges. Next test whether the retained candidate circuit supports the response when the rest of the brain is silenced, explicitly as a separate scientific control. Finally sever an upstream connection and replay the intact trial's recorded relay spike train into that relay's downstream connections. Compare correct replay with mismatched-trial and time-shifted replay. End with continuous full-brain games as external validation.

**Controls.** Match random lesions for baseline activity, signed output strength and distance to the motor cells, not just number of neurons. Use edge lesions to distinguish a pathway from all other functions of the same cell. Clone membrane voltages, conductances, refractory state, pending delayed signals, ring-buffer cursor and RNG state for each branch. Acute lesions have residual delayed inputs; either analyze that transient explicitly or define a settling interval.

**Proceed/stop.** Require loss under the targeted intervention, smaller loss under matched controls, and selective restoration with appropriate relay replay. This is a downstream rescue relative to the severed sensory-to-relay connection, while remaining upstream of the motor decoder. These establish necessity or rescue within this simulation. Sufficiency is a separate outcome: failure with the surrounding brain silenced may reveal meaningful recurrent support. Avoid selecting the circuit and claiming confirmation on the same stimulus panel.

**Cost and failure modes.** One to two days. Replay into the final motor cells would trivially force the answer; keep the rescue at the relay, upstream of the motor decoder, and label the artificial intervention. A small sufficient circuit does not mean other neurons are irrelevant in other tasks. The distinction between anatomical paths and causal influence is central to the [effectome work](https://www.nature.com/articles/s41586-024-07982-0); this proposal measures causal effects in the model, not in a living fly.

### 4. Transplant trained parameter groups to discover what was learned

**Question.** Did the direct-board winner learn spatial sensing, restore motor gain, or mainly acquire a turn bias?

**Minimum test.** Without retraining, evaluate four combinations under the same encoder and fixed decoder: original sensory/original motor gains; trained sensory/original motor; original sensory/trained motor; trained sensory/trained motor. Then revert food, occupancy and body-order sensory groups separately. Evaluate on the counterfactual panel first, then new paired games. Plot stimulus-response curves and left/right preference alongside food, survival and collisions.

**Why it is unusually cheap.** `board_sites` already distinguishes `motor:` and `sensory:<plane>:` gain labels. Runtime group transplantation can reuse the current checkpoint identity and bounded-gain machinery. The original model remains the reference; none of these variants should replace the web default automatically.

**Proceed/stop.** If motor-only changes recover most of the score but do not improve geometry-specific probes, describe the result as improved motor responsiveness under this interface. If occupancy/body gains show repeatable selective benefits, that is evidence for learned use of those channels. To compare learning algorithms later, train motor-only, sensory-only and combined models under matched budgets and several independent seeds.

**Cost and failure modes.** A few hours plus short evaluations for frozen transplantation; longer for independent retraining. Group effects need not add because the circuit is nonlinear. A group that cannot work alone may still be essential in combination. Counterfactual reversion also changes the state distribution, so report both replayed-input and closed-loop tests.

### 5. Recheck mushroom-body access with selective, opponent-aware probes

**Question.** Does the current negative MBON result reflect an inaccessible motor pathway, or a stimulus that simultaneously activates antagonistic channels?

**Repository observation.** `mushroom_body_check.py` co-stimulates every MBON on one side at full drive and watches DNa01/DNa02. The annotation table contains 97 MBONs spanning glutamatergic, GABAergic and cholinergic classes. This is a valid probe of that population stimulus, not an exhaustive test of all MBON effects. Its smell stimulus also drives all 2,639 olfactory sensory cells, rather than separate odor-like patterns.

**Minimum test.** Screen MBON types separately by side at several doses, recording all DNs and candidate intermediate relays. Confirm any effect with new noise seeds and short sensory backgrounds. Then test excitatory and inhibitory populations separately and in selected combinations. Use two synthetic glomerular input patterns, such as two distinct `ORN_*` types, to ask whether they recruit distinguishable, reasonably sparse KC ensembles. Do not call either pattern a specific real odor without a supported receptor mapping.

**Proceed/stop.** First require a selective MBON or relay perturbation to affect a suitable motor output. Separately require odor-pattern discrimination upstream. Only if both succeed should a learning assay connect them. If they fail, retain a bounded odor-conditioning experiment read out at MBONs; do not promise Snake steering. Aso et al. found type-dependent attraction/repulsion and combinatorial MBON effects in animals, making cancellation plausible but unproven here. [Primary experiment](https://elifesciences.org/articles/4580). Work on [UpWind neurons](https://elifesciences.org/articles/85756) supplies another candidate downstream route, but type correspondence and model functionality must be checked.

**Cost and failure modes.** Half a day for the panel; hours of short-window computation depending on batch/device choices. Broad stimulation can cause cancellation or saturation; weak effects might only appear in the presence of background input. Finding an anatomical route alone is insufficient. This should be a bounded reconsideration of the old negative result, not an excuse for a large speculative training run.

## Fifteen additional ideas

### 6. Match sensory budgets before judging richer information

Compare full-board, food-only, a spatially permuted full board and an unrelated-background stimulus matched for each cell type's total external event budget. Sweep occupancy gain without altering food gain. Record actual sensory-neuron and downstream rates because equal injected events do not ensure equal network activity. Add sensory sequences yoked from a different game: unlike zero-input ablation, this preserves realistic stimulation and temporal variation while breaking its relevance to the current board. A separately labelled action-replay control can preserve turn biases and temporal autocorrelation without feedback. The important question is whether correctly located occupancy helps beyond an equally strong irrelevant stimulus. This directly tests the saturation explanation for food-only doing better. Initial implementation: a few hours.

### 7. Measure state dependence rather than assuming continuous simulation implies memory

Present the same final observation after different preceding sensory histories; branch from full saved neural states and compare final actions. Use paired-pulse delays of tens to hundreds of milliseconds, then a simple cue-disappears task if a persistent signal exists. Compare intact state, reset state and a randomized-history control. This distinguishes useful memory from residual conductance or hysteresis. If history does not affect relevant behavior, say that the present task primarily uses instantaneous sensorimotor transformations. One-day diagnostic; no new training required.

### 8. Add a small physiology validation battery that is independent of Snake score

Before changing global parameters, freeze several qualitative neural benchmarks: sugar versus bitter effects on MN9, selected antennal mechanosensory effects, lateral pursuit, and graded threat responses. Reserve one circuit as a held-out check. Shiu et al. validated feeding and grooming transformations; the proposed MaleCNS reproduction is a new test, not guaranteed transfer. [Primary study](https://www.nature.com/articles/s41586-024-07763-9). Reject a score improvement that requires loss of previously reproduced biology, or explicitly label it a task-optimized model. Half to one day to consolidate existing probes.

### 9. Give each conclusion a parameter-robustness range

Evaluate key sensory/lesion conclusions across the already plausible 0.35/0.40/0.45 global weight scales, modest stimulus-dose changes and selected time-step checks. Separately investigate edge-threshold and transmitter-sign assumptions for the specific implicated circuit. Unknown transmitters currently default to excitation; cell-specific receptor biology is not represented by that rule. Do not change uncertain signs silently. A useful conclusion survives a range, or is explicitly described as parameter-sensitive. Run the short physiology/counterfactual panel before expensive games. About one day; avoid a combinatorial full-grid sweep.

### 10. Treat independent training runs as the unit of learning reliability

Game-seed intervals quantify variation for one frozen winner; they do not quantify the chance that the training procedure works again. Run an initial small set of independent training seeds with fresh validation/test partitions and report every run. Summarize food and collision outcomes across runs, and pair environmental seeds where appropriate. Choose the practical effect size and test plan before seeing results. Existing test locks are a good foundation. This has higher information value than spending the entire budget on one much longer winner.

### 11. Test symmetry and transfer before claiming navigation

Snake starts with a fixed east-facing body, while learned gains and neuron assignment can be asymmetric. Test mirrored boards, rotated initial headings, food quadrants, short versus long bodies and different wall proximity. If supporting multiple sizes requires remapping neurons, distinguish encoder transfer from neural-policy transfer. This can reveal a stable turn bias that looks useful on a narrow distribution. Keep naturally asymmetric anatomical responses and engineering mapping asymmetry separate. A few hours for a frozen-model diagnostic.

### 12. Screen eligible synapses for causal influence before expanding plasticity

For each candidate gain group, record presynaptic activity and perform small positive/negative perturbations under fixed inputs. Measure changes in relays and steering, using paired noise. This gives an empirical local sensitivity map and identifies silent or redundant parameter groups. Select a bounded, anatomically justified set on a development panel, then confirm it on unseen contexts. Do not train thousands of parameters simply because they are anatomically reachable. This is also a go/no-go check for reward-modulated local learning: an eligibility trace cannot help a never-active path without another mechanism.

### 13. A bounded central-complex feasibility route

The annotation table includes PFL2/PFL3. Physiological work links heading and goal representations through PFL3 to steering; [Mussells Pires et al.](https://www.nature.com/articles/s41586-023-07006-3) and [Westeinde et al.](https://www.nature.com/articles/s41586-024-07039-2) provide concrete benchmark transformations. First test whether selective PFL3 stimulation produces the expected lateralized DN response. Then, only with supported column identities, inject separate heading and goal population patterns and ask whether relative-angle steering appears. Supplying those representations is an artificial interface, not a demonstration that the fly inferred its heading or goal from images. Do not infer output side from soma side. Budget one feasibility day; stop if the transform or stable dynamics is absent.

### 14. A two-odor conditioning assay with a fixed MBON readout

If the selective odor/KC/MBON probes succeed, learn association at existing KC→MBON synapses, freeze the readout, and test acquisition, reversal, retention after a delay and transfer across stimulus strength. Include unpaired reward, shuffled odor–reward timing, reward alone and plasticity-disabled controls. A change in MBON preference is already an interpretable result even if it never steers Snake. Label a custom three-factor rule as engineered unless it reproduces a specific experimentally supported plasticity mechanism. In the current simulator, stimulating dopamine neurons alone does not implement dopamine-dependent synaptic modulation. This is a separate multi-day experiment after the short feasibility gate.

### 15. Quantify how much body order survives neural transmission

Adjacent segment ranks differ by only 0.75/144 in normalized drive. At 150 Hz for 100 ms this is 0.078125 expected external events per neuron/window, compared with a 0.25-drive occupied-cell baseline of 3.75 events. This arithmetic does not prove that population or temporal information is lost; it motivates measuring discrimination. Compare raw drives, sensory spikes, descending activity and the fixed motor scalar on a held-out neighboring-rank or tail-location panel. Test a direct tail marker or coarse age bins under matched neuron/event budgets, labelled engineered. Do not call the raw invertible plane a lossless neural encoding.

### 16. Replace score-only comparisons with failure-specific outcomes

For each frozen policy, report food, moves, collision, starvation and alive-at-cap separately. Add immediate-obstacle response accuracy, food-approach bias and recovery from an imposed turn error. A controller that spins safely and one that pursues food recklessly can have similar food means for different reasons. Use a fixed evaluation horizon or appropriate survival summaries when games are censored; do not interpret every alive-at-cap game as a win. Most data already exists in the trainer records; this is a reporting and benchmark improvement.

### 17. Measure which parts of the whole connectome are causally used

Distinguish simulated neurons, neurons that spike, neurons whose activity can affect the readout, and neurons required for the tested behavior. Report activity coverage and perform staged outside-circuit lesions on held-out sensory sequences. This does not justify replacing the required whole-brain training loop with a reduced model; it is a diagnostic control. An honest result such as “the entire model runs, and this particular task depends on this small pathway” is more informative than a neuron-count headline. Activity alone is not causal importance. Half to one day after the intervention harness exists.

### 18. Temporal looming inputs grounded in known feature roles

The current threat channels are static danger flags. A separate assay could supply LC4-like angular-expansion-speed input and LPLC2-like angular-size input across a temporal looming sequence, with independently controlled size/speed. Ache et al. experimentally distinguished these components at the giant fiber. [Primary study](https://www.sciencedirect.com/science/article/pii/S0960982219301381). Ask whether removing one channel selectively changes the predicted response component. This is a clearer test of sensory integration than immediately trying to make threat intensity fix Snake. It still injects precomputed sensory features and must be labelled accordingly. One-day probe; defer a game encoder redesign until it passes.

### 19. Use the newly verified medulla coordinates for a narrow spatial-input pilot

The companion annotation audit found both optic-lobe hex coordinates for 1,762 traced Mi1 neurons and 1,767 traced Tm1 neurons, whereas the current LC10/LC4/LPLC2/LC12/Tm3 pools lack those fields. This supports a more anatomical position assignment at an earlier visual stage; it does not establish natural vision. First establish coordinate orientation and coverage, then present a moving bar, a stationary spot and a mirrored trajectory at matched input budgets. Ask whether spatial position and motion can be distinguished downstream before trying whole boards. Mi1/Tm1 belong to different contrast-processing pathways, so identical static positive drive is only an initial perturbation assay. [Physiology](https://www.nature.com/articles/nature13427). Modern visual-system modelling also required task-optimized dynamics; wiring alone should not be assumed sufficient. [Primary modelling study](https://www.nature.com/articles/s41586-024-07939-3). Budget one feasibility day and stop at the downstream-signal test if it fails.

### 20. Test sensitivity to the arbitrary receptive-field assignment itself

The current direct-board positions are assigned by radial order and bodyId within hemispheres, not measured receptive fields. Hold the exact neuron set fixed and create several alternative position assignments within hemisphere and exact neuron type. Separately test choosing a different matched neuron subset; do not mix subset selection with position permutation in the first comparison. Evaluate frozen original and trained models on the diagnostic panel, then independently train only promising interface families under equal budgets. A large mapping effect would show that the engineering interface is an important source of variance; it would neither prove nor disprove the connectome's computational capacity. This experiment can be more informative than comparing optimizers while treating one arbitrary map as fixed biological fact.

## Reasons the bottleneck-first recommendation could be wrong

The existing pilot already improved held-out full-board performance. Twenty-four generations and two training games per candidate do not exhaust a noisy 60-parameter search. The goal of diagnostics is to buy information cheaply, not to prohibit useful exploratory training.

- **Undertraining may be the main limitation.** If development/validation curves are still improving and several fresh small runs reproduce the gain, a modest additional training budget is justified. Existing tooling may make these replications cheaper than building an elaborate new assay. Use new runs and seed ranges; do not reopen a tested run.
- **Local perturbations can miss nonlinear solutions.** Joint gain changes can cross thresholds or alter inhibition when every individual perturbation appears ineffective. A negative sensitivity screen supports “no effect detected under these states and perturbations,” not “this circuit cannot represent the behavior.”
- **More injected events need not mean more useful or saturating activity.** Cell types differ in downstream influence, and common-mode input could participate in selective inhibition. Measure actual propagation before declaring the occupancy plane harmful.
- **A handmade exam can become another overfitted training set.** Keep it small for debugging, and collect most confirmation states automatically from multiple policies, lengths and difficulty levels. Group mirrors and near-duplicates together when splitting data. Freeze a separate confirmation set. Score probability mass on valid action sets rather than forcing one arbitrary answer when several actions are safe; treat finite-horizon route labels as horizon-specific outcomes, not guaranteed optimal moves.
- **Stateful behavior can be missed by reset trials.** Include neural snapshots from actual rollouts, controlled histories, and a closed-loop confirmation subset. A signal absent from a reset brain may appear under the states that the controller actually visits.
- **Ablation is not a learning-method comparison.** Full-board versus food-only on the same frozen model tests that model's input dependence. To claim that richer information helps learning, independently train both input conditions with matched selection opportunity and a declared resource budget. Equal episodes and equal simulated moves cannot generally both hold when one policy survives longer; report realized episodes, neural windows and wall time.

If the counterfactual panel remains negative, the highest-confidence alternative scientific investment is a precise causal demonstration of the already active LC10→AOTU→DNa02 pursuit pathway: edge loss, matched controls, relay rescue and parameter robustness in a simple pursuit task. The two-odor KC→MBON assay is a more exploratory alternative when internal learning is the central goal. Neither requires claiming that failure of this engineered Snake interface reveals a limitation of a biological fly.

## Existing evaluation details worth fixing before the next evidence table

These are observations from static code review, not fixes applied during brainstorming.

1. `scripts/untrained_control.py` accumulates steering spikes for all batch slots, including games that have already died, while its denominator counts moves only while alive. The printed steering-spikes-per-move statistic can therefore be inflated. Food and survival scores are not affected by this particular issue. Mask accumulation by alive slots at the start of each move.
2. `scripts/lesion_scores.py` pairs board seeds but generates separate Poisson tensor columns for the different lesion conditions. It is not a perfectly noise-paired comparison. Shared input-event tapes or independently seeded per-game generators would reduce variance for the causal panel.
3. Count-matched random lesions can select mostly inactive cells. Add activity/degree/path-distance matched controls before interpreting selective susceptibility as evidence of a specific computation.
4. A reset must include all dynamic neural variables and RNG state for a paired counterfactual. Copying only voltages is insufficient because conductance, refractory counters and delayed signals also carry history.
5. The response bank carries state between randomized inputs and resets every four trials. That tests a particular history distribution, not the exact histories a competent or failing game controller visits. Keep bank estimates labelled and validate important comparisons with continuous closed-loop simulations, as the current documentation already does.
6. Any new cache-dependent anatomy experiment should include a full connectome/parameter identity in its cache provenance. The general connectome cache currently accepts a cache when neuron count matches; neuron count alone cannot detect changed threshold/sign/edge content. This is an experiment-provenance concern before threshold or transmitter sensitivity studies.

## Suggested order

Start with the counterfactual panel, input-budget controls and frozen gain transplantation. They can help locate whether the next change belongs in the input mapping, eligible pathways or movement interface. Then run the structured wiring controls and lesion/rescue study for the strongest causal claim. Keep selective MBON and PFL3 assays as bounded exploratory projects with explicit stop rules. Combine these diagnostics with modest fresh training replications when those are inexpensive, and let both sources of evidence guide larger training budgets.
