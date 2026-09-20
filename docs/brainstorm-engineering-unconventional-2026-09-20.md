# A second pass: experiments that could change the direction of Fly Snake

Prepared 2026-09-20. These are proposed separate research tracks, not changes made to the running project. Every brain condition would continue to simulate the entire continuous connectome. A diagnostic or a labelled non-brain null must never silently select actions in a brain condition. Existing published pilots and their frozen decoders remain unchanged.

The opportunity is to ask a smaller, sharper question than “can we raise the Snake score?” A compact suite can reveal which parts of the circuit work, which fail, and whether a successful intervention transfers to Snake.

## A. Give the brain tasks closer to the signals it already receives

### Pursuit without a growing body

Create a simple arena in which a moving spot or food target appears at varying relative bearings. Use the existing sensory stimulation and fixed left/straight/right output, constant movement speed, and no growing snake body. Measure angular error, time to capture, and turn direction after a target jump. Include a stationary-target phase, a moving-target phase, and a phase where food and a single obstacle compete.

**Why it matters:** Snake bundles pursuit, geometric body avoidance, memory, and starvation. A pursuit assay could show intact wiring's competence without burying it under route planning that this input/output interface cannot express. A clean lesion result here would also be easier to interpret than a change in total food after many interactions.

**First gate:** original real wiring must outperform matched scrambles with the identical fixed output rule, and the effect should weaken under the known pursuit-pathway lesion but survive a matched random-cell lesion. If not, investigate the stimulation/output mapping before building a larger world.

**Cost:** about one day to implement from the current arena and runner, then a bounded paired evaluation. This is an engineered navigation task, not a fly biomechanics simulator.

### Directional escape as a separate experiment

The current instinct rule uses giant-fiber asymmetry as a direction cue. Consider a small escape assay in which looming input arrives at different locations and the output is evaluated for directional avoidance. Before choosing an output rule, examine existing LC4 downstream DNs associated with direction, rather than assuming the giant fiber encodes every component of escape.

A primary study reports LC4-connected DNp02 and DNp11 contributions to directional takeoff and distinguishes that role from giant-fiber activation. This motivates a feasibility probe, not a promise that this generic LIF model will reproduce the behavior. [Synaptic gradients transform object location to action](https://www.nature.com/articles/s41586-022-05562-8).

Read-only inspection confirmed traced left/right entries for DNp02, DNp11, and DNp06 in the local MaleCNS annotation file. The first experiment should record their response to graded left/right/ahead sensory input and perform matched lesions, with no game or trained decoder. Only if a direction-specific signal exists should a new explicitly declared output rule be evaluated in a separate task. DNp02/DNp11 findings concern forward/backward escape and should not be automatically treated as a left/right Snake decoder.

**First gate:** repeatable stimulus-location information survives noise, is disrupted by a specific pathway lesion, and is useful for the direction dimensions actually measured by the source study. If only a generic alarm survives, present an alarm assay and stop there.

**Cost:** several hours for the initial response probe; 1-2 days for a small task after a positive result.

### A disappearing-cue memory assay

Present a left or right food cue, remove all directional input, wait a controlled interval, and then allow one fixed-decoder choice. Sweep blank delays from a short interval through several 100 ms windows. Compare a continuously running brain with a brain reset just after the cue, and compare real wiring with controls. Record whether directional information remains in broad descending activity separately from whether it survives in the four fixed steering neurons.

**Why it matters:** the project simulates continuous neural state, but ordinary Snake scores do not establish useful memory. This assay can positively demonstrate a timescale of retained information or show that continuity is behaviorally irrelevant under these settings. The latter is still useful.

**First gate:** on held-out cue/noise sequences, a response depends on the vanished cue after a predeclared delay; resetting the intervening neural state removes that dependence. Counterbalance left/right cues and final neutral input so no persistent board feature leaks the answer. Do not feed the cue identity to the decoder.

**Cost:** 0.5-1 day for an assay and plots, with modest brain time. Do not call residual membrane activity long-term memory or learning.

### A sequence of obstacle puzzles, not a teacher

Build a small procedural curriculum of starting conditions: approach a wall, choose around one block, avoid a U-shaped trap, distinguish growing versus moving tail, then standard Snake. Reward only actual outcomes of actions produced by the whole brain. The curriculum changes initial conditions and the frequency of encounters; it does not supply a safe action or overwrite the decoder.

**Why it matters:** two random short training games per candidate can supply little experience of rare critical configurations. Targeted initial states make the weak behavior observable more often.

**First gate:** improvement transfers to untouched ordinary-board games and does not depend on recognizing the handful of constructed puzzles. Include procedurally generated variations and a separate whole-game test. If an agent learns only the curriculum, describe that narrower result.

**Cost:** 1-3 days including valid-state generation and a bounded search pilot.

## B. Test whether complex learning is doing more than simple tuning

### Four motor gains versus sixty pathway groups

Run a low-dimensional control that can only scale existing inputs to each of the four fixed steering neurons, alongside the current 60-group search. Both preserve connectivity, signs, the encoder, and the fixed decoder. Use equal simulated-move budgets and independent training seeds. An even simpler companion control can tune one common allowed sensory gain in a separately identified encoder experiment.

**Why it matters:** a model can improve by shifting motor excitability into the range where the fixed threshold produces more turns. That is still an internal synaptic change, but it is a narrower result than learning useful board-dependent routing. The saved final test shows a large change in overall straight-action frequency, which makes this a particularly relevant control; it does not establish the explanation.

**First gate:** the larger parameterization needs to beat the simple control on fresh games and on paired obstacle-location assays. If it cannot, retain the smaller model and describe the mechanism as gain calibration.

**Cost:** a few hours to define grouped sites, then the same bounded compute budget as the main model.

### Match the action statistics without using a brain

Build two clearly labelled null controllers: one samples left/straight/right at the learned model's training-set frequencies; the other matches its short action-transition frequencies. Fit those statistics only on training or diagnostic rollouts, freeze them, and evaluate on new game seeds. A repeated pre-recorded turn sequence is another transparent null. These controls must be visually and numerically distinguished from brain play.

**Why it matters:** no-input gives almost no motor signal and consequently mostly straight movement. A brain that merely turns more often may outlive that weak control without using detailed board structure. Matching turn statistics asks whether context-dependent neural decisions add value beyond generic movement patterns.

**First gate:** trained brain behavior beats both nulls on food and safety under a predeclared budget. If not, focus on stimulus-dependent motor margins rather than additional training.

**Cost:** 2-4 hours and cheap game-only rollouts. Existing final-test action counts can motivate the idea, but must not be used to fit a null and then claim an untouched comparison on that same test set.

### Preserve total incoming strength while learning its distribution

Design a separate bounded-plasticity control in which total excitatory and inhibitory input strength to each target is approximately conserved while gains redistribute among existing source groups. Compare against unconstrained bounded gains and the simple motor-gain control.

**Why it matters:** this separates changing overall excitability from changing which sensory pathways influence steering. It also makes a positive result easier to explain: the same overall input budget is allocated differently among existing connections.

**First gate:** a feasible constrained parameterization can change stimulus-dependent steering while retaining all signs and declared gain bounds. If constraints leave almost no controllable space, report that instead of widening the rules after viewing results.

**Cost:** 1-2 days to implement and verify parameter constraints, plus a bounded pilot. This is engineered regularization, not a claim about fly homeostasis.

## C. Turn debugging into an instrument

### A motor controllability atlas

For each of a small set of continuous brain states, perturb one parameter group in both directions and measure the average change in the four motor neurons across matched noise. Assemble a task-by-parameter response matrix. Visualize directions that influence pursuit, obstacle avoidance, common-mode firing, and left/right bias. Groups that only alter unrelated DNs should be visibly separated from groups that can affect the fixed output.

**Why it matters:** a 60-parameter optimizer may be searching a much smaller effective behavioral space. The atlas can guide a smaller parameterization or identify the need for an additional existing relay pathway. It could also make an excellent interactive explanation of why anatomy constrains learning.

**First gate:** finite-difference effects replicate at held-out states and more than one perturbation size. Spike thresholds make local derivatives discontinuous, so show distributions and action changes rather than pretending the system is smoothly differentiable. Any later optimizer built from this information must still choose every move through the full brain.

**Cost:** about one day for the tool plus bounded simulation time.

### Measure recovery, not just the peak response

Apply a brief sensory pulse or a temporary lesion during a standardized live sequence, then restore baseline input and watch the neural and behavioral recovery. Report latency to motor change, duration of the disturbance, first resumed food-seeking response, and whether the network returns to a comparable activity regime. Compare the original model and trained internal-synapse model.

**Why it matters:** a large peak response is visually impressive but may be useless if the circuit remains saturated or takes too long to recover. A trained model might improve a static score while becoming fragile to sensor bursts. This also directly connects the real ultrasonic-hand idea to the continuous-brain claim.

**First gate:** the probe has reproducible timing in simulation milliseconds, and its results separate the input pulse from subsequent recurrent activity. No wall-clock/UI animation timing should be mistaken for neural latency.

**Cost:** 0.5-1 day for reusable pulse protocols and response plots.

### A compact “skeptical reviewer” run

Create one small, shareable experiment bundle that takes a specific claim and runs its strongest feasible controls: same game seeds, identical decoder where required, original/trained/lesioned conditions, real versus matched shuffled wiring, full versus ablated input, and the action-statistic null. Produce one page of results linked to complete per-game records, exact settings, and failed as well as successful seeds.

**Why it matters:** the project already contains unusually honest caveats, but evidence is spread among scripts, models, and prose. A reviewer or judge should be able to ask “what would falsify this interpretation?” and immediately find the relevant run.

**First gate:** another developer can reproduce the exact small benchmark in the same environment without interpreting verbal instructions, and the report does not merge bank estimates with continuous-brain results. Keep each report centered on one claim; a giant collection of incomparable scores would undo the benefit.

**Cost:** 1-2 days using the proposed common manifest and evaluation schema.

## A feasible selection order

Start with the controllability atlas, gain-only control, and action-statistic null because they can explain the existing results without committing to a new world. If the current input/output interface is the bottleneck, build the pursuit assay and directional-escape feasibility probe. Add the disappearing-cue assay only when the project is ready to investigate temporal computation directly. Each of these can succeed as a scientifically useful negative result.
