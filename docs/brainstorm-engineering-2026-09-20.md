# Engineering and experimental design ideas for Fly Snake

Prepared 2026-09-20 from read-only inspection of the working tree, including the uncommitted compiled CPU kernel and direct-board/synaptic experiments. This is an idea portfolio, not a report of newly run brain experiments. Effort estimates assume one developer familiar with the project and exclude long experiments. Acceptance thresholds below are proposed engineering gates, not established scientific standards.

The strongest next investment is diagnosing why useful sensory information fails to change the fixed motor output. More generations alone would be difficult to interpret. In parallel, the project can become substantially easier to reproduce and demonstrate through a common provenance format, measured CPU deployment, and bounded live-session transport.

## Five ideas worth doing first

### 1. Build a sensory-to-steering diagnostic suite before the next training run

**Evidence.** The saved direct-board probe changes 49 input neurons and about 35 descending neurons when body shape changes, but the mean fixed steering difference changes by 0.0 spikes. Moving the left wall changes about 90 descending neurons but only -0.1875 steering spikes. Near-right food produces steering differences between -2 and 0 in all 16 original-brain probe repetitions, which never crosses the fixed decoder's strict less-than -2 right-turn threshold. Near-left food gives 4 left turns in 16 repetitions. These are observations for the particular saved boards and original brain, not a demonstrated universal left/right defect. See models/brain-board-probe.json and flybrain/readout.py.

The full-board occupancy plane covers 23 by 23 relative locations, while the actual board contains 12 by 12 cells. Thus 385 of 529 occupancy inputs represent outside-board space at full drive before smoothing. On the reference board, a lightweight encoder-only check found 432 occupancy inputs at full drive after smoothing, and total input drive of approximately 442.801 occupancy versus 56.500 food and 0.776 body order. These are external input values, not neural spike counts or proof that occupancy dominates the motor circuit. The corresponding evidence is also recorded in docs/brainstorm-evidence-2026-09-20.json.

The body-order plane changes by only 0.75/144 = 0.005208 in amplitude per adjacent segment rank. At the declared 150 Hz input rate over one 100 ms window, that corresponds to 0.078125 expected additional external events, on a baseline of roughly 3.75 or more events per occupied body cell. This is a calculation of the stimulation process, not measured neural spikes. It motivates measuring whether fine segment order survives noise, rather than assuming that distinct raw values imply usable neural information.

**Build.** Create a small catalog of paired boards: same food/different obstacle side; same occupancy/different body order; same food direction/different range; wall approaching; mirrored scenes; safe versus blocked pursuit. For each pair, run the entire continuous brain with matched stimulus-event streams, initially from reset and then after several standard histories. Plot the change at three stages: stimulated cells, descending cells, and the four decoder neurons. Add motor-margin histograms showing the distance to the existing -2/+2 thresholds. Keep the decoder fixed throughout.

A dose sweep should independently lower occupancy, body, and food input amplitudes and compare occupied-boundary-only representations with the current outside-area fill. Each is a new, explicitly versioned encoder experiment. The point is to identify whether the limiting step is input scale, anatomical routing, noisy threshold crossing, or too few plastic pathways.

**First test.** Six pairs, 16 noise seeds, original and saved trained synapses, then three occupancy gains on the two most revealing pairs. Store every response and action. This is a diagnostic test set; do not treat it as an untouched game-performance test after using it to design the encoder.

**Success gate.** Find a repeatable, appropriately directed change in steering when only obstacle location changes, while preserving food pursuit. A proposed early gate is at least 80% correct avoidance on held-out paired layouts and a material reduction in mirror asymmetry. The exact threshold should be chosen before looking at the held-out layouts.

**Stop/change direction.** If obstacle/body manipulations remain separable in the broad DN response but cannot reliably alter the four steering neurons across the entire allowed gain range, stop optimizing those same 60 groups. Investigate additional existing pathways or choose a task that the accessible circuit can express. Do not tune the frozen decoder to hide this bottleneck.

**Effort.** 1-2 days for a reusable assay and visual report; the first compact probe can be assembled in several hours from scripts/probe_board_encoder.py and BoardRunner.

### 2. Give every cache, model, bank, and session the same provenance contract

**Evidence.** New synaptic checkpoints already validate the connectome fingerprint, site identities, simulation settings, and decoder. Direct-board checkpoints additionally validate encoder identity. The older connectome cache is accepted when the neuron count matches. Response-bank compatibility checks include several parameters and readout body IDs, but not a hash of the complete anatomy and weights. Policy.save/load stores only weight and bias arrays. These are differing levels of protection, not evidence that the currently committed artifacts are mismatched.

**Build.** A small versioned manifest shared across artifacts: source-data hashes; traced-neuron IDs and order; edge threshold and transmitter-sign convention; signed-edge hash; neuron annotations used for selection; encoder identity; readout body IDs and order; simulation constants and kernel identity; model hash; training/evaluation seed ledgers; software/device information; source commit plus dirty-worktree fingerprint. Use schema validation and atomic writes. An older model with no manifest can remain usable through an explicit legacy compatibility path rather than being silently assigned verified provenance.

**First test.** With tiny synthetic data, change weights while preserving neuron count, change the body-ID order, alter a transmitter sign, and change a simulation constant. Confirm that the corresponding cache/bank/checkpoint either rebuilds or rejects the mismatch. Then demonstrate that an unchanged artifact loads without rebuilding.

**Success gate.** One command explains exactly which anatomy, encoder, decoder, weights, seeds, and kernel generated any reported score or replay. Stale content must not be accepted merely because shapes match.

**Stop/change direction.** Do not introduce a database or registry service for this. Plain manifests beside existing NPZ/JSON artifacts should suffice until there is a demonstrated multi-user need.

**Effort.** 1-2 days. Start with the connectome cache and saved readouts, where the gap relative to the new synaptic format is largest.

PyTorch explicitly does not promise identical results across releases, platforms, or CPU/GPU even with identical seeds. Record these dimensions and distinguish exact within-environment replay from cross-environment scientific replication. [PyTorch reproducibility documentation](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

### 3. Turn existing CPU acceleration into measured end-to-end improvements

**Evidence.** The saved full-connectome kernel checks report 3.54x and 4.81x warm kernel speedups over 24 continuous windows, with exact all-neuron spike counts and state agreement within 1e-5. The direct-board trainer already uses the compiled CPU path. The live server currently constructs Brain without compiled=True. The compiled path accepts only a single CPU brain, and resize to a larger batch rejects it. Therefore enabling it globally would need deliberate layout-transition handling.

**Build.** Add a benchmark that separately measures cold startup, first JIT call, encoder work, stimulus random generation, integration, synaptic propagation, decoding, JSON construction, network publication, and completed game throughput. Benchmark solo, 8-brain arena, and 16-brain swarm with representative activity. Record median and p95 tick time, peak memory, thermal/power mode, and completed simulated moves. CPU timings should use perf_counter; CUDA sections need correct synchronization or CUDA events. PyTorch's profiler can attribute CPU and GPU operation time, memory, and traces. [PyTorch profiler documentation](https://docs.pytorch.org/docs/2.14/profiler.html).

If solo gains survive the complete pipeline, add an optional backend choice with a warmed compiled solo brain. Define how layout changes recreate or transfer state; do not quietly reset neural state mid-game. Preserve the existing batched implementation unless its own benchmark justifies a change. Keep the original kernel available for verification.

**First benchmark.** A prerecorded, deterministic 24-pattern stimulus sequence plus a real fixed-seed game, 200 windows each after warm-up. Test original Torch, firing-column CPU, compiled CPU, then worker counts 1/2/4 for independent candidate evaluations. Run comparisons one configuration at a time on an otherwise idle machine.

**Success gate.** A proposed adoption threshold is at least 25% lower p95 end-to-end solo latency without changed actions on deterministic validation sequences, stable memory, or layout-transition failures. Publish measured startup cost and break-even duration, not just the best kernel ratio.

**Stop/change direction.** If serialization, drawing, publication, or encoding dominates, improve that stage. Do not pursue a custom GPU rewrite on the basis of a CPU microbenchmark.

**Effort.** Half a day for a useful profile; 1-2 days for safe optional live integration. GPU kernel work would be a separate, substantially larger project.

### 4. Make live sessions resilient to slow viewers, stale inputs, and invalid commands

**Evidence.** The ordinary /ws loop awaits each client's send in sequence. Audience publication already sends concurrently with a one-second timeout, but publication still sits on the simulation path. Generic host commands use an unbounded list drained with pop(0); human turns have a separate bounded queue. External sensor freshness is timestamped when the queued command is applied, not when it arrived. The background loop is started with create_task and there is no explicit retained task health state. These are specific robustness opportunities; no live failure was induced during this brainstorming.

**Build.** Give each viewer a one- or two-frame latest-state queue and a single writer, replacing superseded visualization frames while preserving reliable command receipts. Bound ordinary command ingress. Coalesce continuous settings such as sensor drive and selection by source; preserve ordered discrete events such as feedback and human turns. Add received-at timestamps and reject stale sensor commands based on ingress time. Keep applied-move IDs in receipts. Validation should happen before any mutation, with explicit rejection messages.

Retain the simulation task and expose a small health state: loading/running/paused/failed, last successful move, elapsed wall time, and a readable failure reason. The FastAPI lifespan mechanism provides a suitable place to own and clean up this task. [FastAPI lifespan documentation](https://fastapi.tiangolo.com/advanced/events/). The networking pattern should be implemented for the actual ASGI stack; the websockets project's discussion documents why sequential broadcast can wait behind a slow client and how per-client queues help. [WebSocket broadcast design](https://websockets.readthedocs.io/en/stable/topics/broadcast.html).

**First test.** Use a fake, deterministic brain and three sockets: one healthy viewer, one socket whose send blocks, and one sensor producer. Add a 1,000-command burst, an invalid lesion regular expression, a disconnect/reconnect, and a pause/resume. Verify simulation progress, bounded memory, ordering of discrete commands, and sensor expiry. No GPU is required.

**Success gate.** A stalled viewer cannot stop the simulation or healthy viewers; ordinary queues stay within declared bounds; no sensor older than its hold time is applied; an invalid command cannot terminate the brain loop. A proposed load gate is less than 10% tick-time regression with a deliberately stalled client.

**Stop/change direction.** Do not add complex message brokers or distributed services for this single experiment. A bounded in-process design is enough.

**Effort.** 1-2 days. Command validation and task-health reporting are useful first slices.

### 5. Make the next experiment answer an explicit scientific comparison

**Evidence.** Direct-board training improved over its original full-board baseline on 32 held-out games, but all five tested conditions had 32/32 collisions. Full-board was numerically below the old input and below food-only, with comparison intervals including zero. The first synaptic pilot's gain interval included zero. Current intervals primarily describe games for one selected checkpoint, not independent training runs. The code's test lock is a valuable guard against continuing a tested run.

**Build.** Before another run, write one short experiment manifest stating the question, one primary outcome, baselines, compute budget, parameter-search space, validation schedule, final test seeds, and interpretation of a negative result. For direct-board input, the useful question is whether extra occupancy/body information improves behavior beyond food-only under matched training budget. Compare independently trained models, rather than only deleting inputs from one model after training. Preserve that ablation as a separate mechanistic test.

Run several independent optimizer seeds, and cross a subset of game seeds with several neural-noise seeds so difficult boards can be separated from unstable neural decisions. Report food, collision rate, starvation, survival at the cap, and uncertainty across trained models. Include wall-clock cost and actual brain moves. Use the same decoder and signed existing-edge constraints in every applicable condition. A small fixed budget that produces a clear negative answer is more useful than an open-ended training run.

**First test.** Freeze three encoder conditions after diagnostic work: legacy reference, food-only, and best proposed full-board mapping. A feasible pilot is 3 independent training seeds per trainable condition, equal simulated-move budgets, and a small crossed game/noise validation subset. Only after a candidate clears that stage should a new final test be opened. Exact sample sizes should follow observed variability and available compute; do not claim that three runs are definitive.

**Success gate.** Extra input earns its cost through a repeatable improvement over equally trained food-only controls across independent runs, with no hidden decoder changes. Predeclare what difference would be practically worthwhile before the final test.

**Stop/change direction.** If the full-board condition remains inferior or learns only turn frequency, move effort into sensory mapping and task design. Keep a negative result as part of the project story.

**Effort.** Half a day for a common runner/report; experimental compute depends on the chosen budget.

## Nineteen additional ideas

| # | Idea and reason | First bounded test / useful outcome | Effort |
|---|---|---|---|
| 6 | **Resumable frozen tests.** Current trainers lock a run as soon as test.json exists; an interrupted final test cannot simply resume through the existing flags. Keep the training lock permanent while allowing continuation of the identical frozen evaluation. | Store a per-condition/per-seed completion ledger and immutable model/config hashes. Interrupt after a game, resume, and recover exactly the uninterrupted report without replaying already exposed seeds as training data. | 0.5-1 day |
| 7 | **Circuit controllability map.** Determine which of the 60 gain groups can actually change the fixed motor output before spending many generations searching all of them. | At a fixed set of brain states and boards, apply small paired positive/negative gain perturbations, measure steering response and action changes, and rank useful directions. Diagnose flat versus saturated groups. This is an offline diagnostic, never a controller. | 1 day plus bounded brain runs |
| 8 | **Split the giant “other” groups using anatomy and measured influence.** Some parameters affect hundreds of edges while others affect one. A read-only checkpoint inspection found 5/60 direct-board gains within 5% of the log-bound magnitude; this does not prove that wider bounds would help. | Use the controllability map to divide at most two heterogeneous groups by transmitter, source family, or pathway. Compare equal budgets and regularize changes toward original strengths. Keep signs and connectivity intact. | 1-2 days |
| 9 | **Bilateral and rotation diagnostics.** Rotation of input planes is tested, but a biological hemisphere need not be numerically symmetric and body-ID-based assignment can add engineering asymmetry. | Mirror a catalog of scenes; report left/right motor-margin distributions before and after training. Distinguish input-mapping effects from anatomical effects by mirroring the mapping independently. Do not force anatomical symmetry just to improve a score. | 0.5-1 day |
| 10 | **State-fork counterfactuals.** Replaying the same board from reset does not isolate a lesion if the live brain has a history. | Snapshot membrane, conductance, refractory state, pending delay ring, cursor, all RNG states, and board state. Fork one short sequence with and without a lesion from the exact same state; replay the unmodified branch exactly. | 1-2 days |
| 11 | **Causal decision recorder.** A visually bright cell is not necessarily responsible for a turn. | Save input levels, DN spike counts, decoder margin, chosen action, reward, model hashes, and selected circuit ablations. For linear readouts expose exact per-feature logit contributions; label these algebraic contributions, not causal proof. Add an occasional state-fork ablation to test causality. | 1-2 days |
| 12 | **Check response-bank calibration on encountered states.** Existing banks use randomized histories and periodic resets, while actual policies visit biased sequences. | Collect a held-out continuous-brain trace for an existing readout; compare actual action distributions and state occupancy with bank predictions. Extend banks to previous/current-state conditioning only if the measured mismatch matters. Synaptic/direct-board training should keep its existing continuous-brain loop. | 1-2 days |
| 13 | **Explicit neural-noise fixtures.** Matching integer seeds across encoders with different input-neuron counts does not mean identical Poisson events for common neurons. | For a diagnostic only, save external-event arrays keyed by body ID and time, then replay them through both kernels. Separate event equality from floating-point equality. This enables more precise kernel and input-ablation comparisons. | 1-2 days |
| 14 | **Long-horizon kernel equivalence ladder.** The existing 24-window checks are strong but do not cover every future use. | Add synthetic impulse/delay/refractory fixtures, extreme allowed gains, lesion on/off, varied recording subsets, and a few long continuous scenes. Compare exact spikes where expected; log the first state or spike divergence rather than only aggregate score. | 0.5-1 day plus compute |
| 15 | **Single source of simulation constants.** The compiled kernel currently repeats several constants as numerical literals. | Pass or compile the canonical constants and record them in manifests. A tiny test changes a constant in an isolated fixture and confirms that both backends use it. This is drift prevention, not a claim that current constants disagree. | 2-4 hours |
| 16 | **Episode throughput and worker scheduling.** Direct-board training allocates candidates to workers in fixed strides; long-lived candidates may leave one worker as a straggler. | Profile per-candidate durations and memory at 1/2/4 workers. If imbalance matters, use a worker-owned runner pool with dynamic candidate assignment while preserving candidate order and per-game seeds. Reuse the executor across generations if startup cost is measurable. | 0.5-1 day |
| 17 | **Hoist invariant encoder calculations.** Every board encoding rebuilds a Gaussian impulse to calculate a constant normalization peak. | Cache the normalization for each sigma and encoder identity, then compare output arrays exactly and time end-to-end games. Precompute coordinate transforms or boundary masks only if encoding remains material. | 1-3 hours |
| 18 | **Make saved sessions replayable experiments.** Current frame recording supports visual replay, but not exact continuation of learning or a brain-state fork. | Write a session manifest and an append-only command/receipt log with decision IDs and periodic state snapshots. Restore one short session and verify both neural outputs and learner parameters. Retain a small ordinary JSONL recording path for uncomplicated demos. | 1-3 days |
| 19 | **Unify evaluation records.** Several scripts report similar fields under different names or omit categories such as starvation. | One schema for episode seed, neural seed, model/encoder/kernel IDs, score, moves, end reason, alive-at-cap, and timing; one report generator for paired comparisons. Check matching seeds rather than silently pairing by list position. | 0.5-1 day |
| 20 | **A staged training curriculum that never selects the action.** Full Snake mixes food pursuit, turn control, body avoidance, and route planning from move one. | Start with free-space pursuit, then single-wall avoidance, then one body obstacle, then ordinary boards. All actions still come from the whole brain and fixed decoder; only initial conditions and rewards change. Evaluate every stage on ordinary untouched Snake boards before claiming transfer. | 1-3 days plus training |
| 21 | **Live-learning stability panel.** The summed readout update scales with batch size and aggregate rewards, and one failed training seed is already documented. | Track action entropy, logit scale, update norm, reward source, and experience count. Compare baseline subtraction, batch-size normalization, or an update-norm cap in separate readout-only experiments with multiple seeds. Measure learning speed in brain moves and wall time. | 1-2 days |
| 22 | **Pre-demo health and recovery rehearsal.** The project spans local data, optional packages, models, GPU/CPU, frontend, WebSocket, phones, and tunnels. | One read-only diagnostic confirms artifact compatibility, file availability, selected backend, a tiny synthetic kernel fixture, socket reachability, and current frame age. Rehearse fallback recording and restoration after network loss; show explicit live/replay status. | 0.5-1 day |
| 23 | **Circuit sensitivity envelope.** A useful interpretation should survive small simulation choices rather than depend on a single threshold coincidence. | On a fixed diagnostic scene set, vary weight scale, stimulus rate, window duration, and time step one at a time without tuning the decoder to each result. Record which qualitative pathways and action effects persist. Keep comparisons to the saved model's valid settings separate. | 1-2 days plus compute |
| 24 | **Test neural distinguishability of body order.** Exact segment ranks differ mathematically but may be hard to distinguish through a brief noisy stimulus. | Use boards with the same occupied cells and different valid body order. Measure distinguishability in input-event counts, broader DN responses, and fixed motor responses. Compare a separately versioned coarse-age-band or explicit-tail-marker encoder if fine rank cannot be read reliably. Any diagnostic classifier must remain outside the action loop. | 0.5-1 day plus bounded probes |

## Observations to verify before calling them bugs

### Exact scope of an audience-feedback demonstration

The implemented update can be explained exactly without claiming improved game performance. Let x_i be the saved log(1 + spike-count) feature vector, p_i the saved action probabilities, e_i the one-hot chosen action, r_i its reward, and eta the learning rate. The current code sums:

- Delta W = eta times sum_i r_i (e_i - p_i) x_i transpose.
- Delta b = eta times sum_i r_i (e_i - p_i).
- Therefore, at any query feature x, Delta logits(x) = eta times sum_i r_i (e_i - p_i) (x_i dot x + 1).

For one targeted saved decision evaluated at its own feature vector, this becomes eta times r_i times (squared norm of x_i + 1) times (e_i - p_i). The +1 comes from the bias update. Positive feedback increases every chosen-versus-other logit margin at that saved feature vector, even if feedback is delayed. It does not guarantee a better action elsewhere, improved future reward, or a particular numerical probability change.

A phone vote normally targets one fly, so it contributes one term even in the 16-fly swarm. Automatic learning sums the whole batch without dividing by batch size. The aggregate update is exactly 16 times larger only in the special case of 16 identical, aligned experiences and rewards; ordinary batches can reinforce or cancel each other. Repeated votes on one saved decision reuse its saved probabilities, so their parameter changes add without automatically weakening as the current policy becomes confident. Request IDs are used for receipts rather than deduplication.

A fair coaching comparison should fix the number of flies, automatic simulation windows, learning rate, feedback budget, and vote-aggregation rule; use separate learners for treatment groups; record actual eligible experiences and update norms; and test frozen readouts on new uncoached games. The existing learner.moves counter increments by batch action count, including some respawn windows with no eligible arena move, so it is not by itself an exact count of rewarded training experiences.

### Specific code observations

- The host lesion command accepts regular-expression strings. pandas string fullmatch can raise re.error for invalid patterns, while the command loop suppresses several other exception types but not re.error. In this environment re.error inherits directly from Exception. A malformed pattern therefore appears capable of escaping the ordinary command handler. Test on a fake experiment first; no live crash was attempted.
- Sensor age begins at application time. A sensor command delayed in the ordinary inbox could be made to appear fresh when it is eventually applied. The actual impact depends on queue delay, which has not been measured.
- A connectome cache with the same neuron count but changed source files or sign convention could pass the current compatibility check. There is no evidence that the present local cache is stale.
- The synaptic trainer stores compiled-kernel mode but recalibrates comparisons on device changes, not necessarily a same-device kernel-mode change. Existing parity checks substantially reduce the concern, but kernel identity belongs in the resume contract.
- The live server usually preserves continuous neural state through game respawns, while offline evaluators reset at each new game. Neither convention is intrinsically wrong; report it and test whether it affects scores before comparing those settings directly.
- The direct-board test's food-only ablation removes both occupancy and body-order planes from the trained model. It does not isolate body order alone or answer whether a separately trained food-only model would perform better.

## Ideas to defer unless measurement creates a reason

Do not start with sparse-kernel rewrites, lower numerical precision, larger time steps, asynchronous changes to neural state, or a large increase in generation count. They can alter the experiment or consume substantial time without resolving the suspected information-to-motor limitation. Modest independent training replications can still be useful alongside a small diagnostic panel. In particular, parallelizing outgoing synapse scatter with a naive prange is unsafe because several presynaptic neurons can add to the same target. Numba documents that concurrent writes into shared array elements can race. Prefer independent brains/candidates first, or design an explicitly race-free reduction and verify it. [Numba parallel-loop documentation](https://numba.readthedocs.io/en/stable/user/parallel.html).

The most useful first sprint combines ideas 1, 2, and 3: locate the steering bottleneck, make every artifact auditable, and learn what actually costs time. Idea 4 makes the live demo dependable. Idea 5 then turns the resulting changes into a meaningful experiment.
