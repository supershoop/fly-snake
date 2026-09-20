# A bounded first-afternoon experiment

This is a proposed protocol, not an experiment run during brainstorming. Six geometric board fixtures were checked without running a brain: unique connected body cells, food outside the body, and the intended immediate blocked directions.

**Question:** do the saved learned sensory-pathway gains add obstacle-sensitive behavior beyond adjusting motor excitability, with the existing fixed decoder?

## Frozen inputs and five conditions

Use models/brain-board-direct.npz, the current connectome data and validated fingerprint, the checkpoint's encoder mapping, and the current descending-neuron body-ID order. Verify that the checkpoint matches size 12, food sigma 3, obstacle sigma 0.8, gain 1, 100 ms windows, 0.5 ms steps, the standard weight scale, and the fixed strict +/-2-spike decoder. Use one compiled CPU runner. Do not train or tune the decoder.

The saved labels start with sensory: or motor:. A zero log-gain means original strength. Construct five conditions from the existing saved parameter vector:

| Condition | Sensory-group gains | Motor-group gains | Input |
|---|---|---|---|
| O | Original | Original | Full board |
| S | Saved trained | Original | Full board |
| M | Original | Saved trained | Full board |
| SM | Saved trained | Saved trained | Full board |
| SM-F | Saved trained | Saved trained | Food only |

For food-only, retain the entire stimulus index array and shape and zero the occupancy/body-order levels. This preserves the matching of random draws for common input neurons. All variants retain connectivity, signs, gain bounds, encoder identity, and the original movement decoder. Never save derived variants over an existing model.

## Fixtures and window budget

All boards have size 12, head (6,6), heading up, body length 7, idle 0, score 4, and one food. The three body shapes are:

- Clear: [(6,6), (6,7), (6,8), (6,9), (5,9), (5,8), (4,8)].
- Left block: [(6,6), (6,7), (5,7), (5,6), (4,6), (4,7), (3,7)].
- Right block: [(6,6), (6,7), (7,7), (7,6), (8,6), (8,7), (9,7)].

Cross each shape with food at (3,4) or (9,4), producing six fixtures. The checked immediate blocked directions are respectively [false,false,false], [true,false,false], and [false,false,true]. Food side is left or right as intended. These are controlled initial-state fixtures, not a sample of natural on-policy game states.

**Budget: 1,920 simulation windows**, calculated as 5 conditions x 6 fixtures x 16 neural-noise seeds x 4 continuous windows. This is neither an episode budget nor a promised runtime.

For each trial, reset the brain and its RNG, apply the condition's gains, and present three 100 ms windows of the clear fixture with the same food side. Without resetting neural state, present the target fixture for one final 100 ms window. Do not advance the game during these windows. Record the last window's response and action. Repeating the clear warm-up avoids building general state-snapshot infrastructure for this first assay.

The reset states are identical, and conditions receive the same sequence of board fixtures and matching random draws. Post-warm-up neural states are not expected to be identical across different weight conditions: their divergence is part of the intervention. SM-F deliberately omits occupancy/body stimulation throughout its warm-up and target window, so its board history matches while its injected sensory history is ablated. Within any one condition and matched noise seed, the obstacle variants share the same clear-board warm-up before their target inputs differ.

Use the same neural-noise seed for corresponding trials across conditions and target fixtures. Proposed diagnostic seeds are 90,620,000 through 90,620,015; a text search found no occurrences in current models/docs/scripts, but check the run ledger before reserving them. Put the exact reserved list in the manifest before running.

## Metrics, comparisons, and a decision

Record blocked-side turn frequency, food-side turn frequency on the clear boards, steering differences and distances to the fixed decoder thresholds, four motor-neuron counts, and all descending-neuron counts. Compare obstacle effects within each food-side/seed pair. Report both left and right results before pooling them. A broad descending response difference is not by itself useful fixed-decoder control.

Predeclare at most six contrasts: SM minus O; M minus O; S minus O; SM minus M; SM minus SM-F; and the sensory/motor interaction SM minus M minus S plus O. Keep the first afternoon descriptive rather than claiming generalization from six selected boards.

A proposed proceed gate is a reduction of at least 20 percentage points in blocked-food-side turns versus both SM-F and M, with no more than a 10-point loss in food-side turning on clear boards, and the direction consistent for both food sides. This gate is a practical proposal to freeze before running, not a statistical significance threshold or a finding. If the fixed fixtures produce a floor/ceiling effect that makes the question uninformative, report the assay failure and version a new diagnostic set; do not treat redesign as an untouched test.

If learned sensory groups add no obstacle-dependent motor effect beyond M, or changes appear only in broad DNs, prioritize input-dose, representation, gain-grouping, and anatomical-routing diagnostics before committing to a much larger training run. Six fixtures cannot establish that the search space lacks useful solutions: different histories, coordinated gain changes, or an undertrained checkpoint could escape this assay. A modest independent fresh training replicate may remain worthwhile if its cost is small and its question is predeclared. A positive result justifies an independent whole-game experiment; it does not establish a Snake score improvement or general body planning.

## Exact outputs and protection of previous tests

Write only to a new outputs/diagnostic-steering-UNIQUE directory:

- manifest.json: input file hashes, anatomy/encoder/decoder identities, source commit and dirty-tree fingerprint, library/device/kernel versions, all fixtures, seeds, conditions, window budget, metrics, contrasts, and proceed gate.
- parameters.npz: the five frozen parameter variants and their labels, derived from the saved checkpoint.
- responses.npz: encoded input levels, descending and motor counts, margins, and actions, with body-ID order.
- trials.jsonl: one completed target-window record per trial, with condition, fixture, seed, completion status, and timing.
- report.md and plots: paired effects, distributions, the declared gate, negative results, and the fixture-only scope.

Hash the saved input model before and after. Leave models, existing training-run state, winner files, and test.json locks unchanged.

Existing saved test seeds may be reanalyzed or replayed for explicitly labelled retrospective/post-hoc diagnostics. They are already exposed and cannot become a new independent test of a gain transplant selected after viewing them. Prefer the new diagnostic fixtures/seeds here. Once their results guide design, they are development data too. A future generalization claim must freeze its final variant, predeclare a new untouched game/noise seed set, and write a new evaluation manifest. Never remove an old test lock or resume training in a tested run. A future resumable frozen test may only complete missing records for the same immutable model/configuration/seed list.

## Three engineering projects to defer

1. Custom GPU kernels, lower precision, larger time steps, and within-brain parallel scatter. The compiled CPU path exists; measure end-to-end costs before changing computation or numerical behavior.
2. Full session event sourcing and general state-fork infrastructure. Explicit reset plus a fixed warm-up answers this first question with less implementation risk. Add snapshots when a concrete experiment requires them.
3. Dynamic worker scheduling or distributed experiment orchestration. One serial runner is sufficient for this bounded protocol. Keep per-trial logs and atomic final writes, then add concurrency only when measured throughput justifies it.
