# Nine distinctive Fly Snake experiences

Design sketches, 20 September 2026. These are proposed experiences, not implemented features or new experimental results. The companion `brainstorm-experience-2026-09-20.md` contains the repository audit and a more conventional implementation roadmap.

The common premise: **give the visitor a prediction to make before giving them a button to press.** The emotional payoff should come from an intervention with a traceable consequence, including an honest surprise or null result.

## 1. The circuit escape room

**Premise:** “Open three locks without changing the decoder.” The locks are experimental objectives: reduce food pursuit while preserving threat response; reduce threat response while preserving pursuit; restore both. Visitors receive a limited set of annotated circuit cards and must choose an intervention and a control.

This is a puzzle about understanding the model, not a promise that a fly can solve a human-authored maze. A successful lock is a predeclared response pattern in a short assay, confirmed by a recovery condition. It should not open because the presenter secretly substituted a better controller.

**Minimum prototype:** Three short recorded, accurately labeled assays plus an interactive prediction step. The live version reuses the lesion machinery and fixed decoder, measures neural responses on a declared stimulus schedule, then shows a short Snake illustration. The first card set is AOTU food relays, DNp01, a size-matched random group, and restore; output-cell cards can be introduced after upstream cards so the puzzle is more than disconnecting the decoder's inputs.

For a two-behavior Snake challenge, use a declared, frozen `InstinctPolicy` with its hand-set pursuit/veto/dodge rules. `HardwiredPolicy` does not read DNp01, so it is not interchangeable with that decoder for an escape-behavior demonstration. The neural assay can measure the giant-fiber response independently of either behavioral decoder.

**What could fail:** A lock criterion may be unreliable across neural noise or may be visible only under one decoder. Pretest the assay, specify the decoder, and let a lock return “inconclusive.” If visitors only memorize card names, the puzzle has failed educationally; ask them to predict the effect on the second behavior before opening the lock.

**Goal / effort:** Demonstration and causal learning; 1–2 days for the recorded prototype, several days for a reliable live assay runner.

## 2. Bet a prediction, not a snake score

**Premise:** Visitors have three hypothesis tokens. Before a lesion, they allocate them to “less food,” “more collisions,” “both,” or “little change.” After a fixed trial they see the result and explain the best next control. Reward calibration and useful controls, rather than only guessing the winning outcome.

The memorable twist is that removing pursuit can increase time alive while reducing food. Another is that a trained readout can retain behavior after a lesion that disrupts the nothing-trained controller. This is a public lesson in choosing a metric and specifying the system being tested.

**Minimum prototype:** An offline classroom card game using the repository's actual saved results, with sample size, decoder, and uncertainty on the reveal card. Add live play only after paired trials exist. A QR prediction page can operate independently of the current feedback endpoint and must say it does not train the model.

**What could fail:** A few live games may contradict historical averages. That is part of the reveal, not a scoring error; do not mark reasonable uncertain predictions “wrong” solely because one snake crashed. If necessary, score the visitor's experimental design and interpretation instead of prediction accuracy.

**Goal / effort:** Public engagement and scientific reasoning; half a day for a physical/HTML prototype.

## 3. Build a board the fly cannot distinguish

**Premise:** The visitor wins by constructing two boards that look different to a human but produce the same current five-channel input. A “sensor fingerprint” confirms equality. They then propose the smallest additional observation that would distinguish the pair.

This turns the fair criticism of 24-pattern sensing into the project's most original teaching interaction. It also gives a concrete reason to explore richer input while refusing to assume that richer input will improve control.

**Minimum prototype:** Six validated board pairs exported from the canonical Python encoder, a guess/reveal interaction, and a fingerprint comparison. Later add a board editor. One set can demonstrate that tail-route checking is engineered preprocessing that goes beyond adjacent-cell collision detection.

**What could fail:** Different continuous histories can produce different brain outputs for equal current inputs. Say “same input this move”; only show “same output from the same state” after copying/resetting simulator state and matching random state. Do not assert different optimal moves unless the horizon and objective are actually evaluated. The puzzle works without any optimality claim.

**Goal / effort:** Understanding sensing and state; 1 day for validated fixed examples, 2–4 days for a usable editor.

## 4. One vote, two possible responses

**Premise:** After a visitor rewards a displayed move, show two action distributions for that same saved neural response: the current readout immediately before their feedback update, and immediately after it. “Your vote increased this response's left-turn preference from X to Y.” Their contribution is visible even when the next live snake move happens to look unchanged.

The display then returns to live play, making the distinction between a parameter update and a behavioral outcome tangible. It is a much more convincing receipt than a generic counter of positive clicks.

**Minimum prototype:** In `HumanFeedback.apply`, evaluate the current readout on the saved feature vector immediately before and after the single authorized update, then add that pair to the existing receipt. Use `OnlineLearner.last`'s actual transformed features. This avoids cloning an entire brain to compare one fixed input. Display the actual update magnitude rather than guaranteeing an impressive visual shift.

For one targeted fly, the existing update also gives an exact check: the same-input logit change is `rate × feedback × (one_hot(saved_action) − saved_probabilities) × (sum(saved_features²) + 1)`. The final `+1` is the bias update. A request affecting several flies needs the full summed update instead. This is a readout calculation on a saved neural response, not an alternative continuous-brain rollout.

**What could fail:** Comparing the original move's probabilities with today's probabilities confounds the visitor's vote with all intervening automatic and audience updates. The comparison must bracket this one update. Probability movement is not proof that the chosen action will change or the score will improve. Do not animate hypothetical full trajectories as if they were simulated.

**Goal / effort:** Tangible learning and transparent agency; 1–2 days. This may be the best small addition to the existing phone experience.

## 5. The driver lineup

**Premise:** “Which of these is real wiring with nothing trained? Which has a trained readout? Which is scrambled?” Visitors see short behavior clips with intentionally hidden condition labels, commit a guess, and reveal the full condition cards and scientific limitations.

The point is not a test for consciousness or proof of intelligence. It is that visible behavior alone often underdetermines the mechanism. A trained scrambled network can play; a decoder can hide circuit differences; performance is not a complete account of what the experiment measures.

**Minimum prototype:** Complete, preselected recorded episodes from matched seed sets, visibly marked as recordings. Use readouts trained for the corresponding wiring; never put the real-trained readout on scrambled wiring and call its failure evidence. At the start, tell participants that labels are hidden for a guessing game and will be revealed.

**What could fail:** Unequal clip lengths, cherry-picked episodes, or color/latency cues make the answer trivial. Choose inclusion rules before viewing clips, normalize the presentation, and reveal all included results after the guess. Some guesses should remain difficult; do not manufacture separability.

**Goal / effort:** Mechanistic skepticism and a memorable audience hook; 1–2 days if suitable recordings exist.

## 6. One circuit, three worlds

**Premise:** Keep the same connectome and fixed decoder while moving between Snake, chasing a target dot, and avoiding a looming event. Each world displays its explicit mapping into the same sensory/motor contract. Visitors ask which responses transfer and where the abstraction breaks.

This makes the project about reusable pursuit/escape circuitry rather than a single game leaderboard. It also makes a useful limitation visible: a successful response may come from the engineered observation and decoder as much as the world's appearance.

**Minimum prototype:** Add a tiny two-dimensional pursuit task and a left/right escape assay. Reuse the existing LC10/LC4/LPLC2 stimulus channels, named DN outputs, simulator, and one frozen decoder. Start with a simple drawn agent; do not imply a physically simulated fly body. Predeclare task metrics and avoid tuning the decoder separately for each world while advertising it as unchanged.

**What could fail:** If every world already encodes the answer as “food left,” transfer may be unsurprising. Make this limitation explicit and compare a direct channel rule under the same inputs. A naturalistic visual version is a later research project: anatomical retinotopy is a starting point, not evidence that the present LIF simulator reproduces visual processing.

**Goal / effort:** A distinctive product identity plus broader controls; 2–4 days for simple worlds, much longer for credible naturalistic sensing.

## 7. The room becomes the stimulus

**Premise:** Two physical zones or large touch pads represent left/right engineered sensory drive. Visitors collectively create a pursuit/looming scene and hear or see the descending-neuron response. Then the host applies a circuit lesion and the room discovers which part of the response disappeared.

This is an instrument, not a crowd remotely choosing the snake's action. The visible mapping runs through the whole simulator. The experience can be performed with a single projected browser before adding a sensor or LEDs.

**Minimum prototype:** Host-controlled left/right input pads and rate meters, using a single explicit stimulus aggregator. For hardware, start with the already planned distance sensor after checking its actual model and voltage needs. A sound layer maps recorded rates to tones; it must not claim authentic spike timing from 100 ms counts.

**What could fail:** Simultaneous users may saturate/clamp both channels and remove useful differences. Calibrate the amplitude range and show the aggregate input. The current `sensor` command overwrites the external-input dictionary, so multiple clients need a deliberate aggregator, not a last-writer contest. Keep new audience permissions narrow. A giant-fiber response is not a guarantee that the snake dodges safely.

**Goal / effort:** Physical immediacy and social play; 1–2 days for browser pads, additional time for hardware.

## 8. The fly on trial

**Premise:** Put one carefully worded claim on trial: “The real wiring contributes to untrained behavior.” The audience requests evidence cards: scramble wiring, lesion an upstream path, lesion random cells, or train a decoder. They vote before the reveal and amend the claim afterward.

The surprise is that some attractive evidence is weak. Good trained performance alone does not establish the specific wiring's necessity. A decoder failing on a network it was not trained for is a bad control. A null lesion result in trained mode is not automatically evidence that the pathway has no role.

**Minimum prototype:** A five-minute facilitator script and four actual evidence cards generated from existing measured simulation results. Each card includes condition, N, outcome, uncertainty or caveat, and a reproduction pointer. Let the audience choose which card to request first.

**What could fail:** It can become an argument staged to reach a predetermined verdict. Publish the claim and standards at the start, include inconvenient findings, and allow the verdict “supported only under these conditions.” Avoid the more ambitious untested claim that the whole connectome is necessary for the score.

**Goal / effort:** Research communication and a strong pitch ending; half a day to 1 day.

## 9. Coach versus coincidence

**Premise:** Two independent readouts receive equal amounts of audience feedback. One receives feedback associated with the intended displayed decisions; the other receives a declared shuffled or delayed-association control. After training, both are frozen and evaluated. Does meaningful coaching outperform the same amount of nonspecific feedback?

Participants should be told upfront that the experiment includes a control condition, with the exact arm labels revealed afterward. This is an experiment about the feedback mechanism, not a trick that credits arbitrary improvement to the audience.

**Minimum prototype:** First run a small offline rehearsal using existing decision records and two copied readouts. Then use separate live cohorts or matched sessions with equal update budgets, automatic rewards, initial weights, and evaluation seeds. Predefine the shuffled-feedback procedure; compare against automatic-only learning as well where feasible.

**What could fail:** The existing 16 flies share one readout, so using half as controls leaks updates immediately. Selection, delayed credit, different noise, reward dose, and a small sample can swamp the effect. Human feedback may be too sparse or inconsistent to help. If the experiment is inconclusive, report that and retain the interaction as a teaching demo rather than a performance claim.

**Goal / effort:** A substantive learning result with an excellent audience role; several days for a fair design and implementation, plus actual evaluation time. It is a research stretch, not the first demo fix.

## Which to prototype first

1. **One vote, two possible responses** has the best combination of novelty, immediate agency, and existing implementation support. It improves an already working feature and can be checked exactly.
2. **Build a board the fly cannot distinguish** is the strongest educational identity. It makes the project's limits inspectable and gives future sensing work a concrete purpose.
3. **The circuit escape room** is the most distinctive guided exhibit. It should begin with a small, reproducible response assay rather than an elaborate new game.

The courtroom and hypothesis tokens can be prototyped with writing and existing artifacts before any new server feature. The three-world concept is the strongest longer-term expansion if the project wants to outgrow Snake without hiding its sensory and decoder assumptions.

## Adversarial implementation review: feedback impact

The small version of concept 4 is feasible without new neuroscience, a full experiment runner, or a second brain. Keep its scope precise:

1. Retain the existing eligibility checks, saved decision ownership, and request-specific receipt. Start with one targeted fly, which is what the phone UI already sends.
2. On the simulation worker, immediately before this request calls `learn`, evaluate the **current** weight/bias on the saved transformed feature row. Compute `x @ weight.T + bias`; `x` has already received `log1p`, so calling `Policy.logits(x)` would apply the transform twice and be wrong.
3. Apply the existing update exactly once. Evaluate that same saved row immediately afterward. Store the two three-action vectors, saved action, fly, move, and phase identity in the request's receipt. Do not use the next live frame as the “after” measurement because it has different input and may include other updates.
4. Label the result “Effect of this feedback on the current readout, evaluated on move 418's saved neural response.” It is not a rewrite of the historical decision. The old move's original probabilities can be displayed separately, but they are not the causal baseline for this feedback.
5. Keep private receipts associated with their request IDs. The host's global `feedback.last` can be overwritten by another visitor; it is not enough for a personal impact display. Positive/negative counters count applied requests rather than unique votes or participants. These IDs correlate receipts but currently do not deduplicate retries; do not silently retry an uncertain submission without adding an explicit idempotency mechanism.
6. The 64-decision history expires by moves, not wall-clock seconds: approximately 10.7 seconds at six moves per second, or 32 seconds at two. A pinned archival view can remain readable indefinitely, but its reward buttons must expire. Freeze the pin's fly identity too, so a host changing the selected fly does not retarget an old move.
7. Preserve metadata from the decision itself. Lesions or external input can change while a past decision remains eligible; decorating its snapshot with current settings would be misleading. Copy pre-move food/body lists rather than retaining mutable references from `render_state()`.
8. Add focused contract checks for intervening automatic updates, two visitors, expired/model-replaced decisions, and all-fly feedback. The simple single-target logit formula does not describe several simultaneous targets; evaluate the actual summed update for that scope. Keep the receipt small and measure throughput under a burst before extending the feature to every fly in every frame.

This is useful even if the probability shift is tiny. Do not enlarge the apparent change with an unlabeled moving scale. It does not claim that coaching helps generalization; the separate coaching/control experiment remains necessary for that claim.

## A real five-minute visitor session

Choose **one** of these sessions. Combining lesions, internal plasticity, live readout training, hardware, and a new world in five minutes overloads the explanation. A usable first version can mix a clearly labeled live illustration with an explicitly labeled recorded controlled trial.

### Session A: predict, perturb, restore

| Time | What the visitor does | What the presenter/display establishes |
|---|---|---|
| 0:00–0:35 | Watches one snake and identifies the food. | “Measured wiring; simulated activity. This game is converted into five engineered sensory channels. We use one declared, fixed movement decoder.” Identify the actual decoder by name and keep it unchanged within the comparison. |
| 0:35–1:15 | Pins one completed decision and follows the input → named rates → actual rule → outcome. | The board before the action is distinct from the resulting board. The organism panel is animation. |
| 1:15–1:45 | Predicts what silencing upstream food-relay cells will do to food collection and survival. | Record the prediction before the intervention. Explain the random-cell control with the same cell count. |
| 1:45–2:45 | Applies the lesion to the chosen live fly, then restores it. | This is an illustration. A short live segment may not reveal the average effect; show the actual response without waiting indefinitely for a satisfying result. |
| 2:45–4:05 | Inspects the predeclared paired trial and its control, live if complete or clearly labeled recorded evidence. | Food and survival are separate; sample size, decoder, seed set, and cap are visible. Recovery and null results remain visible. |
| 4:05–4:40 | Suggests the next control or chooses between two prewritten hypotheses. | Anatomical pathways, hand-set decoding, and engineered sensing each have a role; the experiment does not establish a living fly understands Snake. |
| 4:40–5:00 | Takes an evidence card and describes what changed. | The card carries the condition, source label, observed outcome, and reproduction link, rather than only a screenshot of a successful snake. |

**Presenter contingency:** If the network or live simulation stalls, explicitly switch the session to recorded mode and disable ineffective controls. If the selected lesion does not produce a clear change during its allotted minute, say so and proceed to the accumulated trial. Do not change the checkpoint to manufacture a reveal.

**Match the source of the evidence:** The existing AOTU lesion-score script uses `HardwiredPolicy` for its nothing-trained condition. A first session can use that same decoder and focus on pursuit. If the story includes the pursuit/giant-fiber double dissociation, use `InstinctPolicy` and obtain/display matching evidence for that condition; do not relabel the hardwired results as instinct results. The current Normal button selects instinct, while a protocol-selected hardwired policy is also presented as Normal, so the new experiment view needs an explicit decoder label.

### Session B: see exactly what your feedback changes

| Time | What the visitor does | What the presenter/display establishes |
|---|---|---|
| 0:00–0:40 | Watches a learning fly and scans the phone link. | “The connectome stays fixed. Your feedback updates the shared external movement readout; automatic game rewards also train it.” Use an explicitly named initial learner, optionally a copy of the trained readout. |
| 0:40–1:30 | Studies a captured example with a pre-move board, chosen move, and outcome. | This is a read-only teaching example; it need not still be eligible for feedback. Explain positive/negative credit and the difference between action probability and good-action probability. |
| 1:30–2:10 | Picks a **fresh** eligible decision and sends one reward or punishment promptly. | Pin move and fly together. Show a real receipt or a clear expiry rejection; never silently retarget to a newer move. |
| 2:10–3:10 | Compares the same saved neural response immediately before and after that request's update. | This isolates this one feedback update. Other visitors and intervening automatic training are not credited to it. No second brain or fictitious alternative trajectory is shown. |
| 3:10–4:20 | Watches subsequent play and predicts whether a changed probability guarantees a changed action or higher score. | The answer is no. A later action depends on new input, ongoing updates, and sampling. Positive local credit is not proof of improved unseen-game performance. |
| 4:20–5:00 | Chooses the appropriate next experiment: freeze and test on new boards, or merely watch for a lucky high score. | Show the planned or actual held-out comparison, clearly distinguishing them. Export the feedback receipt if available. |

The second session can work with a single phone and one brain. It does not need 16 visitors, a long training curve, a sham cohort, or an improvement claim. Real coaching-versus-control evaluation is a separate research session.

## Practical cuts and priority corrections

- The decoder microscope, failure replay, and portable postcard share the same event record. Treat them as one foundational capture/inspection feature with three views, not three separate backend projects.
- Prediction tokens, the courtroom, the escape room, and the classroom pack are alternative authored experiences over the same trial/evidence data. Prototype one audience story first.
- The full coached-versus-sham experiment is several days of experimental infrastructure. The exact one-feedback impact receipt is the smaller first increment.
- Use the **frozen instinct rule** when demonstrating the pursuit/giant-fiber behavioral dissociation. A fixed `HardwiredPolicy` is a different decoder that does not use DNp01.
- A true “freeze the current learner” operation must snapshot its weights/bias. Switching to today's Trained button selects the saved trained model instead.
- Name the direct-board checkpoint when discussing sensory/motor gain transplantation; it is separate from the web app's five-channel synaptic checkpoint.
- A realistic first two-day experience deliverable is a correct decision record/inspector plus one authored experience using recorded controlled evidence. A live matched-trial runner, full replay editor, independent coaching cohorts, new worlds, and hardware do not all fit that budget.
