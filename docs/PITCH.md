# Pitch notes and judge questions

## The one-sentence version

We took the complete wiring diagram of a fruit fly's brain, changed none of it, and found that its own circuits chase food and flee threats in a game of Snake, and that you can break that behaviour one cell at a time.

## Three minutes

1. **What it is (30 s).** The MaleCNS connectome: 165,000 neurons and 6 million connections from an electron microscope. We simulate every neuron as a simple spiking unit. The game is shown to the fly through real sensory cell types: food drives its small-object detectors (LC10), walls and its own body drive its looming detectors (LC4, LPLC2). We read its real output neurons, the descending neurons that run to the body.
2. **Nothing is trained (45 s).** In Normal mode the snake turns toward the side where the fly's steering neuron (DNa02) fires more, and the escape neuron (giant fiber, DNp01) overrides that when a threat is on that side. Those are the jobs those cells have in a real fly. Show it playing.
3. **The wiring matters (60 s).** Lesions: silence the 2 steering cells and it stops finding food but still dodges; silence the 2 escape cells and it still seeks food but crashes 2.5 times sooner; silence 2,000 random cells and nothing changes. Scramble the wiring, keeping every neuron and connection strength, and it scores zero in three separate scrambles.
4. **It predicted known biology (20 s).** A search through the wiring found the food signal crosses a dozen relay cells in the anterior optic tubercle. That is the pursuit pathway already known from real flies. We did not put it there.
5. **Learning (25 s).** A readout of under 4,000 numbers can learn the game from reward on the real wiring (22 food on average over ten runs) and cannot on scrambled wiring (5). The audience can reward and punish moves from their phones.
6. **Close.** Snake is the measuring stick. The brain is the thing being measured.

## Numbers you can defend

| Claim | Number | Where it comes from |
|---|---|---|
| Real wiring steers, scrambled cannot (nothing trained) | 2.28 food against 0.00, 0.00, 0.09 | `scripts/untrained_control.py`, 32 games, three scrambles |
| Steering cells are necessary for food | intact 5.9 ± 1.0 food; DNa02 silenced 1.1 ± 0.3, yet 170 moves survived | `scripts/lesion_scores.py --games 16`, live brain |
| Escape cells are necessary for survival | 63 moves intact; 25 with DNp01 silenced | same |
| Relay cells found by path search matter | 12 AOTU cells silenced: 0.8 ± 0.2 food | same |
| Random damage does nothing | 12 random cells 5.5; 2,000 random cells 6.4 | same |
| Real wiring is learnable, scrambled is not | 22.0 ± 2.7 against 5.1 ± 0.8, ten runs each | `scripts/live_learning_test.py --seeds 10`, offline on recorded brain responses |
| Reproduces a published result on a different brain | sugar-taste cells drive the feeding motor neuron; bitter cells do not | `scripts/event_probe.py`; Shiu et al. 2024 showed it on the female FlyWire brain |

## Questions judges will ask

**How is this better than training an algorithm to play Snake?**
It is not better at Snake. A rule of six lines scores 22; our untrained fly scores about 6. In a trained model every weight was tuned for the game, so playing well tells you the optimiser works. These 6 million weights were never tuned for anything. Whatever the fly gets right is information about a real animal's wiring, and we can test it by lesion and by scrambling.

**Is the readout not doing the playing?**
In Trained mode, mostly yes, and we say so: a trained readout also plays through a scrambled brain (14.3 against 18.8). That is why the claims rest on Normal mode, where nothing is trained and the rule reads only the fly's steering and escape neurons. Scrambled wiring scores zero there.

**What did you choose yourselves?**
How game events become sensory stimulation; one global synapse-strength factor (0.4, because this dataset counts about twice as many synapses per connection as the one the published model was tuned on); and three firing-rate thresholds in the Normal rule. The neurons, connections and signs come from the data.

**Is this what a real fly would do?**
It is a prediction from a very simplified model: every neuron is the same kind of spiking unit, there are no neuromodulators, no body and no learning inside the brain. What it shows is how much behaviour the wiring alone already contains.

**Does the fly learn?**
No. We checked: our stimuli never reach the fly's learning centre (0 of 4,064 Kenyon cells respond), and its output does not reach the steering neurons in this model. Learning happens in the small readout only.

**Does it feel anything when it eats or dies?**
When it eats we stimulate its sugar-taste neurons and its feeding motor neuron fires at about 79 Hz. When it dies we stimulate its heat sensors: about 6,700 neurons fire and the punishment dopamine neurons reach 83 Hz. The reward dopamine neurons do not respond to sugar in this model, so we say "tastes", not "feels rewarded". All of it is simulated activity.

**Why the male brain?**
The anatomy viewer we built on uses the MaleCNS dataset, which also includes the nerve cord. It also let us check that a result published on the female brain holds on a second, independent connectome.

**What would you do next?**
Better senses, not a cleverer rule: the fly's remaining deaths come from threats straight ahead, where both escape neurons saturate. We built a connectome-derived retinal encoder as a first step. And more scrambled networks and live runs behind the learning result.

## Credits to say out loud

MaleCNS connectome: FlyEM at HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology, Google Research (CC BY 4.0). Neuron model after Shiu et al. 2024. Viewer built on fly-connectome-template by Mert Cobanov. Body mesh: Flybody.
