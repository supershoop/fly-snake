# Submission text (paste-ready)

Fill in the bracketed parts. Everything else is measured and recorded in `AGENTS.md`.

**Project name:** Fly Snake

**Tagline (under 140 characters):**
A real fruit fly's complete brain wiring, simulated and untrained, plays Snake. Silence two cells and watch what it loses.

**Links:** repository https://github.com/supershoop/fly-snake · video [add link] · team [names]

## Inspiration

In 2024 and 2025 the complete wiring diagrams of adult fruit fly brains were published: every neuron, every connection. We wanted to know how much behaviour is already sitting in that wiring, and we wanted a way to show it that a room of people can watch and poke at. Snake needs two things a fly already does: go toward something, and get out of the way.

## What it does

Fly Snake simulates all 165,000 neurons and 6 million connections of the MaleCNS fly connectome and lets it play Snake in the browser, with the brain lighting up next to the game.

- The game reaches the fly through real sensory cell types. Food drives its small-object detectors, walls and its own body drive its looming detectors.
- The move comes from its real output neurons. In **Normal** mode nothing is trained: the snake turns toward the side where the fly's steering neuron fires more, and the fly's escape neuron overrides that when a threat is on that side.
- **Lesion lab.** Pick a fly and silence a cell type. Without its 2 steering cells it stops finding food but still dodges walls. Without its 2 escape cells it still goes for food but crashes 2.5 times sooner. Silencing 2,000 random cells changes nothing.
- **Scrambled control.** The same neurons and connection strengths with random targets score zero.
- **Pathways.** The brain view draws the circuits carrying each signal. The food pathway was found by searching the wiring, and it matches the pursuit pathway known from real flies.
- **Training.** A readout of under 4,000 numbers learns the game from reward while the brain stays fixed, and the audience can reward or punish moves from their phones by QR code. On the real wiring it reaches 22 food on average over ten runs; on scrambled wiring it stalls at 5.
- **Versus.** Race the fly for food.
- When the snake eats, the fly's sugar-taste neurons fire and its feeding motor neuron responds. When it dies, its heat sensors fire and the punishment dopamine neurons respond.

## How we built it

- **Brain:** a batched leaky integrate-and-fire simulation in PyTorch after Shiu et al. 2024, built from the MaleCNS v1.0 flat connectome files: connections with at least 5 synapses, excitatory or inhibitory sign from each neuron's predicted transmitter. Sixteen independent brains run in one sparse matrix product per time step on a laptop GPU.
- **Finding the circuits:** probe experiments stimulated each candidate sensory group and measured which output neurons responded on which side. A path search through the wiring found the relay cells between the object detectors and the steering neuron.
- **Server:** FastAPI and WebSockets stream the game and the brain's activity once per move, with per-fly lesions, live learning, audience feedback and a thermal guard.
- **Page:** React, TypeScript and Three.js on the open fly-connectome-template viewer, which draws 124,000 measured cell-body positions.
- **Evidence scripts** for every number we quote: untrained control, lesion table, scrambled control, learning speed.

## Challenges

- The published neuron model was tuned on a different connectome. On this one every input set off runaway activity across 20,000 neurons, because this dataset counts about twice as many synapses per connection. One global factor of 0.4 brought it back to sparse, stimulus-specific activity, and it then reproduced a published result (sugar-taste cells drive the feeding motor neuron, bitter cells do not).
- Our first control was wrong. A trained readout plays nearly as well through a scrambled brain, so "trained and it works" says nothing about the wiring. We rebuilt the claims around the mode where nothing is trained.
- The untrained fly kept turning into its own body. The diagnosis: its pursuit signal always won and the rule never looked at its escape neuron, which was firing. Letting escape override pursuit doubled its survival and produced the two different lesion deficits.
- Two ideas failed and we kept the results: painting an image onto the fly's eye does not propagate through this kind of model, and the fly's learning centre does not reach its steering neurons here, so learning inside the brain was not possible.
- The laptop overheated and crashed mid-hackathon. The server now idles when no page is visible and slows itself when the GPU runs hot.

## What we learned

The wiring alone already contains recognisable behaviour: pursuit and escape circuits that match known fly biology appear without being programmed. We also learned to distrust our own good results, and that a fair control is worth more than a high score.

## What is honest to say

Everything on screen is simulated activity, never recorded from a fly. The model is heavily simplified. We chose how game events become sensory input, one global scaling factor and three thresholds in the untrained rule. In Training only the small readout learns; the brain never changes.

## What's next

Better senses rather than a cleverer rule, starting from the connectome-derived retinal encoder we built; more scrambled networks and live runs behind the learning result; and the nerve cord's motor neurons, which are already simulated, driving the fly's legs.

## Built with

Python, PyTorch, FastAPI, WebSockets, React, TypeScript, Three.js, Vite. Data: MaleCNS v1.0 connectome (FlyEM at HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology, Google Research; CC BY 4.0). Viewer: fly-connectome-template by Mert Cobanov. Body mesh: Flybody (Apache-2.0).
