# Demo run sheet

For whoever drives the laptop. About four minutes. Every number quoted here is measured and recorded in `AGENTS.md`.

## Before anyone is watching (10 minutes)

1. Laptop plugged in, on a hard surface, lid open. Close the browser tabs and chat apps you do not need. A brain process wants 3-4 GB of RAM.
2. Two terminals in the project folder:
   ```sh
   .venv\Scripts\python -m uvicorn flybrain.server:app --port 8000 --host 0.0.0.0
   npm run dev
   ```
   The first start takes about a minute while the connectome loads. Open the address `npm run dev` prints.
3. Wait for the status pill to say **Live simulation** and check the status line shows a GPU temperature.
4. Click through every step below once. If anything looks wrong, hard-refresh (Ctrl+Shift+R) before debugging.
5. Leave the page on **One fly** and **Normal**. That is the opening shot.

Heat: one fly keeps the GPU around 45 °C. **Swarm** is the hot layout. If the status line passes 80 °C the simulation starts leaving gaps between moves; at 87 °C it holds and shows a banner. To make sure it never holds during judging, start the server with `FLY_THERMAL_GUARD=slow` (PowerShell: `$env:FLY_THERMAL_GUARD = 'slow'` before the uvicorn line).

The simulation idles when the tab is hidden. Switching to slides in another window is fine. Switching to another **tab in the same browser window** pauses it; it resumes the moment you come back.

## The run

| # | Click | What the audience sees | Say |
|---|---|---|---|
| 1 | Nothing. **One fly**, **Normal** already on. | A snake finding food. The brain lights up with each move. | "This is the complete wiring diagram of a fruit fly's brain: 165,000 neurons, 6 million connections, read off an electron microscope. We changed none of it and trained none of it. The game shows the fly food as a small object and walls as something looming at it. Its own steering and escape neurons play." |
| 2 | In the brain panel, pathway selector: **Food pursuit**, then **Escape**. | The brain dims; one coloured circuit stays lit and pulses with the game. | "We did not draw these. A search through the wiring found that the object detectors reach the steering neuron only through about a dozen relay cells. That is the pathway neuroscientists already knew real flies use to chase things. The escape path is the fly's giant fiber." |
| 3 | **Swarm**. Wait for sixteen boards. | Sixteen flies, sixteen independent brains. | "Sixteen copies of the brain, simulated together." |
| 4 | Click one board. In the lesion lab click **Steering · DNa02**. | That board gets a border. Two violet rings appear in the brain. The snake wanders and stops finding food, but keeps dodging walls. | "I have just silenced two cells out of 165,000. This fly has lost the ability to go to food. It still avoids walls." |
| 5 | Click a second board. Un-click the first so only the new one is picked. **Escape · giant fiber**. | This snake still heads for food and crashes again and again. | "Two different cells, the opposite deficit. It still wants the food. It can no longer get out of the way. Measured over 16 games each: no steering neuron, 1.1 food but 170 moves survived; no giant fiber, 25 moves survived instead of 63. Silencing 2,000 random neurons changes nothing." |
| 6 | **Scrambled**. | Every snake drives into a wall within a few moves. | "Same neurons, same number of connections, same strengths, random targets. Three different scrambles score zero. So it is this wiring that plays, not just any network this size." |
| 7 | **One fly**, **Training**. Point at the QR code. | A blank readout starts playing randomly. The learning chart climbs. Phones can reward and punish moves. | "Here the fly's brain stays fixed and a tiny readout, under 4,000 numbers, learns from reward what the brain's output means for this game. Scan the code and you can train it." |
| 8 | If there is time: **Training · scrambled**. | A second curve on the chart that stays flat. | "Same learning rule on the scrambled brain. Over ten runs the real wiring reaches 22 food on average and the scrambled one stalls at 5. The real wiring makes the game learnable." |
| 9 | **Versus**, hand someone the keyboard. | A person races the fly for food. | "Arrow keys. The fly sees your snake as a threat." |

Close with: "Snake is not the point. It is how we test what a real brain's wiring does, and you can break it cell by cell."

## If something goes wrong

| Problem | Do this |
|---|---|
| Page blank or frozen | Ctrl+Shift+R. Still blank: stop `npm run dev`, delete `node_modules\.vite`, start it again. |
| "Server offline" | Look at the uvicorn terminal. If it died, start it again; it needs about a minute. Talk through step 1 meanwhile. |
| "Cooling down" banner | Let it cool, keep talking, or switch to **One fly**. It resumes by itself at 78 °C. |
| Live training is slow to improve | Say "here is one we trained earlier" and click **Trained**. Offline, over ten runs, learning always got there, but one fly on stage learns slowly; use **Swarm** with Training if the GPU is cool. |
| Laptop dies | Play the fallback video. Record one during rehearsal. |

## What not to claim

- Not "the fly learned Snake". In Training only the small readout learns. The brain's connections never change.
- Not "the trained mode proves the wiring matters". A trained readout also plays through a scrambled brain (14.3 against 18.8). The evidence is **Normal**, the lesions and the scrambled control.
- Not "it feels rewarded". When it eats, the sugar-taste neurons fire and the feeding motor neuron responds. The reward dopamine neurons do not respond in this model.
- Everything on screen is simulated activity. None of it was recorded from a fly.
- The three thresholds in the Normal rule are ours, chosen by hand. The neurons they read are the fly's.
