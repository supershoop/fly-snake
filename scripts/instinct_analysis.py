"""Why does the nothing-trained fly die, and can a fixed rule on NAMED fly neurons do better? Offline: brain responses come
from the saved response bank (data/bank-real.npz), so this needs no GPU and runs in seconds.

Part 1: per game situation, what the current rule (DNa02 + DNa01 left-minus-right) does vs what would be safe.
Part 2: play full games with candidate rules, sampling a recorded brain response for every move.

Run: .venv/Scripts/python scripts/instinct_analysis.py [--games 200]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain.connectome import DATA, load_connectome
from flybrain.snake import ALL_STATES, LEFT, RIGHT, STRAIGHT, Snake, teacher

parser = argparse.ArgumentParser()
parser.add_argument("--games", type=int, default=200)
parser.add_argument("--bank", default="bank-real")
parser.add_argument("--max-moves", type=int, default=600)
args = parser.parse_args()

bank = np.load(DATA / f"{args.bank}.npz")["counts"]  # [24 situations, trials, descending neurons], spikes per 100 ms
neurons = load_connectome().neurons
descending = neurons[neurons["superclass"].eq("descending_neuron")].reset_index(drop=True)


def cells(kind, side):
    return np.flatnonzero((descending["type"].eq(kind) & descending["side"].eq(side)).to_numpy())


RATE = {f"{kind}_{side}": bank[:, :, cells(kind, side)].mean(axis=2) * 10 for kind in ("DNa02", "DNa01", "DNp01") for side in "LR"}  # Hz
NAMES = ["LEFT", "STRAIGHT", "RIGHT"]


def steering_only(r, threshold=20):
    """Current rule: turn toward the side whose steering neurons fire more."""
    drive = (r["DNa02_L"] + r["DNa01_L"]) - (r["DNa02_R"] + r["DNa01_R"])
    return LEFT if drive > threshold else RIGHT if drive < -threshold else STRAIGHT


def steering_plus_escape(r, threshold=20, escape=150):
    """Same steering rule, plus the fly's escape neuron: when BOTH giant fibers (DNp01) fire the threat is straight ahead,
    so going straight is not an option - turn the way the steering neurons lean, else away from the louder giant fiber."""
    drive = (r["DNa02_L"] + r["DNa01_L"]) - (r["DNa02_R"] + r["DNa01_R"])
    if min(r["DNp01_L"], r["DNp01_R"]) > escape:
        if abs(drive) > threshold:
            return LEFT if drive > 0 else RIGHT
        return RIGHT if r["DNp01_L"] > r["DNp01_R"] else LEFT
    return LEFT if drive > threshold else RIGHT if drive < -threshold else STRAIGHT


def pursuit_with_escape_veto(r, threshold=20, veto=100):
    """Pursuit steering (DNa02 + DNa01), but escape overrides pursuit: never turn toward the side whose giant fiber (DNp01)
    fires much harder than the other. If the wanted turn is vetoed, go straight; if nothing is wanted but one giant fiber
    dominates, turn away from it."""
    drive = (r["DNa02_L"] + r["DNa01_L"]) - (r["DNa02_R"] + r["DNa01_R"])
    threat = r["DNp01_L"] - r["DNp01_R"]  # positive = threat on the left
    wanted = LEFT if drive > threshold else RIGHT if drive < -threshold else STRAIGHT
    if wanted == LEFT and threat > veto or wanted == RIGHT and threat < -veto:
        return STRAIGHT
    return wanted


def instinct(r, threshold=20, veto=100, alarm=150):
    """Pursuit steering with two escape behaviours, all on named cells (DNa02/DNa01 steering, DNp01 giant fiber):
    1. veto  - never turn toward the side whose giant fiber fires much harder;
    2. dodge - when both giant fibers fire (threat ahead) and nothing pulls sideways, turn away from the louder one."""
    drive = (r["DNa02_L"] + r["DNa01_L"]) - (r["DNa02_R"] + r["DNa01_R"])
    threat = r["DNp01_L"] - r["DNp01_R"]  # positive = more threat on the left
    wanted = LEFT if drive > threshold else RIGHT if drive < -threshold else STRAIGHT
    if wanted == LEFT and threat > veto or wanted == RIGHT and threat < -veto:
        wanted = STRAIGHT
    if wanted == STRAIGHT and min(r["DNp01_L"], r["DNp01_R"]) > alarm:
        wanted = RIGHT if threat > 0 else LEFT
    return wanted


def lesioned(rule, *silenced):
    """Silencing an output neuron offline = its rate is zero."""
    return lambda r: rule({k: (0.0 if k.split("_")[0] in silenced else v) for k, v in r.items()})


RULES = {"steering only (current Normal)": steering_only, "steering + escape (giant fiber)": steering_plus_escape,
         "pursuit with escape veto": pursuit_with_escape_veto, "instinct: pursuit + veto + dodge": instinct}
LESIONS = {"instinct, DNa02 silenced (steering)": lesioned(instinct, "DNa02"), "instinct, DNp01 silenced (giant fiber)": lesioned(instinct, "DNp01"),
           "instinct, DNa01 silenced (turn-away)": lesioned(instinct, "DNa01"), "steering only, DNa02 silenced": lesioned(steering_only, "DNa02"),
           "steering only, DNp01 silenced": lesioned(steering_only, "DNp01")}

print("Part 1 - mean firing (Hz) per situation and what each rule does most often. blocked = L/ahead/R")
print(f"{'food':8s} {'blocked':9s} | {'DNa02 L/R':>11s} {'DNa01 L/R':>11s} {'DNp01 L/R':>11s} | " + " | ".join(f"{name[:22]:22s}" for name in RULES) + " | safe moves")
for index, state in enumerate(ALL_STATES):
    food, *blocked = state
    mean = {k: v[index].mean() for k, v in RATE.items()}
    choices = []
    for rule in RULES.values():
        picks = np.bincount([rule({k: v[index, t] for k, v in RATE.items()}) for t in range(bank.shape[1])], minlength=3)
        action = int(picks.argmax())
        choices.append(f"{NAMES[action]:9s}{'  DIES' if blocked[action] else '      '}{picks[action] / picks.sum():6.0%}")
    safe = ",".join(NAMES[a][0] for a in range(3) if not blocked[a]) or "none"
    print(f"{NAMES[food]:8s} {''.join('X' if b else '.' for b in blocked):9s} | {mean['DNa02_L']:5.0f}/{mean['DNa02_R']:<5.0f} {mean['DNa01_L']:5.0f}/{mean['DNa01_R']:<5.0f} "
          f"{mean['DNp01_L']:5.0f}/{mean['DNp01_R']:<5.0f} | " + " | ".join(choices) + f" | {safe}")

print(f"\nPart 2 - {args.games} full games per rule, a recorded brain response sampled for every move")
state_index = {state: i for i, state in enumerate(ALL_STATES)}
rng = np.random.default_rng(0)
for name, rule in list(RULES.items()) + list(LESIONS.items()) + [("rule-based teacher (upper bound, no brain)", None)]:
    scores, moves = [], []
    for seed in range(args.games):
        game, count = Snake(seed=seed), 0
        while game.alive and count < args.max_moves:
            state = game.state(0, lookahead=False)
            if rule is None:
                action = teacher(state)
            else:
                i, t = state_index[state], rng.integers(bank.shape[1])
                action = rule({k: v[i, t] for k, v in RATE.items()})
            game.step(action)
            count += 1
        scores.append(game.score)
        moves.append(count)
    print(f"  {name:44s} mean food {np.mean(scores):5.2f}  median {np.median(scores):4.1f}  max {max(scores):3d}  mean moves survived {np.mean(moves):6.1f}")
