"""Does better sensing help the nothing-trained fly? The instinct policy (DNa02/DNa01 pursuit + giant-fiber veto and dodge) on
(a) the 5 on/off channels, where a threat straight ahead saturates both giant fibers and hides which side is worse, and
(b) the retinotopic eye (flybrain/vision.py), where obstacles light graded, side-specific detector cells up to 4 cells away.
Same boards, live brain in the loop, nothing trained in either.

Run: .venv/Scripts/python scripts/instinct_senses_eval.py [--games 16] [--max-moves 300]
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain.brain import Brain
from flybrain.channels import build_channels
from flybrain.connectome import load_connectome
from flybrain.readout import InstinctPolicy
from flybrain.snake import Snake
from flybrain.vision import build_retina

parser = argparse.ArgumentParser()
parser.add_argument("--games", type=int, default=16)
parser.add_argument("--max-moves", type=int, default=300)
args = parser.parse_args()

connectome = load_connectome()
channels, retina = build_channels(connectome), build_retina(connectome)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
readout_index = channels.readout_index.to(device)
readout = connectome.neurons.loc[channels.readout_index.numpy()]
groups = {f"{name}_{s}": torch.as_tensor(np.flatnonzero((readout["type"].eq(name) & readout["side"].eq(s)).to_numpy()), device=device)
          for name in ("DNa02", "DNa01", "DNp01") for s in "LR"}
policy = InstinctPolicy(groups, 100.0)
stim = torch.cat([channels.stim_index.to(device), retina.index.to(device)])

n = args.games
games = [Snake(seed=i % n) for i in range(2 * n)]  # first half: channels, second half: retina, same boards
moves, causes = np.zeros(2 * n), [""] * (2 * n)
brain = Brain(connectome, batch=2 * n, seed=1)
for _ in range(args.max_moves):
    if not any(g.alive for g in games):
        break
    levels = np.zeros((5, 2 * n), dtype=np.float32)
    view = np.zeros((len(retina.index), 2 * n), dtype=np.float32)
    for i, game in enumerate(games):
        if i < n:
            levels[:, i] = game.encode()
        else:
            view[:, i] = retina.render(game, 0)
    drive = torch.cat([channels.levels(torch.as_tensor(levels, device=device)), torch.as_tensor(view, device=device)])
    actions = policy.act(brain.run(100, stim, drive, readout_index))[0].tolist()
    for i, (game, action) in enumerate(zip(games, actions)):
        if game.alive:
            game.step(action)
            moves[i] += 1
            if not game.alive:
                causes[i] = getattr(game.snakes[0], "end_reason", "")
for label, block in (("5 on/off channels", slice(0, n)), ("retinotopic eye", slice(n, 2 * n))):
    scores = np.array([g.score for g in games[block]])
    ends = [c for c in causes[block] if c]
    print(f"instinct on {label:18s} food {scores.mean():5.2f} +- {scores.std() / np.sqrt(n):4.2f}  max {scores.max():2d}  moves survived {moves[block].mean():6.1f}  "
          f"alive at the limit {sum(g.alive for g in games[block])}/{n}  deaths: {dict((c, ends.count(c)) for c in set(ends))}")
