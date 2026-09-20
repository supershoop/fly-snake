"""Continuous whole-brain rollouts for the direct-board research experiment."""
import numpy as np
import torch

from .board_encoder import BoardEncoder, board_sites
from .brain import Brain
from .channels import build_channels
from .connectome import load_connectome
from .readout import HardwiredPolicy
from .snake import Snake
from .synaptic import DECODER, SynapticAdapter


class BoardRunner:
    def __init__(self, device="cpu", size=12, food_sigma=3., obstacle_sigma=.8, gain=1., connectome=None):
        self.connectome = load_connectome() if connectome is None else connectome
        self.encoder = BoardEncoder.from_connectome(self.connectome, size=size, food_sigma=food_sigma,
                                                    obstacle_sigma=obstacle_sigma, gain=gain)
        self.channels = build_channels(self.connectome)
        self.sites = board_sites(self.connectome, self.encoder)
        self.brain = Brain(self.connectome, device=device, compiled=device == "cpu")
        self.adapter = SynapticAdapter(self.brain, self.sites)
        self.policy = HardwiredPolicy(self.channels.steer_sign, threshold=DECODER["threshold"])
        self.stimulus = torch.as_tensor(self.encoder.stimulus, device=device)
        self.readout = self.channels.readout_index.to(device)
        self.moves = 0
        if np.intersect1d(self.encoder.stimulus, self.channels.readout_index.numpy()).size:
            raise ValueError("Direct sensory input must never stimulate motor/readout neurons")

    def load(self, path):
        parameters, metadata = self.sites.load(path)
        if metadata.get("encoder") != self.encoder.metadata():
            raise ValueError("Checkpoint encoder does not match this input mapping")
        return parameters, metadata

    def save(self, path, parameters, **metadata):
        self.sites.save(path, parameters, encoder=self.encoder.metadata(), **metadata)

    def step(self, game, *, ablation=None):
        if ablation == "legacy_input":
            levels = self.channels.levels(torch.as_tensor(game.encode(), device=self.brain.device)[:, None])
            counts = self.brain.run(100., self.channels.stim_index.to(self.brain.device), levels, self.readout)
            action = int(self.policy.act(counts)[0][0])
            self.moves += 1
            return action, counts
        levels = self.encoder.encode(game)
        if ablation == "no_input":
            levels[:] = 0
        elif ablation == "food_only":
            levels[self.encoder.width ** 2:] = 0
        elif ablation is not None:
            raise ValueError("Unknown input ablation")
        counts = self.brain.run(100., self.stimulus,
                                torch.as_tensor(levels, device=self.brain.device)[:, None], self.readout)
        action = int(self.policy.act(counts)[0][0])
        self.moves += 1
        return action, counts

    def evaluate(self, parameters, seeds, limit, *, progress=None, ablation=None, capture=False):
        self.adapter.apply(parameters)
        records = []
        for seed in seeds:
            game = Snake(size=self.encoder.size, seed=int(seed))
            self.brain.reset()
            self.brain.rng.manual_seed(int(seed) + 1000000)
            reward_sum, actions, frames = 0., [0, 0, 0], []
            for move in range(limit):
                action, counts = self.step(game, ablation=ablation)
                if capture:
                    frames.append({"board": game.render_state(), "action": action,
                                   "steering_spike_difference": float((self.channels.steer_sign.to(counts.device) @ counts)[0])})
                actions[action] += 1
                reward_sum += game.step(action)
                if not game.alive:
                    break
            record = {"seed": int(seed), "brain_seed": int(seed) + 1000000, "food": game.score,
                      "moves": move + 1, "reward": reward_sum, "actions": actions,
                      "end_reason": game.snakes[0].end_reason, "alive_at_limit": game.alive}
            if capture:
                record["frames"] = frames
                record["final_board"] = game.render_state()
            records.append(record)
            if progress:
                progress({key: value for key, value in record.items() if key not in ("frames", "final_board")})
        return {"games": records, "mean_food": float(np.mean([g["food"] for g in records])),
                "mean_reward": float(np.mean([g["reward"] for g in records])),
                "mean_moves": float(np.mean([g["moves"] for g in records])),
                "collisions": sum(g["end_reason"] == "collision" for g in records),
                "starvations": sum(g["end_reason"] == "starvation" for g in records),
                "alive_at_limit": sum(g["alive_at_limit"] for g in records)}


def objective(result):
    # The dominant term is actual food. Shaped reward and a small survival term
    # break zero-food plateaus. No teacher labels or preferred actions are used.
    return result["mean_food"] + .05 * result["mean_reward"] + .001 * result["mean_moves"]
