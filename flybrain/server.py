"""Live loop: snake -> sensory channels -> connectome sim -> descending neurons -> action, streamed over WebSocket.

Run: .venv/Scripts/python -m uvicorn flybrain.server:app --port 8000
Client -> server messages: {"mode": "real" | "hardwired" | "shuffled"}, {"paused": bool},
                           {"stimulate": {"food_L": 1, ...} | null}  (manual override of the senses)
"""
import asyncio
import contextlib
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .brain import Brain
from .channels import CHANNEL_NAMES, STEER_TYPES, build_channels
from .connectome import load_connectome
from .readout import HardwiredPolicy, Policy
from .snake import Snake

warnings.filterwarnings("ignore")
WINDOW_MS = 100.0
FULL_SCALE_HZ = 100.0  # activity sent to the viewer = firing rate / FULL_SCALE_HZ, clamped to [0, 1]
ATLAS = Path(__file__).resolve().parents[1] / "public/data/brain-atlas"


class Experiment:
    def __init__(self):
        self.connectome = load_connectome()
        self.channels = build_channels(self.connectome)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stim_index = self.channels.stim_index.to(self.device)
        self.readout_index = self.channels.readout_index.to(self.device)
        atlas_ids = np.fromfile(ATLAS / "ids.bin", dtype="<u4")[np.fromfile(ATLAS / "groups.bin", dtype="u1") < 3]
        body_ids = self.connectome.neurons["bodyId"].to_numpy()
        self.visible = torch.as_tensor(np.flatnonzero(np.isin(body_ids, atlas_ids)), device=self.device)
        self.visible_ids = body_ids[self.visible.cpu().numpy()]
        readout = self.connectome.neurons.loc[self.channels.readout_index.numpy()]
        self.steer = {f"{kind}_{side}": torch.as_tensor(np.flatnonzero((readout["type"].eq(kind) & readout["side"].eq(side)).to_numpy()), device=self.device)
                      for kind in STEER_TYPES + ["DNp01"] for side in "LR"}
        self.brains, self.policies = {}, {}
        self.mode, self.paused, self.override = "real", False, None
        self.game, self.episode, self.best, self.clock = Snake(), 1, 0, 0.0

    def controller(self):
        wiring = "shuffled" if self.mode == "shuffled" else "real"
        if wiring not in self.brains:
            self.brains[wiring] = Brain(self.connectome, shuffled=wiring == "shuffled")
        if self.mode not in self.policies:
            self.policies[self.mode] = (HardwiredPolicy(self.channels.steer_sign) if self.mode == "hardwired"
                                        else Policy.load(f"readout-{wiring}", self.device))
        return self.brains[wiring], self.policies[self.mode]

    def tick(self) -> dict:
        brain, policy = self.controller()
        if not self.game.alive:
            self.game.reset()
            brain.reset()
            self.episode += 1
        levels = self.game.encode() if self.override is None else np.array([self.override.get(n, 0.0) for n in CHANNEL_NAMES], dtype=np.float32)
        drive = self.channels.levels(torch.as_tensor(levels, device=self.device)[:, None])
        counts = brain.run(WINDOW_MS, self.stim_index, drive)
        dn_counts = counts[self.readout_index]
        action, probabilities = policy.act(dn_counts)
        if self.override is None:
            self.game.step(int(action[0]))
        self.best = max(self.best, self.game.score)
        self.clock += WINDOW_MS / 1000
        rates = counts[:, 0] / (WINDOW_MS / 1000)
        shown = (rates[self.visible] / FULL_SCALE_HZ).clamp(max=1.0).cpu().numpy()
        firing = np.flatnonzero(shown)
        return {
            "time": round(self.clock, 3), "mode": self.mode, "episode": self.episode, "best": self.best,
            "snake": self.game.render_state() | {"heading": self.game.heading},
            "channels": dict(zip(CHANNEL_NAMES, levels.tolist())), "manual": self.override is not None,
            "action": int(action[0]), "probabilities": [round(p, 3) for p in probabilities[0].tolist()],
            "steer": {name: round(float(dn_counts[index, 0].sum()) / max(1, len(index)) / (WINDOW_MS / 1000), 1) for name, index in self.steer.items()},
            "activeNeurons": int((rates > 0).sum()), "totalNeurons": self.connectome.n,
            "values": [[int(i), round(float(v), 3)] for i, v in zip(self.visible_ids[firing], shown[firing])],
        }


app = FastAPI()
clients: set[WebSocket] = set()
experiment: Experiment | None = None


async def loop():
    global experiment
    experiment = await asyncio.to_thread(Experiment)
    while True:
        if not clients or experiment.paused:
            await asyncio.sleep(0.1)
            continue
        try:
            message = json.dumps(await asyncio.to_thread(experiment.tick))
        except FileNotFoundError:  # e.g. scrambled-wiring readout not trained yet: scripts/train_readout.py --shuffled
            experiment.mode = "real"
            continue
        for client in list(clients):
            with contextlib.suppress(Exception):
                await client.send_text(message)


@app.on_event("startup")
async def start():
    asyncio.create_task(loop())


@app.websocket("/ws")
async def socket(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            message = json.loads(await websocket.receive_text())
            if experiment is None:
                continue
            if message.get("mode") in ("real", "hardwired", "shuffled"):
                experiment.mode = message["mode"]
                experiment.game.reset()
            if "paused" in message:
                experiment.paused = bool(message["paused"])
            if "stimulate" in message:
                experiment.override = message["stimulate"]
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(websocket)
