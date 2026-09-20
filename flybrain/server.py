"""Live loop: snakes -> sensory channels -> connectome sim -> descending neurons -> actions, streamed over WebSocket.

Run: .venv/Scripts/python -m uvicorn flybrain.server:app --port 8000 [--host 0.0.0.0]
Set FLY_RECORD=frames.jsonl to record every frame (replay without a GPU: scripts/replay_server.py frames.jsonl).
The protocol (client -> server messages, server -> client frame) is documented in AGENTS.md.
"""
import asyncio
import contextlib
import json
import os
import time
import warnings
from collections import deque
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .brain import Brain, default_device
from .audience import feedback_urls, install_audience
from .operator_control import install_operator
from .channels import CHANNEL_NAMES, STEER_TYPES, build_channels
from .connectome import load_connectome
from .feedback import HumanFeedback
from .readout import HardwiredPolicy, InstinctPolicy, OnlineLearner, Policy
from .snake import Arena, HEADING_NAMES

warnings.filterwarnings("ignore")
WINDOW_MS = 100.0
FULL_SCALE_HZ = 100.0  # activity sent to the viewer = firing rate / FULL_SCALE_HZ, clamped to [0, 1]
SENSOR_HOLD_S = 0.6    # an external sensor reading goes stale after this long
HUMAN_INPUT_BUFFER = 2  # one upcoming turn plus one follow-up turn; repeated keydown events are coalesced
ATLAS = Path(__file__).resolve().parents[1] / "public/data/brain-atlas"
LAYOUTS = {  # name -> list of arenas, each (board size, snake kinds, foods)
    "solo": [(12, ("fly",), 1)],
    "swarm": [(12, ("fly",), 1)] * 16,
    "versus": [(16, ("fly", "human"), 2)],
    "arena": [(24, ("fly",) * 8, 5)],
}
LESION_PRESETS = ["LC10.*", "LC4", "LPLC2", "DNa02", "DNa01", "DNp01"]


class Experiment:
    def __init__(self):
        self.connectome = load_connectome()
        self.channels = build_channels(self.connectome)
        self.device = default_device()
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
        self.wiring, self.policy_name = "real", "trained"
        self.paused, self.override, self.selected, self.clock = False, None, 0, 0.0
        self.sensor, self.sensor_seen = {}, 0.0
        self.feedback, self.move = HumanFeedback(), 0
        self.history: list[float] = []  # score of every finished fly game, oldest first
        self.death_hold = 0.0           # seconds the game holds still after the displayed fly dies (set by the page)
        self.inbox: list[dict] = []     # client messages, applied between moves so they never race the simulation
        # Human movement arrives on the event-loop thread while tick() runs in a
        # worker. Consume only one heading per tick so a quick pair of turns is
        # preserved instead of every message collapsing into the last direction.
        self.human_moves: deque[str] = deque()
        self.human_move_lock = Lock()
        self.set_layout("solo")

    # --- configuration ---------------------------------------------------------------------------------------------
    def set_layout(self, name: str):
        self.feedback.clear()
        with self.human_move_lock:
            self.human_moves.clear()
        self.layout = name
        self.arenas = [Arena(size, kinds, foods, seed=i) for i, (size, kinds, foods) in enumerate(LAYOUTS[name])]
        self.flies = [(a, s) for a, arena in enumerate(self.arenas) for s, snake in enumerate(arena.snakes) if snake.kind == "fly"]
        if self.device.type == "cpu":
            torch.set_num_threads(min(os.cpu_count() or 1, 4 if len(self.flies) == 1 else 8))
        self.lesions: list[list[str]] = [[] for _ in self.flies]
        self.silenced_ids, self.silenced_total = [[] for _ in self.flies], [0] * len(self.flies)
        self.selected, self.history = 0, []
        for brain in self.brains.values():
            brain.resize(len(self.flies))

    def brain(self) -> Brain:
        if self.wiring not in self.brains:
            self.brains[self.wiring] = Brain(self.connectome, batch=len(self.flies), shuffled=self.wiring == "shuffled",
                                              device=str(self.device))
            self.apply_lesions()
        return self.brains[self.wiring]

    def policy(self):
        if getattr(self, "operator_policy", None) is not None:
            return self.operator_policy
        key = (self.policy_name, self.wiring)
        if key not in self.policies:
            if self.policy_name == "instinct":
                self.policies[key] = InstinctPolicy(self.steer, WINDOW_MS)
            elif self.policy_name == "hardwired":
                self.policies[key] = HardwiredPolicy(self.channels.steer_sign)
            elif self.policy_name == "learning":
                self.policies[key] = OnlineLearner(len(self.readout_index), self.device)
            else:
                self.policies[key] = Policy.load(f"readout-{self.wiring}", self.device)
        return self.policies[key]

    def apply_lesions(self):
        kinds = self.connectome.neurons["type"].fillna("")
        mask = np.zeros((self.connectome.n, len(self.flies)), dtype=bool)
        for fly, patterns in enumerate(self.lesions):
            for pattern in patterns:
                mask[:, fly] |= kinds.str.fullmatch(pattern).to_numpy()
        for brain in self.brains.values():
            brain.set_lesion(torch.as_tensor(mask))
        body_ids, drawn = self.connectome.neurons["bodyId"].to_numpy(), set(self.visible_ids.tolist())
        # per fly: silenced cells that the viewer draws (bodyIds), plus how many silenced cells there are in total
        self.silenced_ids = [[int(i) for i in body_ids[mask[:, fly]] if int(i) in drawn] for fly in range(len(self.flies))]
        self.silenced_total = mask.sum(axis=0).tolist()

    def type_catalogue(self) -> list:
        """[[type, cells, superclass], ...] for every annotated neuron type, most numerous first."""
        neurons = self.connectome.neurons.dropna(subset=["type"])
        table = neurons.groupby("type").agg(cells=("bodyId", "size"), superclass=("superclass", "first")).sort_values("cells", ascending=False)
        return [[kind, int(row.cells), row.superclass if isinstance(row.superclass, str) else ""] for kind, row in table.iterrows()]

    def queue_human_move(self, heading: object):
        """Retain at most two distinct player inputs, consuming one on each tick."""
        if self.layout != "versus" or not isinstance(heading, str) or heading not in HEADING_NAMES:
            return
        with self.human_move_lock:
            # Browsers generate repeated keydown events while a key is held. They
            # do not represent additional turns, so never let them fill the buffer.
            if self.human_moves and self.human_moves[-1] == heading:
                return
            if len(self.human_moves) < HUMAN_INPUT_BUFFER:
                self.human_moves.append(heading)

    def take_human_move(self) -> str | None:
        with self.human_move_lock:
            return self.human_moves.popleft() if self.human_moves else None

    def handle(self, message: dict):
        if {"policy", "learning", "wiring", "synaptic"}.intersection(message):
            operator.clear(self)
        if message.get("layout") in LAYOUTS:
            self.set_layout(message["layout"])
        if message.get("wiring") in ("real", "shuffled"):
            if self.wiring != message["wiring"]:
                self.feedback.clear()
            self.wiring = message["wiring"]
        if message.get("policy") in ("trained", "hardwired", "learning", "instinct"):
            if self.policy_name != message["policy"]:
                self.feedback.clear()
            self.policy_name = message["policy"]
        if message.get("learning") in ("reset", "pretrained"):
            if message["learning"] == "pretrained":
                trained = self.policies.get(("trained", self.wiring))
                if trained is None:
                    trained = Policy.load(f"readout-{self.wiring}", self.device)
                self.policies[("learning", self.wiring)] = OnlineLearner.from_policy(trained)
            else:
                self.policies.pop(("learning", self.wiring), None)
            self.policy_name = "learning"
            self.feedback = HumanFeedback()
            self.history = []
        if "lesion" in message:  # {"fly": index or null for all, "types": [regex on annotation type, ...]}
            targets = range(len(self.flies)) if message["lesion"].get("fly") is None else [int(message["lesion"]["fly"])]
            for fly in targets:
                self.lesions[fly] = list(message["lesion"].get("types", []))
            self.apply_lesions()
        if "feedback" in message:  # human reward (+) / punishment (-) for the move just made
            learner = self.policy() if self.policy_name == "learning" and self.override is None else None
            self.feedback.apply(message, learner)
        if "sensor" in message:  # external hardware: extra drive added on top of the game's senses, e.g. {"danger_ahead": 0.8}
            self.sensor, self.sensor_seen = dict(message["sensor"]), time.monotonic()
        if "stimulate" in message:  # manual override of all senses; the game holds still while it is set
            self.feedback.clear()
            self.override = message["stimulate"]
        if "human" in message:
            self.queue_human_move(message["human"])
        if "deathHold" in message:  # the page reports how long its death animation lasts
            self.death_hold = min(5.0, max(0.0, float(message["deathHold"])))
        if "select" in message:
            self.selected = int(message["select"]) % len(self.flies)
        if "paused" in message:
            self.paused = bool(message["paused"])

    # --- one move --------------------------------------------------------------------------------------------------
    def tick(self) -> dict:
        # One buffered heading is applied to this move. Messages received while
        # this tick is calculating remain queued for the following move.
        human_heading = self.take_human_move()
        while self.inbox:
            with contextlib.suppress(KeyError, ValueError, TypeError, IndexError, AttributeError):
                self.handle(self.inbox.pop(0))
        operator.apply(self)
        audience.apply(self)
        if human_heading is not None:
            for arena in self.arenas:
                for index, snake in enumerate(arena.snakes):
                    if snake.kind == "human":
                        arena.steer_human(index, human_heading)
        brain, policy = self.brain(), self.policy()
        levels = np.stack([self.arenas[a].encode(s) for a, s in self.flies], axis=1)  # [C, B]
        if self.override is not None:
            levels = np.repeat(np.array([[self.override.get(n, 0.0)] for n in CHANNEL_NAMES], dtype=np.float32), len(self.flies), axis=1)
        elif time.monotonic() - self.sensor_seen < SENSOR_HOLD_S:
            extra = np.array([[float(self.sensor.get(n, 0.0))] for n in CHANNEL_NAMES], dtype=np.float32)
            levels = np.clip(levels + extra, 0, 1)
        counts = brain.run(WINDOW_MS, self.stim_index, self.channels.levels(torch.as_tensor(levels, device=self.device)))
        dn_counts = counts[self.readout_index]
        actions, probabilities = policy.act(dn_counts)
        actions = actions.tolist()

        rewards = np.zeros(len(self.flies), dtype=np.float32)
        eligible = [False] * len(self.flies)
        if self.override is None:
            for a, arena in enumerate(self.arenas):
                outcome = arena.step({s: actions[f] for f, (fa, s) in enumerate(self.flies) if fa == a})
                for f, (fa, s) in enumerate(self.flies):
                    if fa == a:
                        rewards[f] = outcome.get(s, 0.0)
                        eligible[f] = s in outcome
                        if s in outcome and not arena.snakes[s].alive:
                            self.history.append(arena.snakes[s].last_score)
        self.move += 1
        if isinstance(policy, OnlineLearner) and self.override is None:
            self.feedback.remember(self.move, policy, eligible)
            policy.learn(torch.as_tensor(rewards))

        self.clock += WINDOW_MS / 1000
        seconds = WINDOW_MS / 1000
        rates = counts[:, self.selected] / seconds
        shown = (rates[self.visible] / FULL_SCALE_HZ).clamp(max=1.0).cpu().numpy()
        firing = np.flatnonzero(shown)
        steer = {name: (dn_counts[index].sum(dim=0) / max(1, len(index)) / seconds).tolist() for name, index in self.steer.items()}
        return {
            "time": round(self.clock, 3), "move": self.move, "layout": self.layout, "wiring": self.wiring, "policy": self.policy_name,
            "manual": self.override is not None, "sensor": self.sensor if time.monotonic() - self.sensor_seen < SENSOR_HOLD_S else {},
            "arenas": [arena.render_state() for arena in self.arenas],
            "flies": [{"arena": a, "snake": s, "channels": dict(zip(CHANNEL_NAMES, levels[:, f].tolist())), "action": actions[f],
                       "probabilities": [round(p, 3) for p in probabilities[f].tolist()], "reward": float(rewards[f]),
                       "feedbackEligible": eligible[f] and isinstance(policy, OnlineLearner),
                       "steer": {name: round(values[f], 1) for name, values in steer.items()}, "lesion": self.lesions[f]}
                      for f, (a, s) in enumerate(self.flies)],
            "selected": self.selected, "lesionPresets": LESION_PRESETS,
            "silenced": self.silenced_ids[self.selected], "silencedTotal": int(self.silenced_total[self.selected]),
            "silencedByFly": {str(f): ids for f, ids in enumerate(self.silenced_ids) if ids},
            "learning": {"moves": getattr(policy, "moves", 0), "games": len(self.history), "scores": self.history[-300:],
                         "feedback": self.feedback.state()},
            "activeNeurons": int((rates > 0).sum()), "totalNeurons": self.connectome.n,
            "values": [[int(i), round(float(v), 3)] for i, v in zip(self.visible_ids[firing], shown[firing])],
        }


app = FastAPI()
clients: set[WebSocket] = set()
experiment: Experiment | None = None
audience = install_audience(app, lambda: experiment)
operator = install_operator(app, lambda: experiment)


async def loop():
    global experiment
    experiment = await asyncio.to_thread(Experiment)
    record = open(os.environ["FLY_RECORD"], "a") if os.environ.get("FLY_RECORD") else None
    while True:
        if (not clients and not audience.clients) or experiment.paused:
            await audience.publish(paused=experiment.paused)
            await asyncio.sleep(0.1)
            continue
        try:
            frame = await asyncio.to_thread(experiment.tick)
            message = json.dumps(frame)
        except FileNotFoundError:  # e.g. scrambled-wiring readout not trained yet: scripts/train_readout.py --shuffled
            experiment.wiring, experiment.policy_name = "real", "trained"
            continue
        if record:
            record.write(message + "\n")
        await audience.publish(frame, paused=experiment.paused)
        for client in list(clients):
            with contextlib.suppress(Exception):
                await client.send_text(message)
        if experiment.death_hold and frame["flies"][frame["selected"]]["reward"] <= -1:
            await asyncio.sleep(experiment.death_hold)  # let the displayed fly's death scene play out


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
            if experiment is not None and isinstance(message, dict) and "hello" in message:  # one-off catalogue for the lesion search
                await websocket.send_text(json.dumps({"hello": {"types": experiment.type_catalogue(), "feedbackUrls": feedback_urls(websocket)}}))
                continue
            if experiment is not None and isinstance(message, dict):
                # Human commands are intentionally queued immediately. This
                # keeps them ordered even while the brain calculation runs in a
                # worker thread, rather than letting a burst overwrite itself in
                # the generic between-tick inbox.
                if "human" in message:
                    experiment.queue_human_move(message["human"])
                    message = {key: value for key, value in message.items() if key != "human"}
                if "paused" in message:
                    experiment.paused = bool(message["paused"])
                if message:
                    experiment.inbox.append(message)
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(websocket)
