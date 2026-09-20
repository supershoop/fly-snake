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
from .leaderboard import Leaderboard, clean_name
from .readout import HardwiredPolicy, InstinctPolicy, OnlineLearner, Policy
from .vision import VisionDisplay, VisionUntrained, build_retina
from .snake import Arena, HEADING_NAMES
from .thermal import ThermalGuard

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
# What the fly feels when the game says so, as real sensory neurons (regex on annotation `type`). Probed in
# scripts/event_probe.py: sugar taste drives the feeding motor neuron MN9 (~30 Hz); the heat/humidity receptors light
# ~6,000 neurons and drive the punishment dopamine neurons PPL1 (~80 Hz). Neither moves DNa02 or the giant fiber.
EVENT_SOURCES = {"taste": r"LB3.*|claw_tpGRN", "pain": r"HRN_.*|TRN_.*"}
EVENT_NAMES = list(EVENT_SOURCES)
LESION_PRESETS = ["DNa02", "DNp01", "AOTU0(25|12|15)"]


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
        self.readout_neurons, self.encoder = readout, "channels"
        self.retina = build_retina(self.connectome)
        self.display = VisionDisplay(self.connectome, self.retina)
        kinds = self.connectome.neurons["type"].fillna("")
        event_masks = [kinds.str.fullmatch(EVENT_SOURCES[name]).to_numpy() for name in EVENT_NAMES]
        event_cells = np.flatnonzero(np.any(event_masks, axis=0))
        self.event_matrix = torch.as_tensor(np.stack([m[event_cells] for m in event_masks], axis=1).astype(np.float32), device=self.device)  # [cells, events]
        self.events_enabled = True
        self.leaderboard, self.player, self.round_over = Leaderboard(), "anonymous", False
        self.all_stim_index = torch.cat([self.stim_index, self.retina.index.to(self.device), torch.as_tensor(event_cells, device=self.device)])
        self.steer = {f"{kind}_{side}": torch.as_tensor(np.flatnonzero((readout["type"].eq(kind) & readout["side"].eq(side)).to_numpy()), device=self.device)
                      for kind in STEER_TYPES + ["DNp01"] for side in "LR"}
        self.brains, self.policies = {}, {}
        self.wiring, self.policy_name = "real", "trained"
        self.paused, self.override, self.selected, self.clock = False, None, 0, 0.0
        self.sensor, self.sensor_seen = {}, 0.0
        self.feedback, self.move = HumanFeedback(), 0
        self.history: list[float] = []  # score of every finished fly game, oldest first
        self.death_hold = 0.0           # seconds the game holds still after the displayed fly dies (set by the page)
        self.step_rate = 0.0            # host-chosen cap on moves/second; 0 = uncapped (as fast as the brain computes)
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
        self.pending_events = np.zeros((len(EVENT_NAMES), len(self.flies)), dtype=np.float32)  # felt during the NEXT brain window
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
        retina = self.encoder == "retina"
        key = (self.policy_name, self.wiring, "retina") if retina else (self.policy_name, self.wiring)
        if key not in self.policies:
            if self.policy_name == "instinct":
                self.policies[key] = InstinctPolicy(self.steer, WINDOW_MS)
            elif self.policy_name == "hardwired":
                self.policies[key] = VisionUntrained(self.readout_neurons) if retina else HardwiredPolicy(self.channels.steer_sign)
            elif self.policy_name == "learning":
                self.policies[key] = OnlineLearner(len(self.readout_index), self.device)
            else:
                self.policies[key] = Policy.load("readout-vision" if retina else f"readout-{self.wiring}", self.device)
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
        if "events" in message:  # taste on eating, pain on dying
            self.events_enabled = bool(message["events"])
        if message.get("encoder") in ("channels", "retina"):  # 5 on/off channels, or the connectome-derived retinotopic eye
            self.encoder = message["encoder"]
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
        if "player" in message:  # the human's name for the versus leaderboard
            self.player = clean_name(message["player"])
        if "deathHold" in message:  # the page reports how long its death animation lasts
            self.death_hold = min(5.0, max(0.0, float(message["deathHold"])))
        if "stepRate" in message:  # host slider: cap moves/second; 0 or omitted-below-min means uncapped
            rate = float(message["stepRate"])
            self.step_rate = 0.0 if rate <= 0 else min(20.0, max(0.25, rate))
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
        if self.round_over:  # the human died last move: both snakes start the next round from scratch
            self.round_over = False
            for arena in self.arenas:
                arena.reset()
            brain_now = self.brains.get(self.wiring)
            if brain_now is not None:
                brain_now.reset()
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
        view = np.zeros((len(self.retina.index), len(self.flies)), dtype=np.float32)  # [retina cells, B]
        for f, (a, s) in enumerate(self.flies):  # with the channel encoder only the displayed fly's view is needed
            if self.encoder == "retina" or f == self.selected:
                view[:, f] = self.retina.render(self.arenas[a], s)
        if self.encoder == "retina" and self.override is None:
            levels = np.zeros_like(levels)  # the game reaches the brain through the retina only
        shown_to_retina = view if self.encoder == "retina" and self.override is None else np.zeros_like(view)
        felt, self.pending_events = self.pending_events, np.zeros_like(self.pending_events)
        feeding = felt[EVENT_NAMES.index("taste")] > 0  # [B] visual input suppresses the feeding response (scripts/event_probe.py), so a
        if feeding.any() and self.override is None:     # feeding fly pauses: no visual input and no move for this one window
            levels[:, feeding] = 0.0
            shown_to_retina = shown_to_retina.copy()
            shown_to_retina[:, feeding] = 0.0
        drive = torch.cat([self.channels.levels(torch.as_tensor(levels, device=self.device)), torch.as_tensor(shown_to_retina, device=self.device),
                           self.event_matrix @ torch.as_tensor(felt, device=self.device)])
        counts = brain.run(WINDOW_MS, self.all_stim_index, drive)
        dn_counts = counts[self.readout_index]
        actions, probabilities = policy.act(dn_counts)
        actions = actions.tolist()
        brain.reset_brains(torch.as_tensor(felt[EVENT_NAMES.index("pain")] > 0))  # that fly died: its next life starts with a quiet brain

        rewards = np.zeros(len(self.flies), dtype=np.float32)
        eligible = [False] * len(self.flies)
        # Captured before stepping: the arena rotates headings, but phone D-pad votes name an
        # absolute direction and must be resolved against the facing the fly decided from.
        headings = [self.arenas[a].snakes[s].heading for a, s in self.flies]
        if self.override is None:
            for a, arena in enumerate(self.arenas):
                outcome = arena.step({s: actions[f] for f, (fa, s) in enumerate(self.flies) if fa == a},
                                     hold={s for f, (fa, s) in enumerate(self.flies) if fa == a and feeding[f]})
                for f, (fa, s) in enumerate(self.flies):
                    if fa == a:
                        rewards[f] = outcome.get(s, 0.0)
                        eligible[f] = s in outcome
                        if s in outcome and not arena.snakes[s].alive:
                            self.history.append(arena.snakes[s].last_score)
                        if self.events_enabled and rewards[f] >= 1:
                            self.pending_events[EVENT_NAMES.index("taste"), f] = 1.0
                        if self.events_enabled and rewards[f] <= -1:
                            self.pending_events[EVENT_NAMES.index("pain"), f] = 1.0
        if self.layout == "versus" and self.override is None:
            for arena in self.arenas:
                humans = [snake for snake in arena.snakes if snake.kind == "human"]
                if any(not snake.alive for snake in humans):  # one round = one human life
                    fly_score = max((snake.score if snake.alive else snake.last_score) for snake in arena.snakes if snake.kind == "fly")
                    mode = "scrambled wiring" if self.wiring == "shuffled" else {"instinct": "normal", "hardwired": "normal", "learning": "training"}.get(self.policy_name, "trained")
                    self.leaderboard.record(self.player, max(snake.last_score for snake in humans if not snake.alive), fly_score, mode)
                    self.round_over = True
        self.move += 1
        if isinstance(policy, OnlineLearner) and self.override is None:
            self.feedback.remember(self.move, policy, eligible, headings)
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
            # "heading" is the pre-move facing this decision was made from, which is what a
            # D-pad vote is resolved against. The board's snake heading has already turned.
            "flies": [{"arena": a, "snake": s, "heading": headings[f],
                       "channels": dict(zip(CHANNEL_NAMES, levels[:, f].tolist())), "action": actions[f],
                       "probabilities": [round(p, 3) for p in probabilities[f].tolist()], "reward": float(rewards[f]),
                       "feedbackEligible": eligible[f] and isinstance(policy, OnlineLearner),
                       "event": next((name for e, name in enumerate(EVENT_NAMES) if felt[e, f] > 0), None),
                       "steer": {name: round(values[f], 1) for name, values in steer.items()}, "lesion": self.lesions[f]}
                      for f, (a, s) in enumerate(self.flies)],
            "selected": self.selected, "lesionPresets": LESION_PRESETS, "encoder": self.encoder, "events": self.events_enabled,
            "leaderboard": {"player": self.player, "top": self.leaderboard.top(), **self.leaderboard.tally()} if self.layout == "versus" else None,
            "vision": self.display.live(counts[:, self.selected], WINDOW_MS / 1000, view[:, self.selected]),
            "silenced": self.silenced_ids[self.selected], "silencedTotal": int(self.silenced_total[self.selected]),
            "silencedByFly": {str(f): ids for f, ids in enumerate(self.silenced_ids) if ids},
            "learning": {"moves": getattr(policy, "moves", 0), "games": len(self.history), "scores": self.history[-300:],
                         "feedback": self.feedback.state()},
            "activeNeurons": int((rates > 0).sum()), "totalNeurons": self.connectome.n,
            "values": [[int(i), round(float(v), 3)] for i, v in zip(self.visible_ids[firing], shown[firing])],
        }


    def event_frame(self, frame: dict) -> dict | None:
        """The game is holding for a death scene: let the brain feel what is pending (the pain) now, with no game move, so
        the flash is seen while the fly flails. Returns the same frame with fresh brain activity, or None if nothing is pending."""
        if not self.pending_events.any():
            return None
        brain = self.brain()
        felt, self.pending_events = self.pending_events, np.zeros_like(self.pending_events)
        quiet = torch.zeros((len(self.all_stim_index) - self.event_matrix.shape[0], len(self.flies)), device=self.device)
        counts = brain.run(WINDOW_MS, self.all_stim_index, torch.cat([quiet, self.event_matrix @ torch.as_tensor(felt, device=self.device)]))
        brain.reset_brains(torch.as_tensor(felt[EVENT_NAMES.index("pain")] > 0))  # the pain burst is self-sustaining otherwise (~7,000 neurons ring on)
        seconds = WINDOW_MS / 1000
        rates = counts[:, self.selected] / seconds
        shown = (rates[self.visible] / FULL_SCALE_HZ).clamp(max=1.0).cpu().numpy()
        firing = np.flatnonzero(shown)
        self.clock += seconds
        flies = [dict(fly, reward=0.0, event=next((name for e, name in enumerate(EVENT_NAMES) if felt[e, f] > 0), None)) for f, fly in enumerate(frame["flies"])]
        return dict(frame, time=round(self.clock, 3), flies=flies, eventOnly=True,
                    vision=self.display.live(counts[:, self.selected], seconds, np.zeros(len(self.retina.index), dtype=np.float32)),
                    activeNeurons=int((rates > 0).sum()),
                    values=[[int(i), round(float(v), 3)] for i, v in zip(self.visible_ids[firing], shown[firing])])


app = FastAPI()
clients: set[WebSocket] = set()
visible: dict[WebSocket, bool] = {}  # pages report whether their tab is showing; clients that never report count as watching
guard = ThermalGuard()
experiment: Experiment | None = None
audience = install_audience(app, lambda: experiment)
operator = install_operator(app, lambda: experiment)


async def loop():
    global experiment
    experiment = await asyncio.to_thread(Experiment)
    record = open(os.environ["FLY_RECORD"], "a") if os.environ.get("FLY_RECORD") else None
    sent_at = None    # when the previous frame actually went out, for the achieved-rate readout
    last_message = None
    while True:
        unwatched = bool(clients) and not any(visible.get(client, True) for client in clients)  # every page is minimised or in a background tab
        if (not clients and not audience.clients) or experiment.paused or unwatched or guard.cooling:
            await audience.publish(paused=experiment.paused)
            if guard.cooling and last_message is not None:  # keep the page informed while the GPU cools down
                last_message["thermal"] = guard.status()
                for client in list(clients):
                    with contextlib.suppress(Exception):
                        await client.send_text(json.dumps(last_message))
                await asyncio.sleep(1.0)
            await asyncio.sleep(0.1)
            sent_at = None  # idle time is not part of the moves/second the slider promises
            continue
        started = time.monotonic()
        try:
            frame = await asyncio.to_thread(experiment.tick)
        except FileNotFoundError:  # e.g. scrambled-wiring readout not trained yet: scripts/train_readout.py --shuffled
            experiment.wiring, experiment.policy_name = "real", "trained"
            continue
        elapsed = time.monotonic() - started
        # A rate cap can only slow the game down: the brain still takes as long as it takes to
        # compute one move, so the slider's floor is whatever elapsed just now, not a promise.
        if experiment.step_rate:
            await asyncio.sleep(max(0.0, 1 / experiment.step_rate - elapsed))
        now = time.monotonic()
        # The achieved rate is the full period between sends, including any pacing sleep -
        # reporting only compute time would understate the cap and mislead the slider's readout.
        # Two decimals: the dashboard's slider moves in quarter-steps down to 0.25/s, and one
        # decimal cannot distinguish 0.25 from 0.2/0.3 once scheduling jitter is folded in.
        frame["stepRate"] = round(1 / max(now - sent_at, 1e-6), 2) if sent_at else None
        sent_at = now
        frame["thermal"] = guard.status()
        last_message = frame
        message = json.dumps(frame)
        if record:
            record.write(message + "\n")
        await audience.publish(frame, paused=experiment.paused)
        for client in list(clients):
            with contextlib.suppress(Exception):
                await client.send_text(message)
        if experiment.death_hold and frame["flies"][frame["selected"]]["reward"] <= -1:
            felt = await asyncio.to_thread(experiment.event_frame, frame)  # the pain lands while the death scene plays
            if felt is not None:
                for client in list(clients):
                    with contextlib.suppress(Exception):
                        await client.send_text(json.dumps(felt))
            await asyncio.sleep(experiment.death_hold)  # let the displayed fly's death scene play out
        if guard.gap():
            await asyncio.sleep(guard.gap())  # running hot: leave a gap between moves so the GPU gets a rest


@app.on_event("startup")
async def start():
    asyncio.create_task(guard.watch(lambda: f"{experiment.layout} x{len(experiment.flies)} {experiment.policy_name}" if experiment else "starting"))
    asyncio.create_task(loop())


@app.websocket("/ws")
async def socket(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            message = json.loads(await websocket.receive_text())
            if experiment is not None and isinstance(message, dict) and "hello" in message:  # one-off catalogue for the lesion search
                await websocket.send_text(json.dumps({"hello": {"types": experiment.type_catalogue(), "feedbackUrls": feedback_urls(websocket), "vision": experiment.display.static}}))
                continue
            if experiment is not None and isinstance(message, dict):
                # Human commands are intentionally queued immediately. This
                # keeps them ordered even while the brain calculation runs in a
                # worker thread, rather than letting a burst overwrite itself in
                # the generic between-tick inbox.
                if "human" in message:
                    experiment.queue_human_move(message["human"])
                    message = {key: value for key, value in message.items() if key != "human"}
                if "visible" in message:
                    visible[websocket] = bool(message.pop("visible"))
                if "paused" in message:
                    experiment.paused = bool(message["paused"])
                if message:
                    experiment.inbox.append(message)
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(websocket)
        visible.pop(websocket, None)
