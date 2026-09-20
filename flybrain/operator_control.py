"""Private live-readout selection. No public links, model-change events, or action overrides."""
import asyncio
from collections import deque
import contextlib
import json
import os
from pathlib import Path
import secrets

from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import torch

from .readout import OnlineLearner, Policy

UI = Path(__file__).with_name("operator_ui")
PRESETS = (
    {"id": "crash", "label": "1 · Crash fast", "description": "A strongly straight-biased readout. Usually hits the wall within a few moves.", "model": None},
    {"id": "untrained", "label": "2 · Untrained", "description": "The saved random initial readout, before reward-driven evolution.", "model": "readout-evolved-initial"},
    {"id": "legacy", "label": "3 · Original trained", "description": "The earlier trained readout, before survival retraining.", "model": "readout-real-legacy"},
    {"id": "evolved", "label": "4 · Evolved", "description": "The saved reward-evolved readout. Performance varies between games.", "model": "readout-evolved-ridge100"},
    {"id": "best", "label": "5 · Best saved", "description": "The current survival-trained readout. Strong play, without a guarantee of perfect games.", "model": "readout-real"},
)


class OperatorControl:
    def __init__(self, key=""):
        self.key = key
        self.pending = deque()

    def authorized(self, key):
        return bool(self.key) and isinstance(key, str) and secrets.compare_digest(key.encode(), self.key.encode())

    @staticmethod
    def clear(experiment):
        experiment.operator_policy = None
        experiment.operator_preset = None

    def state(self, experiment):
        supported = experiment is not None and experiment.wiring == "real" and experiment.policy_name in ("trained", "learning") and getattr(experiment, "synaptic_parameters", None) is None
        return {"presets": [{k: v for k, v in preset.items() if k != "model"} for preset in PRESETS],
                "active": getattr(experiment, "operator_preset", None), "supported": supported,
                "mode": getattr(experiment, "policy_name", None), "paused": getattr(experiment, "paused", False),
                "move": getattr(experiment, "move", 0),
                "reason": "" if supported else "Choose Trained or Training with original, real wiring on the host first."}

    async def submit(self, preset, experiment):
        if preset not in {item["id"] for item in PRESETS} | {"release"}:
            return {"ok": False, "reason": "Unknown preset."}
        if experiment is None or experiment.paused:
            return {"ok": False, "reason": "Wait for the host to resume the simulation."}
        if len(self.pending) >= 16:
            return {"ok": False, "reason": "A change is already queued. Try again shortly."}
        future = asyncio.get_running_loop().create_future()
        self.pending.append((preset, future))
        try:
            return await asyncio.wait_for(future, 15)
        except asyncio.TimeoutError:
            return {"ok": False, "reason": "The simulation did not respond in time. Try again."}

    def apply(self, experiment):
        """The brain worker calls this between moves; no model can change mid-decision."""
        while self.pending:
            preset_id, future = self.pending.popleft()
            if future.done():
                continue
            try:
                if experiment.paused:
                    raise ValueError("Wait for the host to resume the simulation.")
                if preset_id == "release":
                    current = experiment.policy()
                    self.clear(experiment)
                    restored = experiment.policy()
                    if isinstance(current, OnlineLearner) and isinstance(restored, OnlineLearner):
                        restored.moves = current.moves
                        restored.rng.set_state(current.rng.get_state())
                else:
                    state = self.state(experiment)
                    if not state["supported"]:
                        raise ValueError(state["reason"])
                    preset = next(item for item in PRESETS if item["id"] == preset_id)
                    saved = Policy.load(preset["model"] or "readout-real", experiment.device)
                    if preset["model"] is None:
                        saved = Policy(torch.zeros_like(saved.weight), torch.tensor([0., 30., 0.], device=experiment.device))
                    if saved.weight.shape != (3, len(experiment.readout_index)) or saved.bias.shape != (3,) or not torch.isfinite(saved.weight).all() or not torch.isfinite(saved.bias).all():
                        raise ValueError("This saved readout is incompatible with the running brain.")
                    current = experiment.policy()
                    policy = OnlineLearner.from_policy(saved) if experiment.policy_name == "learning" else saved
                    if isinstance(current, OnlineLearner) and isinstance(policy, OnlineLearner):
                        policy.moves = current.moves
                        policy.rate = current.rate
                        policy.rng.set_state(current.rng.get_state())
                    experiment.operator_policy, experiment.operator_preset = policy, preset_id
                experiment.feedback.clear()
                result = {"ok": True, "state": self.state(experiment)}
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                result = {"ok": False, "reason": str(error)}

            def resolve(future=future, result=result):
                if not future.done():
                    future.set_result(result)
            future.get_loop().call_soon_threadsafe(resolve)


def install_operator(app, get_experiment):
    control = OperatorControl(os.environ.get("FLY_OPERATOR_KEY", ""))

    @app.get("/operator/", include_in_schema=False)
    async def page():
        if not control.key:
            raise HTTPException(404)
        return FileResponse(UI / "index.html", headers={"Cache-Control": "no-store", "X-Robots-Tag": "noindex, nofollow", "Referrer-Policy": "no-referrer"})

    @app.websocket("/operator/ws")
    async def socket(websocket: WebSocket):
        if not control.key:
            await websocket.close()
            return
        await websocket.accept()
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), 5)
            auth = json.loads(raw) if len(raw) <= 1024 else None
            if not isinstance(auth, dict) or not control.authorized(auth.get("key")):
                await websocket.close(code=1008)
                return
            await websocket.send_json({"state": control.state(get_experiment())})
            while True:
                raw = await websocket.receive_text()
                if len(raw) > 1024:
                    await websocket.close(code=1009)
                    return
                message = json.loads(raw)
                if not isinstance(message, dict):
                    await websocket.close(code=1008)
                    return
                if message == {"status": True}:
                    await websocket.send_json({"state": control.state(get_experiment())})
                elif set(message) == {"preset"} and isinstance(message["preset"], str):
                    await websocket.send_json({"result": await control.submit(message["preset"], get_experiment())})
                else:
                    await websocket.send_json({"result": {"ok": False, "reason": "Only preset selection is available here."}})
        except (WebSocketDisconnect, asyncio.TimeoutError, ValueError):
            with contextlib.suppress(Exception):
                await websocket.close(code=1008)

    return control
