"""Phone feedback for the shared experiment; no second brain or audience mode controls."""
import asyncio
from collections import deque
import contextlib
import ipaddress
import json
import os
from pathlib import Path
import socket
from urllib.parse import urlsplit, urlunsplit

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

PUBLIC = Path(__file__).resolve().parents[1] / "public" / "feedback"


def feedback_urls(websocket):
    """Prefer the reachable server host; replace loopback with LAN interfaces."""
    override = os.environ.get("FLY_FEEDBACK_URL")
    if override:
        return [override]
    url = urlsplit(str(websocket.url))
    scheme = "https" if url.scheme == "wss" else "http"
    path = url.path.removesuffix("/ws") + "/feedback/"
    host = url.hostname or "localhost"
    try:
        local = ipaddress.ip_address(host).is_loopback or ipaddress.ip_address(host).is_unspecified
    except ValueError:
        local = host.lower() == "localhost"
    if not local:
        return [urlunsplit((scheme, url.netloc, path, "", ""))]
    addresses = []
    # This UDP connect selects a route without sending a packet.
    with contextlib.suppress(OSError):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 80))
            addresses.append(probe.getsockname()[0])
    with contextlib.suppress(OSError):
        addresses.extend(item[4][0] for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    port = f":{url.port}" if url.port else ""
    return [urlunsplit((scheme, f"{address}{port}", path, "", "")) for address in dict.fromkeys(addresses)
            if not ipaddress.ip_address(address).is_loopback and not ipaddress.ip_address(address).is_unspecified]


class AudienceFeedback:
    def __init__(self):
        self.clients = set()
        self.pending = deque()
        self.latest = None

    async def submit(self, message, experiment):
        if not isinstance(message, dict) or set(message) != {"id", "feedback", "fly", "move"}:
            return {"status": "rejected", "reason": "Send feedback for a displayed move only."}
        if not isinstance(message["id"], str) or len(message["id"]) > 80:
            return {"status": "rejected", "reason": "Invalid feedback request."}
        if experiment is None or experiment.paused:
            return {"status": "rejected", "reason": "Waiting for the host to resume the game."}
        if len(self.pending) >= 128:
            return {"status": "rejected", "reason": "The feedback queue is full. Try again in a moment."}
        future = asyncio.get_running_loop().create_future()
        self.pending.append((message, future))
        try:
            return await asyncio.wait_for(future, timeout=10)
        except asyncio.TimeoutError:
            return {"status": "rejected", "reason": "The brain did not respond in time. Try a new move."}

    def apply(self, experiment):
        """Called on the simulation worker between moves, after host commands."""
        while self.pending:
            message, future = self.pending.popleft()
            if future.done():
                continue
            if experiment.paused or experiment.override is not None:
                receipt = {"status": "rejected", "reason": "Waiting for the host to resume normal play."}
            else:
                learner = experiment.policy() if experiment.policy_name == "learning" else None
                experiment.feedback.apply(message, learner)
                receipt = dict(experiment.feedback.last)
            def resolve(future=future, receipt=receipt):
                if not future.done():
                    future.set_result(receipt)
            future.get_loop().call_soon_threadsafe(resolve)

    async def publish(self, frame=None, *, paused=False):
        if frame is not None:
            self.latest = {key: frame[key] for key in ("move", "policy", "manual", "selected", "arenas", "flies")}
            self.latest["feedback"] = {key: frame["learning"]["feedback"][key] for key in ("positive", "negative")}
        elif self.latest is None or self.latest.get("paused") == paused:
            return
        self.latest["paused"] = paused
        message = json.dumps({"frame": self.latest})
        async def send(client):
            try:
                await asyncio.wait_for(client.send_text(message), timeout=1)
            except Exception:
                self.clients.discard(client)
                with contextlib.suppress(Exception):
                    await client.close()
        await asyncio.gather(*(send(client) for client in list(self.clients)))


def install_audience(app, get_experiment):
    audience = AudienceFeedback()

    @app.get("/feedback/")
    async def page():
        return FileResponse(PUBLIC / "index.html")

    @app.get("/feedback/controller.js")
    async def script():
        return FileResponse(PUBLIC / "controller.js", media_type="text/javascript")

    @app.get("/feedback/style.css")
    async def style():
        return FileResponse(PUBLIC / "style.css", media_type="text/css")

    @app.websocket("/feedback/ws")
    async def socket(websocket: WebSocket):
        await websocket.accept()
        audience.clients.add(websocket)
        try:
            if audience.latest is not None:
                await websocket.send_json({"frame": audience.latest})
            while True:
                raw = await websocket.receive_text()
                if len(raw) > 2048:
                    await websocket.close(code=1009)
                    break
                try:
                    message = json.loads(raw)
                except ValueError:
                    await websocket.send_json({"receipt": {"status": "rejected", "reason": "Invalid feedback request."}})
                    continue
                receipt = await audience.submit(message, get_experiment())
                request_id = message.get("id") if isinstance(message, dict) else None
                await websocket.send_json({"receipt": receipt, "id": request_id})
        except WebSocketDisconnect:
            pass
        finally:
            audience.clients.discard(websocket)

    return audience
