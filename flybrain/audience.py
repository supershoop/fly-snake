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
# A tunnel writes its public URL here (scripts/public_tunnel.py). Read per request so the
# QR code follows a new tunnel without restarting the brain or losing the live learner.
URL_FILE = Path(__file__).resolve().parents[1] / ".feedback-url"


def published_url():
    """The explicit override, else a URL published by a running tunnel."""
    override = os.environ.get("FLY_FEEDBACK_URL", "").strip()
    if override:
        return override
    path = Path(os.environ.get("FLY_FEEDBACK_URL_FILE") or URL_FILE)
    with contextlib.suppress(OSError):
        published = path.read_text().strip()
        if published.startswith(("http://", "https://")):
            return published
    return None


def feedback_urls(websocket):
    """Prefer the reachable server host; replace loopback with LAN interfaces."""
    override = published_url()
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


# One unit of crowd influence per move, shared by everyone connected, so the audience biases
# the readout without drowning the game's own rewards (food +1, death -1, closer +-0.1) or the
# brain, which is never modified. A lone voter nudges as hard as a full room does together.
TEACH_BUDGET = 1.0


class AudienceFeedback:
    TEACH_KEYS = {"id", "direction", "fly", "move"}
    FEEDBACK_KEYS = {"id", "feedback", "fly", "move"}

    def __init__(self):
        self.clients = set()
        self.pending = deque()
        self.latest = None
        self.spent = {}   # move -> crowd influence already applied to it
        self.voted = {}   # move -> participants who already taught it

    def participants(self):
        """Connected phones, which is what divides each vote's weight."""
        return max(1, len(self.clients))

    def vote_weight(self, move, voter):
        """base / N, refused once a voter repeats or the move's shared budget is gone."""
        if voter in self.voted.setdefault(move, set()):
            return 0.0
        remaining = TEACH_BUDGET - self.spent.get(move, 0.0)
        return min(TEACH_BUDGET / self.participants(), remaining) if remaining > 1e-9 else 0.0

    def charge(self, move, voter, weight):
        self.spent[move] = self.spent.get(move, 0.0) + weight
        self.voted.setdefault(move, set()).add(voter)
        for stale in [key for key in self.spent if key < move - 256]:
            self.spent.pop(stale, None)
            self.voted.pop(stale, None)

    async def submit(self, message, experiment):
        if isinstance(message, dict) and set(message) == self.TEACH_KEYS:
            return await self.enqueue(message, experiment)
        if not isinstance(message, dict) or set(message) != self.FEEDBACK_KEYS:
            return {"status": "rejected", "reason": "Send feedback for a displayed move only."}
        return await self.enqueue(message, experiment)

    async def enqueue(self, message, experiment):
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
                if set(message) == self.TEACH_KEYS:
                    move, voter = message.get("move"), message.get("id")
                    weight = self.vote_weight(move, voter) if isinstance(move, int) else 0.0
                    receipt = dict(experiment.feedback.teach(message, learner, weight))
                    if receipt.get("status") == "applied":
                        self.charge(move, voter, weight)
                    elif weight <= 0 and isinstance(move, int) and learner is not None:
                        receipt = {"status": "rejected", "reason": "You already taught this move. Wait for the next one."}
                else:
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
            self.latest["audience"] = {"participants": self.participants(),
                                       "taught": frame["learning"]["feedback"].get("taught", 0),
                                       "directions": frame["learning"]["feedback"].get("directions", {}),
                                       "share": round(TEACH_BUDGET / self.participants(), 4)}
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
