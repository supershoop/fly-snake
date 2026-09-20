"""Private selector transport and real readout plumbing; no performance estimates."""
import asyncio
from contextlib import asynccontextmanager
import json
import socket
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

from fastapi import FastAPI
import torch
import uvicorn
import websockets

from flybrain.operator_control import OperatorControl, PRESETS, install_operator
from flybrain.readout import OnlineLearner, Policy
from test_audience_gateway import gateway
from test_feedback import small_experiment


@asynccontextmanager
async def running(app):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.listen()
    service = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    task = asyncio.create_task(service.serve(sockets=[listener]))
    try:
        for _ in range(100):
            if service.started:
                break
            await asyncio.sleep(.01)
        assert service.started
        yield port
    finally:
        service.should_exit = True
        await task
        listener.close()


class OperatorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.experiment = small_experiment()
        self.control = OperatorControl("test-private-key")
        self.saved = Policy(torch.tensor([[.4, 0.], [0., .4], [.2, .2]]), torch.zeros(3))

    async def select(self, preset):
        with patch.object(Policy, "load", return_value=self.saved):
            task = asyncio.create_task(self.control.submit(preset, self.experiment))
            await asyncio.sleep(0)
            await asyncio.to_thread(self.control.apply, self.experiment)
            return await task

    async def test_every_preset_preserves_game_mode_and_continues_brain_simulation(self):
        exp = self.experiment
        for preset in PRESETS:
            arenas = exp.arenas
            before = exp.tick()
            old_learner = exp.policy()
            result = await self.select(preset["id"])
            self.assertTrue(result["ok"])
            self.assertIs(exp.arenas, arenas)
            self.assertEqual(exp.move, before["move"])
            self.assertEqual(exp.policy_name, "learning")
            self.assertEqual(exp.wiring, "real")
            self.assertFalse(exp.feedback.decisions)
            self.assertEqual(exp.policy().moves, old_learner.moves)
            self.assertIsInstance(exp.policy(), OnlineLearner)
            exp.brains["real"].run.reset_mock()
            after = exp.tick()
            exp.brains["real"].run.assert_called_once()
            self.assertEqual(set(before), set(after))
            self.assertNotIn("operator", json.dumps(after))
            self.assertEqual(after["policy"], before["policy"])
            self.assertEqual(after["learning"]["moves"], before["learning"]["moves"] + 1)
            if preset["id"] == "crash":
                self.assertEqual(after["flies"][0]["action"], 1)

    async def test_feedback_trains_selected_copy_and_release_restores_host(self):
        exp = self.experiment
        host = exp.policy()
        exp.tick()
        stale_move = exp.move
        self.assertTrue((await self.select("best"))["ok"])
        active = exp.policy()
        before = active.weight.clone()
        exp.feedback.apply({"feedback": 1, "move": stale_move}, active)
        self.assertEqual(exp.feedback.last["status"], "rejected")
        frame = exp.tick()
        exp.handle({"feedback": 1, "move": frame["move"]})
        self.assertEqual(exp.feedback.last["status"], "applied")
        self.assertFalse(torch.equal(before, active.weight))
        torch.testing.assert_close(self.saved.weight, before)
        moves = active.moves
        self.assertTrue((await self.select("release"))["ok"])
        self.assertIs(exp.policy(), host)
        self.assertEqual(host.moves, moves)
        self.assertIsNone(exp.operator_preset)

    async def test_host_choice_clears_override_and_invalid_modes_or_models_cannot_switch(self):
        exp = self.experiment
        self.assertTrue((await self.select("best"))["ok"])
        exp.handle({"policy": "trained"})
        self.assertIsNone(exp.operator_policy)
        self.assertTrue((await self.select("best"))["ok"])
        self.assertNotIsInstance(exp.policy(), OnlineLearner)
        exp.handle({"policy": "instinct"})
        self.assertFalse((await self.select("best"))["ok"])
        exp.policy_name = "trained"
        exp.wiring = "shuffled"
        self.assertFalse((await self.select("best"))["ok"])
        exp.wiring = "real"
        exp.synaptic_parameters = [1]
        self.assertFalse((await self.select("best"))["ok"])
        exp.synaptic_parameters = None
        self.saved.bias = torch.zeros(2)
        self.assertFalse((await self.select("best"))["ok"])
        self.assertIsNone(exp.operator_policy)
        self.assertFalse((await self.select("unknown"))["ok"])
        exp.paused = True
        self.assertFalse((await self.select("best"))["ok"])

    async def test_cancelled_request_never_changes_readout(self):
        task = asyncio.create_task(self.control.submit("crash", self.experiment))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.control.apply(self.experiment)
        self.assertIsNone(getattr(self.experiment, "operator_policy", None))

    async def test_authenticated_gateway_and_private_protocol(self):
        app = FastAPI()
        with patch.dict("os.environ", {"FLY_OPERATOR_KEY": self.control.key}):
            control = install_operator(app, lambda: self.experiment)
        async with running(app) as backend:
            proxy = gateway.create_app(f"ws://127.0.0.1:{backend}/feedback/ws", f"ws://127.0.0.1:{backend}/operator/ws")
            async with running(proxy) as port:
                url = f"ws://127.0.0.1:{port}/operator/ws"
                for key in ("wrong", "秘密", None):
                    async with websockets.connect(url) as ws:
                        await ws.send(json.dumps({"key": key}))
                        with self.assertRaises(websockets.ConnectionClosed) as error:
                            await ws.recv()
                        self.assertEqual(error.exception.rcvd.code, 1008)
                async with websockets.connect(url) as ws:
                    await ws.send(json.dumps({"key": control.key}))
                    state = json.loads(await ws.recv())["state"]
                    self.assertEqual(len(state["presets"]), 5)
                    await ws.send(json.dumps({"policy": "hardwired"}))
                    self.assertFalse(json.loads(await ws.recv())["result"]["ok"])
                    await ws.send(json.dumps({"preset": "best"}))
                    for _ in range(100):
                        if control.pending:
                            break
                        await asyncio.sleep(.01)
                    with patch.object(Policy, "load", return_value=self.saved):
                        await asyncio.to_thread(control.apply, self.experiment)
                    self.assertTrue(json.loads(await ws.recv())["result"]["ok"])
                    await ws.send(json.dumps({"status": True}))
                    self.assertEqual(json.loads(await ws.recv())["state"]["active"], "best")
                def get():
                    with urlopen(f"http://127.0.0.1:{port}/operator/", timeout=5) as response:
                        return response.headers, response.read().decode()
                headers, html = await asyncio.to_thread(get)
                self.assertEqual(headers["Cache-Control"], "no-store")
                self.assertNotIn(control.key, html)

    async def test_disabled_operator_route_is_absent(self):
        app = FastAPI()
        with patch.dict("os.environ", {"FLY_OPERATOR_KEY": ""}):
            install_operator(app, lambda: self.experiment)
        async with running(app) as port:
            with self.assertRaises(HTTPError) as error:
                await asyncio.to_thread(urlopen, f"http://127.0.0.1:{port}/operator/")
            self.assertEqual(error.exception.code, 404)
            with self.assertRaises(websockets.exceptions.InvalidStatus):
                async with websockets.connect(f"ws://127.0.0.1:{port}/operator/ws"):
                    self.fail("Disabled operator endpoint accepted a connection")
