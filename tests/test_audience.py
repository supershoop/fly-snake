"""Audience transport uses synthetic spikes; these tests make no performance claims."""
import asyncio
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.request import urlopen

import torch
import uvicorn
import websockets

from flybrain import server
from flybrain.audience import AudienceFeedback, feedback_urls
from test_feedback import small_experiment


class FeedbackUrlTests(unittest.TestCase):
    def setUp(self):
        # A tunnel running on this machine must not change what these cases observe.
        self.unpublished = Path(tempfile.mkdtemp()) / "absent-url"

    def environment(self, **extra):
        return patch.dict("os.environ", {"FLY_FEEDBACK_URL_FILE": str(self.unpublished), **extra}, clear=True)

    def test_public_tls_and_proxy_prefix_are_preserved(self):
        with self.environment():
            self.assertEqual(feedback_urls(SimpleNamespace(url="wss://demo.example/brain/ws")),
                             ["https://demo.example/brain/feedback/"])

    def test_loopback_is_replaced_with_lan_address(self):
        with self.environment(), patch("flybrain.audience.socket.socket") as probe, \
                patch("flybrain.audience.socket.getaddrinfo", return_value=[]):
            probe.return_value.__enter__.return_value.getsockname.return_value = ("192.168.1.12", 30000)
            self.assertEqual(feedback_urls(SimpleNamespace(url="ws://127.0.0.1:8000/ws")),
                             ["http://192.168.1.12:8000/feedback/"])

    def test_explicit_public_link(self):
        with self.environment(FLY_FEEDBACK_URL="https://fly.example/feedback/"):
            self.assertEqual(feedback_urls(SimpleNamespace(url="ws://localhost:8000/ws")),
                             ["https://fly.example/feedback/"])

    def test_running_tunnel_replaces_local_addresses_without_a_restart(self):
        published = Path(tempfile.mkdtemp()) / "url"
        published.write_text("https://tunnel.example/feedback/\n")
        with patch.dict("os.environ", {"FLY_FEEDBACK_URL_FILE": str(published)}, clear=True):
            self.assertEqual(feedback_urls(SimpleNamespace(url="ws://127.0.0.1:8000/ws")),
                             ["https://tunnel.example/feedback/"])
        # An explicit override still wins, and a closed tunnel stops being advertised.
        with patch.dict("os.environ", {"FLY_FEEDBACK_URL_FILE": str(published),
                                       "FLY_FEEDBACK_URL": "https://manual.example/feedback/"}, clear=True):
            self.assertEqual(feedback_urls(SimpleNamespace(url="ws://127.0.0.1:8000/ws")),
                             ["https://manual.example/feedback/"])
        published.unlink()
        with patch.dict("os.environ", {"FLY_FEEDBACK_URL_FILE": str(published)}, clear=True), \
                patch("flybrain.audience.socket.socket") as probe, \
                patch("flybrain.audience.socket.getaddrinfo", return_value=[]):
            probe.return_value.__enter__.return_value.getsockname.return_value = ("192.168.1.12", 30000)
            self.assertEqual(feedback_urls(SimpleNamespace(url="ws://127.0.0.1:8000/ws")),
                             ["http://192.168.1.12:8000/feedback/"])

    def test_a_blank_or_bogus_published_url_is_ignored(self):
        published = Path(tempfile.mkdtemp()) / "url"
        for content in ("", "   \n", "not-a-url", "javascript:alert(1)"):
            published.write_text(content)
            with patch.dict("os.environ", {"FLY_FEEDBACK_URL_FILE": str(published)}, clear=True), \
                    patch("flybrain.audience.socket.socket") as probe, \
                    patch("flybrain.audience.socket.getaddrinfo", return_value=[]):
                probe.return_value.__enter__.return_value.getsockname.return_value = ("192.168.1.12", 30000)
                self.assertEqual(feedback_urls(SimpleNamespace(url="ws://127.0.0.1:8000/ws")),
                                 ["http://192.168.1.12:8000/feedback/"])


class AudienceQueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.experiment = small_experiment()
        self.frame = self.experiment.tick()
        self.hub = AudienceFeedback()

    def command(self, value=1, **extra):
        return {"id": "visitor-1", "feedback": value, "fly": 0, "move": self.frame["move"], **extra}

    async def apply(self, command):
        result = asyncio.create_task(self.hub.submit(command, self.experiment))
        await asyncio.sleep(0)
        await asyncio.to_thread(self.hub.apply, self.experiment)
        return await result

    async def test_two_visitors_receive_their_own_receipts(self):
        positive = asyncio.create_task(self.hub.submit(self.command(1), self.experiment))
        negative = asyncio.create_task(self.hub.submit(self.command(-1, id="visitor-2"), self.experiment))
        await asyncio.sleep(0)
        await asyncio.to_thread(self.hub.apply, self.experiment)
        receipts = await asyncio.gather(positive, negative)
        self.assertEqual([r["value"] for r in receipts], [1, -1])
        self.assertEqual(self.experiment.feedback.positive, 1)
        self.assertEqual(self.experiment.feedback.negative, 1)

    async def test_positive_and_negative_change_the_shared_readout(self):
        for value in (1, -1):
            before = self.experiment.policy().weight.clone()
            receipt = await self.apply(self.command(value))
            self.assertEqual(receipt["status"], "applied")
            self.assertFalse(torch.equal(before, self.experiment.policy().weight))
            self.assertEqual(receipt["move"], self.frame["move"])

    async def test_mode_change_stale_move_and_invalid_values_are_rejected(self):
        for command in (self.command(0), self.command(True), self.command(2),
                        self.command(move=99999), self.command(fly=-1), self.command(policy="trained")):
            before = self.experiment.policy().weight.clone()
            self.assertEqual((await self.apply(command))["status"], "rejected")
            torch.testing.assert_close(before, self.experiment.policy().weight)
        self.experiment.handle({"learning": "reset"})
        self.assertEqual((await self.apply(self.command()))["status"], "rejected")

    async def test_paused_manual_and_not_learning_do_not_accept_feedback(self):
        self.experiment.paused = True
        self.assertEqual((await self.apply(self.command()))["status"], "rejected")
        self.experiment.paused = False
        self.experiment.override = {"food_L": 1}
        self.assertEqual((await self.apply(self.command()))["status"], "rejected")
        self.experiment.override = None
        self.experiment.policy_name = "trained"
        self.assertEqual((await self.apply(self.command()))["status"], "rejected")

    async def test_cancelled_request_cannot_train_later(self):
        task = asyncio.create_task(self.hub.submit(self.command(), self.experiment))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await asyncio.to_thread(self.hub.apply, self.experiment)
        self.assertEqual(self.experiment.feedback.positive, 0)


class AudienceSocketTests(unittest.IsolatedAsyncioTestCase):
    async def test_phone_page_and_socket_reach_host_experiment(self):
        experiment = small_experiment()
        host_socket = socket.socket()
        host_socket.bind(("127.0.0.1", 0))
        port = host_socket.getsockname()[1]
        host_socket.listen()
        service = uvicorn.Server(uvicorn.Config(server.app, log_level="error", lifespan="off"))
        # Use the actual registered routes and the actual Experiment.tick feedback hook.
        with patch.object(server, "experiment", experiment):
            running = asyncio.create_task(service.serve(sockets=[host_socket]))
            try:
                for _ in range(100):
                    if service.started:
                        break
                    await asyncio.sleep(.01)
                self.assertTrue(service.started)
                def read_page():
                    with urlopen(f"http://127.0.0.1:{port}/feedback/") as response:
                        return response.read().decode()
                self.assertIn("Help the fly learn.", await asyncio.to_thread(read_page))
                async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as host, \
                        websockets.connect(f"ws://127.0.0.1:{port}/feedback/ws") as phone:
                    frame = experiment.tick()
                    await server.audience.publish(frame)
                    state = json.loads(await phone.recv())["frame"]
                    self.assertEqual(state["move"], frame["move"])
                    self.assertNotIn("values", state)
                    await phone.send(json.dumps({"id": "phone-1", "feedback": 1, "fly": 0, "move": state["move"]}))
                    for _ in range(100):
                        if server.audience.pending:
                            break
                        await asyncio.sleep(.01)
                    updated = await asyncio.to_thread(experiment.tick)
                    receipt = json.loads(await phone.recv())
                    self.assertEqual(receipt["id"], "phone-1")
                    self.assertEqual(receipt["receipt"]["status"], "applied")
                    self.assertEqual(updated["learning"]["feedback"]["positive"], 1)
                    # The host's normal stream observes that same receipt and counter.
                    for client in list(server.clients):
                        await client.send_json(updated)
                    self.assertEqual(json.loads(await host.recv())["learning"]["feedback"]["last"], receipt["receipt"])
                    await server.audience.publish(paused=True)
                    self.assertTrue(json.loads(await phone.recv())["frame"]["paused"])
                    await phone.send(json.dumps({"id": "bad", "feedback": 1, "fly": 0, "move": state["move"], "learning": "reset"}))
                    self.assertEqual(json.loads(await phone.recv())["receipt"]["status"], "rejected")
            finally:
                service.should_exit = True
                await running
                host_socket.close()
                server.audience.latest = None
