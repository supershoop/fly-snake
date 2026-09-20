"""Exercise the tunnel gateway without a brain or external tunnel service."""
import asyncio
import importlib.util
import json
from pathlib import Path
import socket
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

import uvicorn
import websockets

spec = importlib.util.spec_from_file_location("audience_gateway", Path(__file__).resolve().parents[1] / "scripts/audience_gateway.py")
gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway)


class AudienceGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_controller_assets_and_feedback_socket_are_exposed(self):
        received = []

        async def brain(connection):
            received.append(connection.request.path)
            await connection.send(json.dumps({"frame": {"move": 12}}))
            async for message in connection:
                received.append(json.loads(message))
                await connection.send(json.dumps({"id": "phone", "receipt": {"status": "applied"}}))

        async with websockets.serve(brain, "127.0.0.1", 0) as upstream:
            brain_port = upstream.sockets[0].getsockname()[1]
            app = gateway.create_app(f"ws://127.0.0.1:{brain_port}/feedback/ws")
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            listener.listen()
            service = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
            running = asyncio.create_task(service.serve(sockets=[listener]))
            try:
                for _ in range(100):
                    if service.started:
                        break
                    await asyncio.sleep(.01)
                self.assertTrue(service.started)

                def get(path):
                    try:
                        with urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
                            return response.status, response.read().decode()
                    except HTTPError as error:
                        return error.code, ""

                for path in ("/feedback/", "/feedback/controller.js", "/feedback/style.css"):
                    code, body = await asyncio.to_thread(get, path)
                    self.assertEqual(code, 200)
                    self.assertTrue(body)
                for path in ("/ws", "/docs", "/openapi.json", "/feedback/README.md"):
                    self.assertEqual((await asyncio.to_thread(get, path))[0], 404)
                with self.assertRaises(websockets.exceptions.InvalidStatus):
                    async with websockets.connect(f"ws://127.0.0.1:{port}/ws"):
                        self.fail("Host controls must not be public")
                async with websockets.connect(f"ws://127.0.0.1:{port}/feedback/ws") as phone:
                    self.assertEqual(json.loads(await phone.recv())["frame"]["move"], 12)
                    command = {"id": "phone", "feedback": -1, "fly": 0, "move": 12}
                    await phone.send(json.dumps(command))
                    self.assertEqual(json.loads(await phone.recv())["receipt"]["status"], "applied")
                    self.assertEqual(received, ["/feedback/ws", command])
            finally:
                service.should_exit = True
                await running
                listener.close()
