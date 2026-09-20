"""Expose just the phone controller through an HTTPS tunnel, using the existing live brain.

Run this on loopback, then point cloudflared at its port. The host's /ws controls
and other brain-server routes are deliberately absent from this gateway.
"""
import argparse
import asyncio
import contextlib
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
import uvicorn
import websockets

PUBLIC = Path(__file__).resolve().parents[1] / "public" / "feedback"


def create_app(brain_ws):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/")
    async def root():
        return RedirectResponse("/feedback/")

    @app.get("/feedback/")
    async def page():
        return FileResponse(PUBLIC / "index.html")

    @app.get("/feedback/{asset}")
    async def asset(asset: str):
        if asset not in ("style.css", "controller.js"):
            raise HTTPException(404)
        media = "text/css" if asset == "style.css" else "text/javascript"
        return FileResponse(PUBLIC / asset, media_type=media)

    @app.websocket("/feedback/ws")
    async def feedback(phone: WebSocket):
        tasks = []
        try:
            async with websockets.connect(brain_ws, max_size=2**20, open_timeout=5) as brain:
                await phone.accept()

                async def to_brain():
                    while True:
                        message = await phone.receive_text()
                        if len(message) > 2048:
                            await phone.close(code=1009)
                            return
                        await brain.send(message)

                async def to_phone():
                    async for message in brain:
                        await phone.send_text(message)

                tasks = [asyncio.create_task(to_brain()), asyncio.create_task(to_phone())]
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
        except (WebSocketDisconnect, websockets.ConnectionClosed, OSError, asyncio.TimeoutError):
            pass
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            with contextlib.suppress(Exception):
                await phone.close(code=1013)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-ws", default="ws://127.0.0.1:8000/feedback/ws")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    uvicorn.run(create_app(args.brain_ws), host="127.0.0.1", port=args.port)
