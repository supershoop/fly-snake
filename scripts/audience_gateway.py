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
OPERATOR_UI = Path(__file__).resolve().parents[1] / "flybrain" / "operator_ui" / "index.html"


def create_app(brain_ws, operator_ws=None):
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

    async def proxy(phone: WebSocket, upstream):
        tasks = []
        close_code = 1000
        try:
            async with websockets.connect(upstream, max_size=2**20, open_timeout=5) as brain:
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
        except websockets.ConnectionClosed as error:
            close_code = error.rcvd.code if error.rcvd and error.rcvd.code in (1000, 1008, 1009) else 1013
        except WebSocketDisconnect:
            pass
        except (OSError, asyncio.TimeoutError, websockets.exceptions.InvalidStatus):
            close_code = 1013
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            with contextlib.suppress(Exception):
                await phone.close(code=close_code)

    @app.websocket("/feedback/ws")
    async def feedback(phone: WebSocket):
        await proxy(phone, brain_ws)

    if operator_ws:
        @app.get("/operator/")
        async def operator_page():
            return FileResponse(OPERATOR_UI, headers={"Cache-Control": "no-store", "X-Robots-Tag": "noindex, nofollow", "Referrer-Policy": "no-referrer"})

        @app.websocket("/operator/ws")
        async def operator(phone: WebSocket):
            # The brain authenticates the first message; this gateway knows no keys.
            await proxy(phone, operator_ws)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-ws", default="ws://127.0.0.1:8000/feedback/ws")
    parser.add_argument("--operator-ws", help="Opt in to the private, key-protected operator page and socket")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    uvicorn.run(create_app(args.brain_ws, args.operator_ws), host="127.0.0.1", port=args.port)
