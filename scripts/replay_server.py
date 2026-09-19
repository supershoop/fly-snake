"""Serve a recorded session over the same WebSocket as the real brain server, so the web page and hardware can be
developed without a GPU. Record with FLY_RECORD=frames.jsonl on the real server.

Run: .venv/Scripts/python scripts/replay_server.py frames.jsonl   (then open the web page as usual)
"""
import asyncio
import sys

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

frames = open(sys.argv[1]).read().splitlines()
app = FastAPI()


@app.websocket("/ws")
async def socket(websocket: WebSocket):
    await websocket.accept()
    try:
        index = 0
        while True:
            await websocket.send_text(frames[index % len(frames)])
            index += 1
            await asyncio.sleep(0.17)
    except WebSocketDisconnect:
        pass


uvicorn.run(app, port=8000)
