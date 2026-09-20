"""Send HC-SR04 threat readings to the Fly Snake hardware WebSocket."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import time
from contextlib import suppress

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from .sensor import HCSR04DistanceSource, MockDistanceSource, ThreatConfig, ThreatEstimator

LOG = logging.getLogger("fly-snake-hardware")
DEFAULT_URL = "ws://localhost:8000/ws/hardware"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("FLY_BRAIN_HARDWARE_WS", DEFAULT_URL))
    parser.add_argument("--trigger-pin", type=int, default=23, help="HC-SR04 trigger BCM pin")
    parser.add_argument("--echo-pin", type=int, default=24, help="HC-SR04 echo BCM pin (through a voltage divider)")
    parser.add_argument("--near-cm", type=float, default=8.0, help="distance that represents maximum proximity")
    parser.add_argument("--far-cm", type=float, default=100.0, help="distance at which proximity becomes zero")
    parser.add_argument("--max-approach-cm-s", type=float, default=150.0, help="approach speed treated as maximum urgency")
    parser.add_argument("--rate", type=float, default=10.0, help="messages per second (must remain at least 5)")
    parser.add_argument("--mock", action="store_true", help="send a synthetic approach pattern without GPIO")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


async def publish(url: str, source: object, estimator: ThreatEstimator, rate: float) -> None:
    """Reconnect forever and publish only the latest sensor sample."""
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be finite and positive")
    interval = 1.0 / rate
    while True:
        try:
            LOG.info("connecting to %s", url)
            async with connect(url, open_timeout=5, ping_interval=20, ping_timeout=20) as websocket:
                LOG.info("connected")
                deadline = time.monotonic()
                while True:
                    distance = await asyncio.to_thread(source.read_metres)
                    danger = estimator.update(distance)
                    message = {"sensor": {"danger_ahead": round(danger, 4)}}
                    await websocket.send(json.dumps(message, separators=(",", ":")))
                    LOG.debug("distance=%s m danger_ahead=%.4f", distance, danger)
                    deadline += interval
                    await asyncio.sleep(max(0.0, deadline - time.monotonic()))
        except asyncio.CancelledError:
            raise
        except (ConnectionClosed, OSError, TimeoutError) as error:
            LOG.warning("connection lost (%s); retrying in 2 seconds", error)
            estimator = ThreatEstimator(estimator.config)
            await asyncio.sleep(2.0)


async def main() -> None:
    args = parse_args()
    if not math.isfinite(args.rate) or args.rate < 5:
        raise SystemExit("--rate must be finite and at least 5 Hz because server readings expire after 0.6 seconds")
    config = ThreatConfig(
        near_metres=args.near_cm / 100.0,
        far_metres=args.far_cm / 100.0,
        max_approach_metres_per_second=args.max_approach_cm_s / 100.0,
    )
    source = MockDistanceSource(config.near_metres, config.far_metres) if args.mock else HCSR04DistanceSource(
        trigger_pin=args.trigger_pin,
        echo_pin=args.echo_pin,
        max_distance_metres=config.far_metres,
    )
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        await publish(args.url, source, ThreatEstimator(config), args.rate)
    finally:
        with suppress(Exception):
            source.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
