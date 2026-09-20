"""Thermal guard: keep the laptop alive during long GPU runs without getting in the way of a demo.

Every few seconds the GPU temperature is read with nvidia-smi and appended to outputs/gpu-temps.csv (flushed line by line, so
the last readings survive a hard power-off). When the GPU runs hot the simulation first leaves short gaps between moves
(state "slow": the demo keeps running, the GPU gets a lower duty cycle); only close to the hardware limit does it hold still
until the GPU has cooled (state "cooling", shown as a banner on the page).

Environment:
  FLY_THERMAL_GUARD   on (default) | slow (never pauses, only slows) | off (only logs)
  FLY_THERMAL_SLOW    degrees C to start slowing   (default 80)
  FLY_THERMAL_PAUSE   degrees C to hold still      (default 87)
  FLY_THERMAL_RESUME  degrees C to resume a hold   (default 78)
Only the GPU is watched; Windows does not expose CPU temperature reliably.
"""
import asyncio
import os
import time
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "outputs" / "gpu-temps.csv"
QUERY = ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,power.draw", "--format=csv,noheader,nounits"]
INTERVAL_S = 5.0
SLOW_GAP_S = 0.35  # extra pause between moves while "slow"


class ThermalGuard:
    def __init__(self):
        self.mode = os.environ.get("FLY_THERMAL_GUARD", "on").lower()
        self.slow_at = float(os.environ.get("FLY_THERMAL_SLOW", 80))
        self.pause_at = float(os.environ.get("FLY_THERMAL_PAUSE", 87))
        self.resume_at = float(os.environ.get("FLY_THERMAL_RESUME", 78))
        self.gpu: float | None = None
        self.state = "off" if self.mode == "off" else "ok"

    def update(self, temperature: float | None):
        self.gpu = temperature
        if self.mode == "off" or temperature is None:
            self.state = "off" if self.mode == "off" else "ok"
        elif self.state == "cooling":
            if temperature <= self.resume_at:
                self.state = "ok"
        elif self.mode == "on" and temperature >= self.pause_at:
            self.state = "cooling"
        else:
            self.state = "slow" if temperature >= self.slow_at else "ok"

    @property
    def cooling(self) -> bool:
        return self.state == "cooling"

    def gap(self) -> float:
        return SLOW_GAP_S if self.state == "slow" else 0.0

    def status(self) -> dict:
        return {"gpu": self.gpu, "state": self.state, "slowAt": self.slow_at, "pauseAt": self.pause_at}

    async def watch(self, describe=lambda: ""):
        """Runs forever. `describe` returns a short note for the log, e.g. the current layout."""
        LOG.parent.mkdir(exist_ok=True)
        new = not LOG.exists()
        with open(LOG, "a", buffering=1) as log:
            if new:
                log.write("time,gpu_c,gpu_util_pct,power_w,state,note\n")
            while True:
                try:
                    process = await asyncio.create_subprocess_exec(*QUERY, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
                    output, _ = await asyncio.wait_for(process.communicate(), 4)
                    temperature, utilisation, power = (part.strip() for part in output.decode().splitlines()[0].split(","))
                    self.update(float(temperature))
                    log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{temperature},{utilisation},{power},{self.state},{describe()}\n")
                except Exception:  # no NVIDIA GPU, nvidia-smi missing or busy: the guard simply stays out of the way
                    self.update(None)
                await asyncio.sleep(INTERVAL_S)
