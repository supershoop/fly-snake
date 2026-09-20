"""HC-SR04 distance sampling and distance-to-threat conversion."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol


class DistanceSource(Protocol):
    """A source returning distance in metres, or None when no echo is available."""

    def read_metres(self) -> float | None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ThreatConfig:
    near_metres: float = 0.08
    far_metres: float = 1.0
    max_approach_metres_per_second: float = 1.5
    approach_weight: float = 0.35
    smoothing: float = 0.35

    def __post_init__(self) -> None:
        values = (
            self.near_metres,
            self.far_metres,
            self.max_approach_metres_per_second,
            self.approach_weight,
            self.smoothing,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("threat configuration values must be finite")
        if self.near_metres < 0 or self.far_metres <= self.near_metres:
            raise ValueError("far_metres must be greater than near_metres >= 0")
        if self.max_approach_metres_per_second <= 0:
            raise ValueError("max approach speed must be positive")
        if not 0 <= self.approach_weight <= 1 or not 0 < self.smoothing <= 1:
            raise ValueError("weights must be between 0 and 1")


class ThreatEstimator:
    """Combine proximity and positive approach speed into a smoothed 0..1 value."""

    def __init__(self, config: ThreatConfig = ThreatConfig()) -> None:
        self.config = config
        self._previous: tuple[float, float] | None = None
        self._smoothed = 0.0

    def update(self, distance_metres: float | None, now: float | None = None) -> float:
        timestamp = time.monotonic() if now is None else now
        if distance_metres is None or not math.isfinite(distance_metres) or distance_metres < 0:
            self._previous = None
            self._smoothed = 0.0
            return 0.0

        span = self.config.far_metres - self.config.near_metres
        proximity = _clamp((self.config.far_metres - distance_metres) / span)
        approach = 0.0
        if self._previous is not None:
            previous_distance, previous_time = self._previous
            elapsed = timestamp - previous_time
            if elapsed > 0:
                speed = max(0.0, (previous_distance - distance_metres) / elapsed)
                approach = _clamp(speed / self.config.max_approach_metres_per_second)
        self._previous = (distance_metres, timestamp)

        # Proximity remains the baseline; approach speed increases urgency without
        # allowing a distant stationary object to produce a threat response.
        raw = proximity + self.config.approach_weight * approach * (1.0 - proximity)
        alpha = self.config.smoothing
        self._smoothed = alpha * raw + (1.0 - alpha) * self._smoothed
        return _clamp(self._smoothed)


class HCSR04DistanceSource:
    """Read an HC-SR04 through gpiozero using BCM pin numbering."""

    def __init__(self, trigger_pin: int, echo_pin: int, max_distance_metres: float) -> None:
        try:
            from gpiozero import DistanceSensor
        except ImportError as error:
            raise RuntimeError("gpiozero is required for the HC-SR04; install hardware/requirements.txt") from error
        self._sensor = DistanceSensor(
            echo=echo_pin,
            trigger=trigger_pin,
            max_distance=max_distance_metres,
            queue_len=3,
        )

    def read_metres(self) -> float | None:
        distance = float(self._sensor.distance)
        return distance if math.isfinite(distance) else None

    def close(self) -> None:
        self._sensor.close()


class MockDistanceSource:
    """A repeating hand-approach pattern for testing without GPIO hardware."""

    def __init__(self, near_metres: float = 0.08, far_metres: float = 1.0) -> None:
        self.near_metres = near_metres
        self.far_metres = far_metres
        self._started = time.monotonic()

    def read_metres(self) -> float:
        phase = (time.monotonic() - self._started) % 6.0
        fraction = phase / 3.0 if phase < 3.0 else (6.0 - phase) / 3.0
        return self.far_metres - fraction * (self.far_metres - self.near_metres)

    def close(self) -> None:
        pass


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
