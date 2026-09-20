"""Validation for messages received from physical sensors."""

import math


class HardwareMessageError(ValueError):
    """Raised when a hardware WebSocket message violates the protocol."""


def parse_hardware_message(message: object) -> dict[str, dict[str, float]]:
    """Validate and normalize an ultrasonic sensor message.

    The hardware endpoint intentionally accepts only the RFID-independent
    ``danger_ahead`` channel.
    """
    if not isinstance(message, dict) or set(message) != {"sensor"}:
        raise HardwareMessageError("expected an object containing only 'sensor'")

    sensor = message["sensor"]
    if not isinstance(sensor, dict) or set(sensor) != {"danger_ahead"}:
        raise HardwareMessageError("sensor must contain only 'danger_ahead'")

    value = sensor["danger_ahead"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HardwareMessageError("danger_ahead must be a number")

    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise HardwareMessageError("danger_ahead must be finite and between 0 and 1")

    return {"sensor": {"danger_ahead": value}}
