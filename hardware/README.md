# Ultrasonic hardware client

This standalone Raspberry Pi client reads an HC-SR04 at 10 Hz, combines proximity and approach speed into a `danger_ahead` value from 0 to 1, and sends it to the brain server's dedicated `ws://HOST:8000/ws/hardware` endpoint. It contains no RFID behavior.

## Wiring

GPIO pin numbers below use BCM numbering. Defaults are trigger GPIO 23 and echo GPIO 24.

**The HC-SR04 echo output is 5 V. Never connect it directly to a Raspberry Pi GPIO.** Put a voltage divider (for example, 1 kΩ from Echo to GPIO and 2 kΩ from GPIO to ground) between Echo and the Pi's 3.3 V input. Connect grounds together and power the sensor from 5 V.

## Install and run on the Pi

```sh
python3 -m venv .venv
.venv/bin/pip install -r hardware/requirements.txt
.venv/bin/python -m hardware.client --url ws://BRAIN_SERVER_IP:8000/ws/hardware
```

The brain server must listen on the network:

```sh
.venv/bin/python -m uvicorn flybrain.server:app --port 8000 --host 0.0.0.0
```

Keep the web UI connected to `/ws`: hardware sockets do not receive simulation frames and do not start simulation ticks by themselves. The server applies each hardware reading to every simulated fly, and drops it after 0.6 seconds if updates stop.

Use a synthetic six-second approach/retreat cycle when developing without GPIO:

```sh
.venv/bin/python -m hardware.client --mock --url ws://localhost:8000/ws/hardware --verbose
```

`FLY_BRAIN_HARDWARE_WS` can set the URL instead of `--url`. Calibration options include `--near-cm`, `--far-cm`, `--max-approach-cm-s`, and `--rate`; the rate is rejected below 5 Hz. Stop the client with Ctrl-C.
