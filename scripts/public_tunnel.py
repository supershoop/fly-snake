"""Publish the phone controller on a public HTTPS URL, for Wi-Fi that blocks device-to-device traffic.

Runs cloudflared against the loopback audience gateway, then writes the public URL to
.feedback-url so the dashboard QR code points at it. The brain keeps running untouched:
no restart, no lost live learner. The URL is removed again on exit so a dead tunnel is
never advertised.

Run (three terminals):
  .venv/bin/python -m uvicorn flybrain.server:app --port 8000
  .venv/bin/python scripts/audience_gateway.py --brain-ws ws://127.0.0.1:8000/feedback/ws --port 8002
  .venv/bin/python scripts/public_tunnel.py --port 8002
"""
import argparse
import contextlib
import re
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

URL_FILE = Path(__file__).resolve().parents[1] / ".feedback-url"
QUICK_TUNNEL = re.compile(rb"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8002, help="local audience gateway port to expose")
    parser.add_argument("--path", default="/feedback/", help="path appended to the public origin for the QR code")
    parser.add_argument("--url-file", type=Path, default=URL_FILE)
    args = parser.parse_args()

    cloudflared = shutil.which("cloudflared")
    if not cloudflared:
        return fail("cloudflared is not installed. Install it (brew install cloudflared) or set "
                    "FLY_FEEDBACK_URL to a public https URL you already have.")

    tunnel = subprocess.Popen(
        [cloudflared, "tunnel", "--url", f"http://127.0.0.1:{args.port}", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    found = threading.Event()

    def publish():
        """Mirror cloudflared's output and capture the first quick-tunnel URL it prints."""
        for line in tunnel.stdout:
            sys.stderr.buffer.write(line)
            sys.stderr.buffer.flush()
            match = None if found.is_set() else QUICK_TUNNEL.search(line)
            if match:
                url = match.group().decode() + args.path
                args.url_file.write_text(url + "\n")
                found.set()
                print(f"\nPhone controller: {url}\nThe dashboard QR code now points here; reload the page if it is open.\n",
                      flush=True)

    reader = threading.Thread(target=publish, daemon=True)
    reader.start()
    try:
        if not found.wait(timeout=40) and tunnel.poll() is None:
            print("No tunnel URL yet; still waiting. Leave this running.", flush=True)
        return tunnel.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        # Never leave a URL advertised for a tunnel that is gone.
        if tunnel.poll() is None:
            tunnel.send_signal(signal.SIGINT)
            with contextlib.suppress(subprocess.TimeoutExpired):
                tunnel.wait(timeout=10)
            if tunnel.poll() is None:
                tunnel.kill()
        with contextlib.suppress(OSError):
            args.url_file.unlink()
        print("Tunnel closed; the QR code falls back to local addresses.", flush=True)


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
