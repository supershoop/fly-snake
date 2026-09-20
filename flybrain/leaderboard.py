"""Human leaderboard for the versus layout. One round = one human life; the fly's score in the same round is kept next to it.
Stored in outputs/leaderboard.json (gitignored), so it survives server restarts on the demo machine."""
import json
import time
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "outputs" / "leaderboard.json"
NAME_LENGTH, KEPT, SHOWN = 16, 200, 10


def clean_name(name: object) -> str:
    text = "".join(ch for ch in str(name) if ch.isprintable()).strip()[:NAME_LENGTH]
    return text or "anonymous"


class Leaderboard:
    def __init__(self, path: Path = PATH):
        self.path, self.rounds = path, []
        try:
            self.rounds = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass

    def record(self, name: str, human: int, fly: int, brain: str):
        """brain = which fly the human played against, e.g. 'instinct / real wiring'."""
        self.rounds.append({"name": clean_name(name), "human": int(human), "fly": int(fly), "brain": brain, "when": int(time.time())})
        self.rounds = sorted(self.rounds, key=lambda r: (-r["human"], r["when"]))[:KEPT]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.rounds), encoding="utf-8")
        except OSError:
            pass  # a read-only disk must not stop the game

    def top(self) -> list[dict]:
        return self.rounds[:SHOWN]

    def tally(self) -> dict:
        """How the humans are doing against the flies overall."""
        return {"rounds": len(self.rounds), "humanWins": sum(r["human"] > r["fly"] for r in self.rounds),
                "flyWins": sum(r["fly"] > r["human"] for r in self.rounds)}
