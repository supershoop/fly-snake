"""Host-selected sensory cues for game events, independent of readout rewards."""

STIMULUS_CUES = {"positive": "taste", "negative": "pain", "none": None}


class EventStimuli:
    def __init__(self):
        self.choices = {"food": "positive", "death": "negative"}

    def update(self, changes):
        """Validate a partial update completely before applying any choices."""
        if (not isinstance(changes, dict) or not changes
                or any(key not in self.choices or not isinstance(value, str) or value not in STIMULUS_CUES
                       for key, value in changes.items())):
            raise ValueError("Choose positive, negative or none for food or death.")
        self.choices.update(changes)

    def cue(self, reward):
        event = "food" if reward >= 1 else "death" if reward <= -1 else None
        return STIMULUS_CUES[self.choices[event]] if event else None

    def state(self, enabled=True):
        return dict(self.choices) if enabled else dict.fromkeys(self.choices, "none")
