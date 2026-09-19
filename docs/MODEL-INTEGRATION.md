# Connecting your model

The template is a viewer and integration starting point, not a neural model.
Supply your own training, visual encoding, connectivity and motor model.

## Replay format

```json
{
  "version": 1,
  "dataset": "male-cns:v1.0",
  "source": {
    "kind": "predicted",
    "name": "Your model and checkpoint identifier",
    "normalization": "Firing rate / 50 Hz, clamped to [0, 1]"
  },
  "frames": [
    {"time": 0, "values": []},
    {"time": 1, "values": []}
  ]
}
```

Each values entry is `[MaleCNS_bodyId, normalized_value]`. The empty arrays above
are intentional: use IDs from your model and the bundled `ids.bin`, not invented
IDs. IDs must belong to the visible brain selection (`groups.bin` value < 3).
The loader rejects an incompatible dataset, unknown/non-visible IDs, duplicate
IDs, invalid values and non-increasing timestamps. Maximum upload is 10 MB;
there must be 2–10,000 frames starting at 0 seconds. An omitted ID is zero in
that frame, not an incremental update. Time samples are held until replaced.

`source.kind` is `synthetic`, `predicted`, or `measured`. Be precise: simulated
firing is predicted/simulated, not measured in a biological fly. Declare a
fixed, interpretable normalization. Document checkpoint, experiment protocol,
training data, units and limitations in your own project. The viewer only
validates the file structure, not these scientific claims.

## Producing data from Python

The binary atlas uses little-endian float32 xyz positions, uint32 body IDs and
uint8 group IDs in matching row order. Decode with NumPy if your model uses it:

```python
import json
from pathlib import Path
import numpy as np

root = Path("public/data/brain-atlas")
ids = np.fromfile(root / "ids.bin", dtype="<u4")
groups = np.fromfile(root / "groups.bin", dtype="u1")
brain_ids = ids[groups < 3]

# Your simulator supplies body_ids, times_s and rates_hz. Reorder by body ID,
# never by the viewer's screen positions. Do not substitute human-brain outputs.
# values = np.clip(rates_hz / 50.0, 0, 1)
# frames = [{"time": float(t), "values": [[int(i), float(v)]
#           for i, v in zip(body_ids, row)]} for t, row in zip(times_s, values)]
```

The runnable synthetic fixture in `public/examples/model-output.example.json`
uses real IDs with authored demonstration values. It is not stimulus-driven,
trained or biologically inferred.

## Live adapter

`BrainScene` accepts `frame: ActivityFrame | null`. For a live adapter, validate
server messages against the same ID set and update this prop at your desired
sampling rate. Keep one stimulus clock, associate each output frame with that
clock, and expose source kind, model identity, normalization and connection
status in the UI. Clear stale activity on disconnect. Keep secrets on your own
backend, not in browser environment variables. The template does not silently
invent data if the model is absent.

Implement your own environment observation and action contracts. Connect a
Gymnasium/PyTorch/MuJoCo process through a separate service or offline exports.
The web viewer and display-only Flybody mesh are not replacements for a physics
or reinforcement-learning environment.
