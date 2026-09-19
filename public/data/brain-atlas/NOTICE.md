# MaleCNS v1.0 soma atlas

Dataset creators: FlyEM / HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology, and Google Research.

License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). No endorsement of this application is implied.

Scientific project, publication and downloads: https://male-cns.janelia.org/download/

Source: https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather

Pinned source SHA-256: `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2`

All Traced annotations with measured somaLocation. No invented positions. The renderer includes only optic, central and descending groups; VNC-associated and unclassified somata are not shown.

The export contains 140,024 measured positions. The current brain view shows 124,289 classified somata. 25,098 Traced annotations without soma locations are omitted. Positions are never invented or substituted with tosomaLocation.

Changes: filtering, body-ID sorting, and lossless float32/uint32 binary export. The display centers, rigidly rotates, uniformly scales and colors the points. Source coordinates and IDs remain unchanged in the binaries. Marker size is not anatomical soma size.

Soma locations only, not neurites or the synaptic connectivity graph. Brain selection is superclass-based, not a complete anatomical brain segmentation. The video/game overlay is illustrative and not measured or predicted neural activity.

Reproduce: `uv run --with pyarrow python scripts/build-brain-atlas.py SOURCE.feather`. Audit without writing: append `--check`. The audit requires the exact source hash and compares every output byte, including all positions and body IDs.
