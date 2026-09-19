"""Export genuine MaleCNS soma positions. Run with: uv run --with pyarrow python scripts/build-brain-atlas.py SOURCE.feather"""
import argparse
import math
import array
import hashlib
import json
from pathlib import Path
import sys
import pyarrow.feather as feather

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source', type=Path)
parser.add_argument('--check', action='store_true', help='Compare the checked-in export with the pinned source without writing files')
args = parser.parse_args()
source = args.source
expected_sha256 = '2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2'
source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
if source_sha256 != expected_sha256:
 raise ValueError('Source differs from the pinned MaleCNS v1.0 annotation file')
out = Path(__file__).resolve().parents[1] / 'public/data/brain-atlas'
if not args.check: out.mkdir(parents=True, exist_ok=True)
rows = feather.read_table(source, columns=['bodyId', 'somaLocation', 'superclass', 'somaNeuromere', 'status']).to_pylist()
classes = {
 'optic': ['ol_intrinsic', 'ol_sensory', 'visual_projection', 'visual_centrifugal', 'visual_projection_tbc'],
 'central': ['cb_intrinsic', 'cb_sensory', 'cb_motor', 'cb_endocrine', 'cb_sensory_tbc', 'cb_efferent'],
 'descending': ['descending_neuron', 'descending_neuron_tbc', 'sensory_descending', 'efferent_descending'],
 'vnc': ['vnc_intrinsic', 'vnc_sensory', 'vnc_motor', 'vnc_efferent', 'vnc_tbc', 'vnc_sensory_tbc', 'vnc_endocrine', 'ascending_neuron', 'sensory_ascending', 'sensory_ascending_tbc', 'efferent_ascending'],
}
labels = list(classes) + ['other']
colors = ['#81b5c8', '#cfcac0', '#dfb672', '#af9bc3', '#6c7777']
points = []
for row in rows:
 if row['status'] != 'Traced' or not row['somaLocation']: continue
 group = next((k for k, values in classes.items() if row['superclass'] in values), 'other')
 loc = row['somaLocation']
 if len(loc) != 3 or not all(math.isfinite(v) for v in loc):
  raise ValueError(f"Invalid source coordinates for body {row['bodyId']}")
 points.append((row['bodyId'], *loc, labels.index(group)))
points.sort(key=lambda p: p[0])
positions = array.array('f', [v for p in points for v in p[1:4]])
ids = array.array('I', [p[0] for p in points])
if sys.byteorder != 'little': positions.byteswap(); ids.byteswap()
if len(set(ids)) != len(ids): raise ValueError('Duplicate body IDs')
if list(positions) != [v for p in points for v in p[1:4]]:
 raise ValueError('Float32 export would alter source coordinate values')
exports = {'positions.bin': positions.tobytes(), 'ids.bin': ids.tobytes(), 'groups.bin': bytes(p[4] for p in points)}
meta = {
 'dataset':'male-cns:v1.0', 'count':len(points), 'totalAnnotations':len(rows),
 'tracedWithoutSoma':sum(r['status']=='Traced' and not r['somaLocation'] for r in rows),
 'selection':'All Traced annotations with measured somaLocation. No invented positions. The renderer includes only optic, central and descending groups; VNC-associated and unclassified somata are not shown.',
 'coordinateUnits':'8 nm voxels, native MaleCNS EM space', 'byteOrder':'little-endian',
 'files':{'positions':'positions.bin','ids':'ids.bin','groups':'groups.bin'},
 'groups':[{'id':i,'key':k,'color':colors[i],'count':sum(p[4]==i for p in points)} for i,k in enumerate(labels)],
 'source':'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather',
 'sourceSha256':source_sha256,
 'exportSha256':{name: hashlib.sha256(data).hexdigest() for name, data in exports.items()},
 'groupSuperclasses':classes,
 'reference':'https://male-cns.janelia.org/download/',
 'displayTransform':'Native (x, y, z) to (x, -y, -z); bounding-box centering; one uniform scale 5/max(axis extents); rigid view rotation. Orthographic projection. No per-axis normalization.',
 'pointMeaning':'One curated somaLocation per Traced body ID; point size is a display marker, not anatomical cell size.',
 'limitations':'Soma locations only, not neurites or the synaptic connectivity graph. Brain selection is superclass-based, not a complete anatomical brain segmentation. The video/game overlay is illustrative and not measured or predicted neural activity.',
 'license':'CC BY 4.0', 'attribution':'FlyEM / HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology, and Google Research',
 'activity':'Anatomy only. No measured or predicted neural activity is bundled.'
}
brain = [p for p in points if p[4] < labels.index('vnc')]
meta['brainCount'] = len(brain)
meta['brainView'] = 'Classified optic, central and descending somata only; excludes VNC-associated and unclassified somata.'
meta['brainNativeBounds'] = {'min':[min(p[i] for p in brain) for i in (1,2,3)], 'max':[max(p[i] for p in brain) for i in (1,2,3)]}
exports['manifest.json'] = (json.dumps(meta, indent=2)+'\n').encode()
notice = '\n\n'.join([
 '# MaleCNS v1.0 soma atlas',
 'Dataset creators: '+meta['attribution']+'.',
 'License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). No endorsement of this application is implied.',
 'Scientific project, publication and downloads: https://male-cns.janelia.org/download/',
 'Source: '+meta['source'],
 'Pinned source SHA-256: `'+source_sha256+'`',
 meta['selection'],
 f"The export contains {meta['count']:,} measured positions. The current brain view shows {meta['brainCount']:,} classified somata. {meta['tracedWithoutSoma']:,} Traced annotations without soma locations are omitted. Positions are never invented or substituted with tosomaLocation.",
 'Changes: filtering, body-ID sorting, and lossless float32/uint32 binary export. The display centers, rigidly rotates, uniformly scales and colors the points. Source coordinates and IDs remain unchanged in the binaries. Marker size is not anatomical soma size.',
 meta['limitations'],
 'Reproduce: `uv run --with pyarrow python scripts/build-brain-atlas.py SOURCE.feather`. Audit without writing: append `--check`. The audit requires the exact source hash and compares every output byte, including all positions and body IDs.'
])+'\n'
exports['NOTICE.md'] = notice.encode()
for name, data in exports.items():
 if args.check:
  if (out/name).read_bytes() != data: raise ValueError(f'Export mismatch: {name}')
 else:
  (out/name).write_bytes(data)
print(json.dumps({'mode':'verified' if args.check else 'exported','count':meta['count'],'brainCount':meta['brainCount'],'sourceSha256':source_sha256,'exportSha256':meta['exportSha256']}))
