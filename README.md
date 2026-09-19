<p align="center">
  <img src="assets/preview.svg" alt="fly-connectome-template: real MaleCNS anatomy and a starter for your own experiment" width="760">
</p>

<p align="center">
  A browser workbench for building your own fly-connectome experiments.
  Real anatomy, a replaceable environment, and model outputs mapped by neuron ID.
</p>

<p align="center">
  <a href="https://github.com/cobanov/fly-connectome-template/actions/workflows/ci.yml"><img alt="build" src="https://github.com/cobanov/fly-connectome-template/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Node.js" src="https://img.shields.io/badge/node-%E2%89%A522.18-527fa3?labelColor=151b22">
  <img alt="anatomy" src="https://img.shields.io/badge/anatomy-MaleCNS_v1.0-527fa3?labelColor=151b22">
  <a href="LICENSE"><img alt="licence: attribution required" src="https://img.shields.io/badge/licence-attribution_required-527fa3?labelColor=151b22"></a>
</p>

---

Start with a fly body and measured brain coordinates already on screen. Replace
the environment, connect your own model and inspect its outputs against the
same MaleCNS neuron IDs. Training and inference stay in your own stack; the
browser handles the experiment view.

- **Real anatomy.** 124,289 classified brain soma positions from MaleCNS v1.0,
  rendered without stretching the axes, plus the anatomical Flybody mesh.
- **Replaceable parts.** Environment on the left, brain above the body on the
  right. Each is a separate React component; the layout stacks on mobile.
- **An explicit model boundary.** JSON replay with timestamps, body IDs,
  normalized values and declared provenance. No neural activity is invented
  when no model is connected.
- **A small web stack.** React, TypeScript, Three.js and Vite. No required
  account, backend, database or hosting provider.

The code is **source-available with mandatory attribution** in your web UI and
repository README. Your own models and weights can remain private. See
[Licence](#licence) before reusing.

## Start

[Use this template][generate] to create your repository, then clone it.
With Node.js **22.18+**:

```sh
npm ci
npm run dev
```

Open the URL printed by Vite. The example stimulus starts automatically;
the brain initially shows anatomy only. **Load synthetic example**, then
**Play**, demonstrates the output pipeline with clearly labeled test values.
**Load model JSON** reads your own replay locally in the browser.

## Make it yours

| File | Replace or connect |
| --- | --- |
| `src/components/Environment.tsx` | Your game, video or sensory scene |
| `src/components/BrainScene.tsx` | Your model's `ActivityFrame`, keyed by MaleCNS body ID |
| `src/components/FlyScene.tsx` | Your motor decoder or physics adapter |
| `src/App.tsx` | Experiment clock, controls and replay/live adapter |
| `src/style.css` | Your layout and visual design |

The [model integration guide](docs/MODEL-INTEGRATION.md) covers the JSON format,
Python-side data layout and live adapters. The validator rejects incompatible
datasets, unknown IDs, duplicate IDs, invalid values and unordered timestamps.
It validates the format, not the scientific truth of a model's output.

## What the anatomy means

These are **cell-body positions**, not neurite branches or a synaptic graph.
The source is the adult male MaleCNS dataset, not female FlyWire. Of 140,024
bundled measured positions, the viewer draws 124,289 classified optic, central
and descending somata; it omits VNC-associated and unclassified cells.
Missing positions are never generated.

Native 8 nm coordinates are centered, rigidly rotated and uniformly scaled.
**XY view** resets the projection; **Orbit** controls rotation. Point size and
color are display choices. Flybody is a surface mesh here, not a physics
simulation. No trained policy, neural simulator or biological firing data is
included.

The [atlas manifest](public/data/brain-atlas/manifest.json) records source,
filters and hashes. The [data notice](public/data/brain-atlas/NOTICE.md)
includes the command to reproduce or audit the export. The header image is a
sample of those same measured coordinates.

## Verify and deploy

```sh
npm test                # model-output contract and the bundled fixture
npm run check:assets    # anatomical asset hashes
npm run build           # type check and static build
npm run preview         # inspect dist/ locally
```

GitHub Actions runs the same checks. Serve `dist/` on any static host; assets
also work under a subpath. On Cloudflare Pages, use build command
`npm run build` and output directory `dist`. GPU training and inference run
separately and provide replay files or live outputs to the frontend.

## Licence

**[Cobanov Template Attribution License 1.0](LICENSE)** is a custom,
attribution-required source-available license, not an OSI-approved license.
Keep this linked credit in both your web UI and repository README:

Built with [fly-connectome-template][repo] by [Mert Cobanov][author].

The ready-made UI credit is `src/components/Attribution.tsx`. You may restyle it
or move it to an About/Credits view reachable in one click from the main UI;
you may not hide it or remove its links. [ATTRIBUTION.md](ATTRIBUTION.md) explains
the requirements. Derived versions must identify that they were modified.
Your own models, weights and modifications do not have to be published.

MaleCNS data remains **CC BY 4.0**, credited to FlyEM / HHMI Janelia, University
of Cambridge, MRC Laboratory of Molecular Biology and Google Research.
Flybody remains **Apache-2.0**. Their licenses are separate from the template's;
see [third-party notices](THIRD_PARTY_NOTICES.md).

[repo]: https://github.com/cobanov/fly-connectome-template
[author]: https://github.com/cobanov
[generate]: https://github.com/new?template_name=fly-connectome-template&template_owner=cobanov
