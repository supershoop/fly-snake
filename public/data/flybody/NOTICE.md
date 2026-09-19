# Flybody model notice

The browser model in this directory is derived from the anatomically detailed
fruit-fly body model published by the Turaga Lab:

- Project: `flybody`, fruit fly body model for MuJoCo physics
- Source: https://github.com/TuragaLab/flybody
- Authors: Roman Vaxenburg, Igor Siwanowicz, Josh Merel, Alice A. Robie,
  Carmen Morrow, Guido Novati, Zinovia Stefanidi, Gert-Jan Both, Gwyneth M.
  Card, Michael B. Reiser, Matthew M. Botvinick, Kristin M. Branson,
  Yuval Tassa, and Srinivas C. Turaga
- Development collaboration: Google DeepMind and HHMI Janelia Research Campus
- License: Apache License 2.0, included as `LICENSE`
- Publication: "Whole-body physics simulation of fruit fly locomotion",
  Nature 643, 1312-1320 (2025), https://doi.org/10.1038/s41586-025-09029-4

PinFly converts the original MuJoCo OBJ meshes into a compact binary browser
representation, simplifies the triangle meshes for interactive rendering, and
groups the two front legs separately so the F/J motor visualization can animate
them. The anatomical surface geometry comes from Flybody. The keyboard pose and
motor animation are PinFly additions and are not research simulation output.
