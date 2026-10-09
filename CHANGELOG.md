# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). While the version is 0.x, minor
releases may change the API.

A released version is never changed: fixes ship as a new version.

## [Unreleased]

### Added

- Project skeleton: `src/` layout, Apache-2.0 license, CI (lint, tests on
  Python 3.11 and 3.14, wheel build and install check).
- `gaugelens.geometry`: needle angle <-> value on a linear circular scale
  (clock-convention angles, scales that cross 12 o'clock, dead-zone split into
  `below_range` / `above_range`, provisional 1% FS edge margin).
- `gaugelens.synth` (`[synth]` extra): 2D procedural dial renderer with exact
  labels (value, center, needle tip, min/max marks, bbox). Randomizes range,
  unit, sweep, ticks, colors, needle and bezel; RGBA output for a later
  capture stage.
- Capture stage (`gaugelens.synth.capture`): places a dial in a scene
  through one homography (camera tilt, roll, distance), then adds glare,
  shadow, lighting changes, blur, noise, and JPEG compression. Keypoints go
  through the same homography, and the shooting conditions are recorded in
  the labels.
- Backgrounds (`gaugelens.synth.backgrounds`): procedural patterns plus CC0
  photos from Poly Haven, downloaded with `download_polyhaven()` into a
  folder with a source/license manifest.
- Dataset writer and CLI (`python -m gaugelens.synth`): `download-backgrounds`
  and `generate`. A dataset folder holds `images/`, `labels.jsonl` and
  `dataset.json`; scenes depend only on their seed, so parallel workers write
  identical files. Train and dev use disjoint seeds and disjoint background
  photos.
- Design A0, the classical baseline (`gaugelens.classical.read_classical`):
  - Hough circle: the precise ALT variant, with the classic one as fallback.
  - Tilt correction: the face is filled between Canny edges, an ellipse is
    fitted to it, and the ellipse is stretched back into a circle.
  - Polar unwrap; the needle is found in the middle band.
  - Scale ends from the major ticks and the widest empty gap.
  - Every outcome has a status.
- Dev evaluation runner (`python -m evaluation.run_dev`): error in % of full
  scale, share within 1/2/5% FS, confidently wrong readings, abstentions,
  out-of-range handling, and a breakdown by camera tilt.
- Prior-art and license survey (`docs/prior-art-and-licenses.md`) and the
  design decision log (`docs/decisions.md`).
