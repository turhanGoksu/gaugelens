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
- Prior-art and license survey (`docs/prior-art-and-licenses.md`) and the
  design decision log (`docs/decisions.md`).
