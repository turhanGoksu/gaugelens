# gaugelens

Read analog dial gauges (pressure, temperature, …) from photos. Every reading
comes with an explicit status (`ok`, `low_confidence`, `unreadable`,
`no_gauge`); the library never returns a number without one.

> **Status: pre-alpha, under construction.** Nothing is released yet and
> there are no benchmark numbers yet. This README will report them once the
> pre-registered test set has been run.

## What this project measures

When is a general vision-language model good enough to read a gauge, and when
do you need a specialized pipeline? Four approaches are compared on the same
pre-registered real-image test set:

| Design | Approach |
|---|---|
| A0 | Classical computer vision baseline (no learning) |
| A | Specialized pipeline: trained detector + gauge geometry, ONNX-exportable |
| B | Foundation models without training: open-vocabulary detection + SAM 2 |
| C | Vision-language model asked for the reading directly |

Metrics: absolute error in % of full scale, share of readings within
tolerance, explicit failures vs. confidently wrong readings, latency, model
size, and cost.

## Scope of v0.1

- Single circular gauges with one needle and a linear scale.
- Known-range mode: you pass `min`, `max`, and `unit` from the gauge's spec.
  See [docs/decisions.md](docs/decisions.md) for why.

## Documentation

- [Prior art and license survey](docs/prior-art-and-licenses.md)
- [Design decisions](docs/decisions.md)

## License

Apache-2.0. Every default dependency is Apache-2.0-compatible. Models with
restrictive licenses are used for evaluation only and are never defaults.
