# Real dev set: MeasureBench dial gauges

Real photos used for tuning and design choices (D10, D14). Never used as a
headline result. The test set is `evaluation/test_set_v1`.

## Source

[FlagEval/MeasureBench](https://huggingface.co/datasets/FlagEval/MeasureBench),
`real_world` split, dataset revision
`93c001671055f739bc396c73aa6b2d5e6171abed`, license CC BY-SA 4.0.
`python -m evaluation.measurebench data/external/measurebench --test-images
data/raw/commons` downloads the 139 photos of `pressure_gauge` and
`sphygmomanometer` with `design == "dial"`. The images and MeasureBench's
readings are not stored in this repository.

MeasureBench gives, for each photo, the interval of accepted readings and
its unit, one interval per scale. It gives no scale range.

## Labels

`labels.csv` adds what the designs need as input: eligibility under the test
set's guideline (`docs/labeling-guideline.md`), `min`, `max` and `unit` of the
primary scale as printed, `inferred_end`, and `reading_index`, which picks
the MeasureBench interval that belongs to that scale.

The labels were made by the AI assistant, not the author; the author checked
a sample (D14). Readings come from MeasureBench, so this set measures error
against MeasureBench's intervals.

| | Photos |
|---|---|
| Downloaded | 139 |
| **Eligible** | **97** (80 pressure gauges, 17 blood pressure gauges) |
| `out_of_scope` | 15 |
| `range_illegible` | 12 |
| `several_gauges` | 4 |
| `duplicate_image` (same frame as another row) | 4 |
| `reading_suspect` (MeasureBench interval does not match the dial) | 3 |
| `same_gauge_as_test` (another photo of a test-set gauge) | 2 |
| `reading_outside_range` | 1 |
| `not_a_photo` | 1 |

38 eligible labels use `inferred_end`; most are blood pressure gauges, whose
zero is a box instead of a number, and tyre gauges that start numbering at 10.

## Overlap with the test set

`evaluation/dedup.py` compares every photo with every test candidate by a
256-bit difference hash. No photo is a copy of a test candidate. Two photos
show the MAXIMATOR gauge of test candidate `commons_45045263` with a different
reading; they are excluded as `same_gauge_as_test`.

## Limits

- 121 of the 139 photos are frames from YouTube videos, so this set is
  closer to video stills than to the museum photos of the test set.
- Readings are MeasureBench's. Three were dropped because the interval does
  not fit the dial; others may be off without it being visible.
