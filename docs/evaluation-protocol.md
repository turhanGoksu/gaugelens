# Evaluation protocol (pre-registered)

_Status: draft. The test-set v1 labels are committed first. This protocol is
frozen in a later commit, before any system runs on the test images. Open
items are marked **open** and must be settled before that commit. None of
them is chosen by looking at the test set._

## Data

- **Test set v1:** Wikimedia Commons photos (D10). The user labels
  eligibility and readings following `docs/labeling-guideline.md`. Labels,
  image SHA-256 hashes, and attributions are committed before any system
  runs on the images.
- Every design configuration runs on the test set **once**. Results are
  reported as they come out, including failures.
- All tuning and every decision uses the synthetic dev set and the real dev
  set (Aalborg, MeasureBench dial images), never the test set.

## Inputs

Every design receives the image, `min`, `max`, and `unit` (D8). No mark
angles are given. Design C is also run without the range, as a separate,
labeled row.

## Accuracy

Images with label status `value`:

| Metric | Definition |
|---|---|
| **Accuracy at 2% (primary)** | Share read within ±2% of full scale with status `ok` or `low_confidence`. Abstentions count as misses. |
| Accuracy at 1% and 5% | The same at ±1% and ±5% FS. |
| Coverage | Share answered with a number. |
| Median error | Median absolute error in % FS over answered images. |

## Failure behavior

| Metric | Definition |
|---|---|
| **Confidently wrong (headline failure)** | Status `ok` with error above 2% FS, as a share of the `value` images. |
| Flagged wrong | Status `low_confidence` with error above 2% FS. |
| Abstained | Status `unreadable` or `no_gauge` on `value` images. |
| Off-scale handled | For images labeled `below_range` / `above_range`: share with the matching status, and share given as `ok`, which is the worst case. |
| Number for an unreadable image | For images labeled `unreadable`: share that received status `ok`. |

## Label noise

- Labels marked `sure: no` are kept. Results on the `sure: yes` subset are
  reported as a robustness check.
- Labels with an end value that is not printed on the dial (`inferred_end`,
  guideline rule 6) are kept and reported as their own subset.
- A 10% relabel, done at least one day later, gives the intra-rater
  agreement: identical status, and readings within ±2% FS.

## Scale shape

Gauges labeled `uneven_scale` (guideline rule 4, D12) stay in the test set.
Every accuracy metric is also reported without them, because a reader that
spaces values evenly between min and max has a built-in error on these
dials.

## Uncertainty

Every share comes with its sample size, counted both in images and in
gauges, and a 95% confidence interval.

- Some gauges appear in more than one photo (marked `same gauge` in the
  label notes, D13). Their photos are not independent, so intervals come
  from a bootstrap that resamples gauges: all photos of a gauge are drawn
  together. 10,000 resamples, percentile interval, fixed seed.
- The Wilson interval over images is reported next to it for reference.

## Speed, size, cost

- Latency per image at batch size 1, after warm-up: median and p90, on CPU
  (the local machine) and on GPU (Kaggle T4).
- Model size on disk and peak memory.
- Cost per 1,000 images: API price (free tiers noted with their rate
  limits) and an estimate of local compute.
- Every number is reported with its platform, library versions, and model
  revisions.

## Open items

- **open:** the final edge margin (D4). It is chosen on the dev set; the
  provisional value is 1% FS.
- **open:** the release gate, meaning what the default design must reach on
  the test set to be released as non-experimental.
