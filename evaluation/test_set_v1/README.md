# Test set v1

Real photos of analog dial gauges from Wikimedia Commons, labeled for the
gaugelens comparison (D10). Labels and image hashes are committed before any
system runs on these images. The test is run once per design configuration
(`docs/evaluation-protocol.md`).

## Contents

| File | What it holds |
|---|---|
| `labels.csv` | One row per candidate: eligibility, exclusion reason, scale (`min`, `max`, `unit`), `inferred_end`, `uneven_scale`, status, reading, `sure`, notes. |
| `sources.csv` | Commons page, license, author and credit for every image, the URL of the downloaded file, and its SHA-256. |
| `relabel.csv` | The second labeling pass of a fixed 10% sample. |
| `agreement.json` | Intra-rater agreement between the first pass and the relabel. |
| `shown_to_assistant.jsonl` | Every time the AI assistant looked at a candidate, with the reason (D11). |

The images are not in the repository. Each row of `sources.csv` gives the
downloaded URL and the SHA-256 of the exact file that was labeled.

## How the candidates were chosen

`evaluation/commons.py` collected files from the Commons categories Pressure
gauges, Manometers, Vacuum gauges and Gauges, by metadata only: category,
license (CC0, public domain, CC BY, CC BY-SA), format and size. Off-topic
subcategories and files used by MeasureBench were skipped. This gave 488
candidates. One was removed afterwards because it is a rotated copy of
another candidate (`commons_75537145`, a copy of `commons_75537134`), leaving
487.

## Counts

| | Images |
|---|---|
| Candidates | 487 |
| **Eligible** | **149** |
| status `value` | 137 |
| status `below_range` | 8 |
| status `above_range` | 1 |
| status `unreadable` | 3 |
| Excluded | 338 |
| `several_gauges` | 119 |
| `out_of_scope` | 100 |
| `no_dial` | 80 |
| `range_illegible` | 20 |
| `not_a_photo` | 17 |
| `counterclockwise` | 2 |

Subsets reported separately (protocol): 72 eligible labels are `sure: yes`;
65 are `uneven_scale`; 29 are `inferred_end`. 19 eligible photos show a gauge
that appears in more than one photo (`same gauge` in the notes); intervals
resample gauges, not photos (D13).

## How the labels were made

The author labeled every candidate following `docs/labeling-guideline.md`.
The guideline was refined while labeling; each rule change was applied to all
candidates labeled before it.

An AI assistant, which is not one of the evaluated systems, was consulted on
rule questions and reviewed the labels; it did not supply readings (D11). It
looked at every candidate during the final review. `shown_to_assistant.jsonl`
records each look.

A fixed 10% of the candidates (chosen by image hash) was relabeled at least
one day after the first pass, without the first labels in view. Agreement
was computed before the assistant reviewed those images:

| | Agreement |
|---|---|
| Eligible or excluded | 47 / 47 |
| Same exclusion reason | 26 / 26 |
| Same scale (`min`, `max`) | 21 / 21 |
| Same status | 21 / 21 |
| Reading within ±2% FS | 21 / 21 |

The sample has 49 images; two are left out of the agreement because the
assistant looked at them before the relabel (`commons_112781964`,
`commons_97601141`). 15 of the 21 readings are at the scale minimum. After
the relabel, the author changed a flag (`uneven_scale` or `inferred_end`) on
three first-pass labels; no reading changed. `agreement.json` compares
against the first-pass labels as exported before any relabel or review.

## Limits

- Many photos come from museums and show old gauges at rest, so readings at
  the scale minimum are common.
- Vision-language models were trained on web images and may have seen these
  photos. The readings were never published, so no model can recall them.
- Once these labels are public, they can leak into future training data.
  Later releases use a new test set (D2).
