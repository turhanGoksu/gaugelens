# Training data: Roboflow Universe gauges

Real photos and video frames with four labeled points, for training Design A
(D10, D16). Never used for evaluation.

## Sources

| Dataset | Version | License | Labels | Images |
|---|---|---|---|---|
| [keypoints](https://universe.roboflow.com/project-kaqwa/keypoints-4lma6) by project-kaqwa | 1 (2023-04-12) | CC BY 4.0 | center, pointer_tip, min, max, meter | 862 |
| [gauge reader](https://universe.roboflow.com/computer-vision-ygqpn/gauge-reader-idem4) by computer-vision-ygqpn | 11 (2024-07-23) | CC BY 4.0 | base, tip, min_value, max_value, gauge | 373 |

Both datasets are by their Roboflow Universe uploaders and are used under
CC BY 4.0. The license was read twice: from the Roboflow API (`project.license`)
and from the `README.dataset.txt` of each export. Neither version is augmented.

The points are labeled as small boxes, and a point is the centre of its box.
In both sets the labels mean the same things; this was checked by drawing the
points on samples:

| Point | keypoints | gauge reader |
|---|---|---|
| needle pivot | center | base |
| needle tip | pointer_tip | tip |
| min mark | min | min_value |
| max mark | max | max_value |

## Selection (D16)

`manifest.csv` lists every image with `keep`, the `reason` when it is left
out, its SHA-256 and, for kept images, a `scene`.

| | keypoints | gauge reader |
|---|---|---|
| **Kept** | **445** | **118** |
| `web_image`: catalog photo or screenshot from the web | – | 241 |
| `stock_watermark`: Storyblocks stock footage | 114 | – |
| `same_gauge_as_dev`: a gauge of a dev set | 282 | 14 |
| `aalborg_dev_video`: frames of the Aalborg test video man3 | 21 | – |
| `near_test` / `near_dev`: within 25 bits of a test or dev image | 0 | 0 |

The kept 563 images fall into 109 scenes: groups of near-copies by the
256-bit difference hash of `evaluation/dedup.py` (25 bits or fewer apart). A
train/validation split must keep each scene on one side.

How the dev-set gauges were found:
- keypoints' `meter_a` videos show the TS 0–10 bar gauge of MeasureBench's
  `pressure_gauge_0`, on the same wall.
- gauge reader's phone photos of 2022-06-10, and four of 2022-06-13 at
  13:53, show Aalborg's man1, man2, man4/man5 and man6 at their pump
  stations. man4's serial number, 092655307, is legible in one of them.

Kept although they come from the Aalborg data, because their gauges are not in
the dev set:
- gauge reader's `out2` frames, from a training video of a 0–4 bar gauge;
- two frames of keypoints' `handheld5` and `handheld11` videos.

## Rejected candidates

- [dial-meter-detection-4-points](https://universe.roboflow.com/itu-weijie-wang/dial-meter-detection-4-points)
  (CC BY 4.0, 298 images in its unaugmented version): two videos of the TS
  gauge of `pressure_gauge_0`.
- [gauge-analog](https://universe.roboflow.com/analog-meter-reading/gauge-analog-tndze)
  (MIT, 9,139 images): has no downloadable version, and most likely gathers
  the same sources.
- Five others, found by the same search on 2026-11-16:
  - Three have no min or max marks.
  - Two offer only augmented versions.
  - `python -m evaluation.roboflow info` lists all of them.

## Reproduce

```
python -m evaluation.roboflow download project-kaqwa/keypoints-4lma6 1
python -m evaluation.roboflow download computer-vision-ygqpn/gauge-reader-idem4 11
python -m evaluation.roboflow select
```

The downloads need a Roboflow API key (`.env.example`). `select` writes
`manifest.csv` again; an unchanged file means the download matches. The
images are not stored in this repository.

## Limits

- **Few gauges.** The 445 keypoints images show four gauges (`meter_b`,
  `meter_c`, `meter_d`, `meter_f`): 411 frames of six videos, 32 single photos
  and the two Aalborg frames. Synthetic data stays the main training source;
  these images add real light, glare and camera angles.
- **Dev-gauge check by eye.** In gauge reader, every image left after the
  rules was checked. In keypoints, the check covered one frame of each video
  and every single photo, not every frame.
- **Licenses taken on trust.** The rules remove what is visibly not the
  uploader's to license. For everything else, the CC BY 4.0 statement is the
  uploader's word.
