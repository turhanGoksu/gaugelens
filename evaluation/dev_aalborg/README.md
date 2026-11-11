# Real dev set: Aalborg pump station videos

Frames of real gauge videos, used for tuning and design choices (D10, D15).
Never used as a headline result. The test set is `evaluation/test_set_v1`.

## Source

[Pressure Gauge Reader Data](https://www.kaggle.com/datasets/juliusgrassme/pressure-gauge-reader-data)
on Kaggle, version 1 (2018-12-13), license CC BY 4.0. It belongs to the paper
Lauridsen, Graasmé, Pedersen, Jensen, Andersen and Moeslund, "Reading circular
analogue gauges using digital image processing", VISAPP 2019.

This set uses the paper's seven test videos, `Data/2 Test videos/edited
videos/man1cropclipscale.mp4` … `man7`: each is cropped to the dial and
scaled to 512x512 at 29.97 fps. `videos.csv` pins their size and SHA-256.
The rest of the data (raw and hand-held videos, about 6 GB) is not used.

The data has no readings. Its `angle.npy` and `med_test.mat` files hold the
needle angles that the paper's algorithm measured, and `errorfile.txt` lists
the frames where it failed. A reader's output is not ground truth, so the
readings here are new labels.

```
python -m evaluation.aalborg download data/external/aalborg "Data/2 Test videos/edited videos/*"
python -m evaluation.aalborg frames data/external/aalborg
```

The first command needs a Kaggle key (`.env.example`); the second checks the
videos against `videos.csv` and writes the labeled frames to
`data/external/aalborg/frames/<id>.png`. Videos and frames are not stored in
this repository.

## Labels

`labels.csv` has one row per frame, with the test set's columns
(`docs/labeling-guideline.md`) plus `gauge` and `frame` (0-based, in decoding
order). The id is `<gauge>_f<frame>`.

The labels, readings included, were made by the AI assistant, not the author;
a sample went to the author for review (D15). Readings follow the guideline: to one fifth
of the smallest division, read from the frame as it is.

| Gauge | Scale | Video | Needle | Frames | Labels |
|---|---|---|---|---|---|
| man1 | 0 … 6 bar | 394 s | steady at 0.80–0.86, then falls and swings below 0 | 8 | 6 value, 2 `below_range` |
| man2 | −1 … 3 bar | 74 s | steady near 0.7, then falls to −0.44 | 8 | 8 value |
| man3 | 0 … 2.5 bar | 60 s | steady near 0.3, then falls below 0 | 8 | 7 value, 1 `below_range` |
| man4 | −1 … 1.5 bar | 37 s | does not move | 2 | 2 value |
| man5 | −1 … 1.5 bar | 15 s | does not move | 2 | 2 value |
| man6 | −1 … 3 bar | 35 s | does not move | 2 | 2 `above_range` |
| man7 | 0 … 6 bar | 19 s | does not move | 2 | 2 `below_range` |

32 frames: 25 value, 5 `below_range`, 2 `above_range`; 26 `sure: yes`. All
seven gauges are eligible. Six labels are `sure: no`: three needles smeared
by vibration, one crossing the zero mark, and both frames of man7, whose
needle rests about half a division below zero, close to the guideline's
limit for `below_range`.

## Stability check

In man4 … man7 the needle stays put for the whole video (checked on 24
evenly spaced frames each). Any spread in a design's readings over their
3,186 frames is error, so this check needs no labels. man5 and man6 are
hand-held, so the view shifts while the value stays.

## Limits

- Seven gauges. Frames of one gauge are near-copies, so intervals resample
  gauges, not frames (D13).
- Clean views: each video is cropped and centred on the dial, unlike the
  test set's photos.
- The values cover 0–45 % of the full scale; no needle points into the upper
  half of its scale.
- Readings are the assistant's. A sample of 10, chosen to include every
  out-of-range label, went to the author, who asked for no changes.
