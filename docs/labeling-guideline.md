# Labeling guideline (test set v1)

_Status: draft. It is final once committed together with the first labels;
after that, any change requires a new test set version (D2)._

The person labeling follows these rules for every candidate image. No reader,
no model and no assistant looks at a test image before its label is
committed (D10).

## 1. Is the image eligible?

Label the image **eligible** only if all of these hold. Otherwise label it
**excluded** and pick the first reason that applies.

| # | Rule | Exclusion reason |
|---|---|---|
| 1 | It is a photo, not a drawing, render, diagram or screenshot. | `not_a_photo` |
| 2 | A gauge dial is in view: the face of a measuring instrument with a scale and a needle, turned toward the camera. Pipes, valves, regulators, the back or side of a gauge, or people and places with no dial in view fail this rule. | `no_dial` |
| 3 | Exactly one gauge is the subject. Other gauges may appear only if cut off by the frame. The subject may itself be cropped, as long as rule 6 holds. | `several_gauges` |
| 4 | The dial is circular, with one needle and a linear scale: consecutive numbers are equally far apart along the arc. A second needle, such as a red drag pointer or a barometer's set hand, excludes it; so do digital displays. A painted mark on the dial or bezel is not a needle. | `out_of_scope` |
| 5 | Values increase clockwise. | `counterclockwise` |
| 6 | The numbers at the two ends of the scale are legible, so min and max are known. A scale marked only E–F or with colored zones and no numbers fails this rule. | `range_illegible` |

Gauges with two concentric scales (for example bar outside, psi inside) are
eligible. Label the **outer** scale.

Decide eligibility before readability. Two fully visible gauges make the
image `several_gauges` even if only one of them is readable, and even if
the second one is small: a reader given one range cannot know which gauge
is meant, and keeping only the readable one would bias the test set toward
easy images.

## 2. What to record for an eligible image

| Field | Meaning |
|---|---|
| `min`, `max` | The numbers at the two ends of the labeled scale, as printed. |
| `unit` | As printed (`bar`, `psi`, `kPa`, `MPa`, `kg/cm²`, `°C`, `°F`, ...). Empty if none is printed. If the dial prints a multiplier such as ×0.1 MPa, write `min` and `max` as printed and put the multiplier in the unit (`×0.1 MPa`). |
| `status` | `value`, `below_range`, `above_range` or `unreadable` (section 4). |
| `value` | The reading, for status `value` only (section 3). |
| `sure` | `yes` if you would give the same reading again; `no` if it is a best guess within about one smallest division. |

## 3. How to read the value

1. Find the two ticks the needle tip lies between. The smallest division is
   the distance between two neighboring ticks; if there are no small ticks
   between the numbers, it is the distance between two numbers. For a thick
   needle, use the line through its middle; if the tip stops short of the
   ticks, extend that line in your mind.
2. Estimate where the tip's center line falls between them, to the nearest
   **one fifth of the smallest division**. Example: on a scale with 0.2 bar
   divisions, read to 0.04 bar.
3. Read the photo as it is. Do not try to correct for parallax or camera
   angle in your head; if the angle makes the reading uncertain by more than
   one division, set `sure` to `no`.
4. Write the value in the scale's units, with the decimals that the
   precision needs. The labeling tool's helper does the arithmetic: enter
   the ticks on both sides and click how many fifths past the lower one the
   tip is.

## 4. Statuses

- `below_range` / `above_range`: the needle is visibly past the min or max
  mark, by more than the width of the needle. A needle resting on the stop
  pin below zero is `below_range`. A needle exactly on the min mark, for
  example resting on zero, is status `value` with the min value.
- `unreadable`: the gauge is eligible but its reading cannot be determined.
  For example, glare or a reflection hides the needle tip, the needle cannot
  be located within one smallest division, or the image is too blurred.

## 5. Process

1. Label every candidate once, in the labeling tool, in one or more sittings.
2. At least one day later, relabel a random 10% without looking at the
   first labels. The agreement between the two passes is reported as the
   label noise.
3. Commit the labels together with the SHA-256 of every image before any
   system runs on them.
