# Labeling guideline (test set v1)

_Status: draft. It is final once committed together with the first labels;
after that, any change requires a new test set version (D2)._

The person labeling follows these rules for every candidate image. No
evaluated reader or model looks at a test image before its label is
committed (D10); the assistant's role is set by D11.

## 1. Is the image eligible?

Label the image **eligible** only if all of these hold. Otherwise label it
**excluded** and pick the first reason that applies.

| # | Rule | Exclusion reason |
|---|---|---|
| 1 | It is a photo, not a drawing, render, diagram or screenshot. | `not_a_photo` |
| 2 | A gauge dial is in view: the face of a measuring instrument, turned toward the camera. What kind of face it is (one needle, digital, reeds) is rule 4's question. Pipes, valves, regulators, the back or side of a gauge, or people and places with no dial in view fail this rule. So does a picture of a gauge (printed on packaging or a poster, or shown on a screen): it is not an instrument. | `no_dial` |
| 3 | Exactly one gauge dial is the subject. Another gauge counts only if its dial is in view, and such a dial may appear only if cut off by the frame. Only round dials count here: straight-scale instruments such as glass thermometers and level or sight glasses do not. The subject may itself be cropped, as long as rule 6 holds. | `several_gauges` |
| 4 | The dial is circular, with one needle and a linear scale: equal value steps take equal arcs everywhere. If the step between printed numbers changes (10, 20 … 50, then 55, 60, 65), the gaps must shrink in the same proportion; if they stay the same size, the scale is not linear. One exception stays in (D12): an **uneven scale**, where the numbers are evenly spaced except in one section, most often the first one (from the min mark to the first number), which has fewer gaps or a different layout than the others. Many ordinary gauges are built this way. Label it as usual and tick "uneven scale". Judge by the marks, not by measuring the photo: count the gaps in each section and compare the layouts, since a tilted photo makes sections look longer or shorter. A scale that runs more than one turn (numbers continue on a second ring, often with a window showing the turn) fails this rule: one needle angle stands for two values. So does a compound scale whose vacuum part uses another unit (for example 76 and 40 cmHg left of 0, then 1, 2, 3 bar): no single min, max and unit describe it without converting. A compound scale in one unit (−1 … 3 bar) is fine. A full-circle scale whose min and max marks meet at the same tick (for example 50 next to 0 at the top) also fails: there is no dead zone, and one angle stands for both ends. A second needle, such as a red drag pointer or a barometer's set hand, excludes it; so do digital displays. A painted mark on the dial or bezel is not a needle. | `out_of_scope` |
| 5 | Values increase clockwise. | `counterclockwise` |
| 6 | The numbers at the two ends of the scale are legible, so min and max are known. A scale marked only E–F or with colored zones and no numbers fails this rule. One narrow exception: an end number that is not printed may be filled in when its tick is visible and its value follows unambiguously from evenly spaced printed numbers (5, 10, 15 … with the first tick one step before 5 is 0); tick "an end number is not printed". On a pressure gauge, an unprinted start mark is 0 even when even spacing would put it elsewhere, because a pressure scale starts at 0 unless another start number is printed; tick "uneven scale" as well (D12). An end whose tick is not visible always fails. | `range_illegible` |

Gauges with two concentric scales (for example bar outside, psi inside) are
eligible. Label the **primary** scale. A scale qualifies if it passes rules 4–6 on its own: it is linear, its values increase clockwise, and the numbers at both ends of the sweep are legible (an unprinted end is allowed under rule 6's exception). If both scales qualify, label the outer one. If only one qualifies, label that one, even if it is the inner one; for example, an outer inHg vacuum scale that increases counterclockwise next to an inner −100 … 0 kPa scale that increases clockwise. If neither qualifies, exclude the image with the first rule the outer scale fails. Two rings that show the same range in two units are a dual scale; a second ring that continues the numbers is a multi-turn scale (rule 4).

Decide eligibility before readability. Two fully visible dials make the
image `several_gauges` even if only one of them is readable, and even if
the second one is small: a reader given one range cannot know which gauge
is meant, and keeping only the readable one would bias the test set toward
easy images.

## 2. What to record for an eligible image

| Field | Meaning |
|---|---|
| `min`, `max` | The numbers at the two ends of the labeled scale, as printed. |
| `unit` | As printed (`bar`, `psi`, `kPa`, `MPa`, `kg/cm²`, `°C`, `°F`, ...). Empty if none is printed. If the dial prints a multiplier such as ×0.1 MPa, write `min` and `max` as printed and put the multiplier in the unit (`×0.1 MPa`). Brand names (Bourdon, WIKA), instrument names (Manomètre) and class marks (Cl. 1.6, EN 837) are not units. |
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

If a candidate shows the same physical gauge as another candidate (same
dial in the same place, photographed again), label each photo on its own
and add `same gauge: <pages>` to the notes of every one of them, for
example `same gauge: 93/94/95` (D13).
