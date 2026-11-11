# Design Decisions

Each decision records what was decided, why, and when. Decisions are revisited only with new data, never with test-set results.

## D1 — Reading mode for v0.1 (2026-10-08)

**Decision:** v0.1 supports known-range mode only.
- `min`, `max`, and `unit` are required.
- The angles of the min/max scale marks are optional.
- Calling `read()` without a range raises a `ValueError` with a clear message.

**Why:** In auto-range mode, an OCR error produces a large error that still looks confident. Example: on a 0–10 bar gauge reading 5 bar, if OCR reads "10" as "16", the reported value becomes 8 bar, which is 30% FS off. A missing input should fail loudly instead of returning a number.

**Exception vs. status:**
- Missing or invalid caller input → exception (the caller's bug).
- Outcomes that depend on the image → status (`ok`, `low_confidence`, `unreadable`, `no_gauge`).

**Auto-range:** not part of the v0.1 API. It is measured only as Design C without the range in the prompt; that result informs the v0.2 decision.

## D2 — Test sets across versions (2026-10-08)

**Decision:** every release that changes a model gets a fresh test set.
- The new test set uses new photos and the same labeling guideline.
- Labels are committed before any system runs on it, and it is run once.
- Older test sets are reported separately as "seen" regression sets. They are never used as headline numbers.

**Why:** a test set stops being an unbiased measurement once results on it have been seen. Later versions also need different data, such as gauges with readable scale numbers for auto-range or 7-segment displays. A test set sealed today would lock in today's data distribution.

## D3 — Core dependencies (2026-10-08)

**Decision:** the core install contains `numpy`, `opencv-python-headless`, `onnxruntime`, and `huggingface-hub`. After `pip install gaugelens`, the default reader (Design A, ONNX) works with no extra.

**Why:** a default that requires an extra fails on first use. ONNX Runtime is small compared with PyTorch. PyTorch is needed only for training (on Kaggle) and for Designs B and C (extras).

**Extras:** `[synth]`, `[foundation]`, `[vlm]`, and `[demo]` are each added in the commit that introduces their code. Training dependencies are never an extra; they live in `training/requirements.txt`.

**Revisit if:** a core dependency becomes a problem, for example missing wheels on a platform or install size.

## D4 — Angle convention and needles outside the scale (2026-10-08)

**Angle convention:** clock convention.
- 0° points to 12 o'clock and angles grow clockwise.
- Pixel y points down.
- Values increase clockwise from the min mark to the max mark. Counterclockwise scales are out of scope for v0.1.

**Outside the scale:** the dead zone between the max mark and the min mark is split in half.
- The half next to the max mark counts as above the range; the half next to the min mark counts as below it.
- An exact tie counts as above, because missing an overpressure is the worse error.
- A needle more than an edge margin beyond an end mark gets `value=None` and status `below_range` or `above_range`.
- Within the margin, the value is clamped to the end mark. This covers measurement noise, for example a needle resting on zero.

**Why:**
- Clamping would report 10 bar on a 0–10 bar gauge whatever the real overpressure is. That is a confidently wrong reading in exactly the case that matters most for safety.
- Extrapolating reports a number the gauge is not calibrated for.

**Status set:** `ok`, `low_confidence`, `unreadable`, `no_gauge`, `below_range`, `above_range`.

**Edge margin:** the provisional default is 1% FS. The final value is chosen on the dev set together with the tolerances.

## D5 — Synthetic renderer (2026-10-08)

**Decision:** the v0.1 generator is a 2D procedural renderer (Pillow). It ships in the library as `gaugelens.synth` behind the `[synth]` extra.

**Why:**
- It is fast on CPU and runs on Kaggle and in CI.
- Its code is Apache-2.0 and unit-testable.
- Labels are exact by construction.

**Later:** a 3D renderer (Blender) is added only if the data calls for it. That means the measured synthetic→real gap stays large after 2D realism improvements made one change at a time.
- It would be a separate tool outside the library. Blender renders are free to use, but published bpy scripts must be GPL-compliant.
- SyncG (Blender-rendered) can serve as a 3D comparison if its license allows.

## D6 — Backgrounds for synthetic scenes (2026-10-08)

**Decision:** backgrounds mix procedural patterns with CC0 photos from Poly Haven. By default 70% of scenes use a photo and 30% a procedural pattern.
- 30 panoramas of industrial indoor scenes (tonemapped JPG). Crops come from their middle band, where the panorama projection distorts least.
- 59 surface textures: metal, concrete, plaster, and brick.

**Why:**
- The photos give realistic industrial context under the cleanest license (CC0).
- The procedural share keeps a model from memorizing a few photos.

**Poly Haven API terms (checked 2026-10-08):**
- Free for any purpose.
- Users must be told the assets came from Poly Haven.
- Requests need a unique User-Agent.

Downloads go to `data/backgrounds/`, which is not committed. A `manifest.json` there records each file's source, authors, and license.

## D7 — Synthetic train/dev splits (2026-10-08)

**Decision:** synthetic train and dev sets share nothing a model could memorize.
- Seeds come from ranges that can never overlap: train starts at 0, dev at 100,000,000.
- Background photos are split by a hash of the file name: about 20% are dev-only and the rest train-only. A photo keeps its split when new photos are added.
- Procedural backgrounds are fresh for every seed, so both splits use them.

**Why:** if dev scenes reused the training backgrounds, a model could separate gauge from background by memory. The dev score would then be optimistic, while real photos always show backgrounds the model never saw.

**What each set answers:**
- Synthetic dev: "did the model learn the synthetic task?"
- Real dev images: "does it transfer to real photos?"

## D8 — Same inputs for every design; A0 finds the scale ends itself (2026-10-09)

**Decision:** every design gets the same inputs: `min`, `max`, and `unit`. Each design finds the angles of the min and max marks on its own. A caller may still pass the angles, as an optional override for any design.

A0 finds the ends from the ticks:
- Major ticks are ink runs that start at the tick ring and reach inward.
- The dead zone is the widest empty gap between neighboring ticks.
- If a second gap scores almost as high, the reading is `low_confidence`.

**Why:**
- If A0 got the angles while Design A had to find them, the comparison would not be fair.
- Assuming a standard 270° sweep would be a silent guess.

## D9 — A0 is frozen after one improvement (2026-10-09)

**Decision:** A0 gets one improvement, an affine tilt correction, and is then frozen. The correction fits an ellipse to the face and stretches it into a circle. The face is filled between Canny edges, which act as walls.

**Synthetic dev results (in-range needles, n = 914):**

| | within 2% FS | within 5% FS | `ok` but >5% off | abstained | off-scale status right |
|---|---|---|---|---|---|
| before | 21.2% | 27.7% | 8.6% | 47.9% | 37.2% |
| after | 34.0% | 53.7% | 10.0% | 24.3% | 62.8% |

**Known limitation:** under perspective, the dial's true center is not the center of its ellipse. With the camera at three dial-sizes' distance and a 20° tilt, the gap is already about 5% of the radius, even though the ellipse itself is fitted to within 0.1%. That gap is now A0's main error.

The full fix is projective rectification: given the ellipse and the true center, the vanishing line is the polar of the center with respect to the ellipse. It needs the center as an input. Design A predicts the center as a keypoint, so this rectification is built as shared geometry when Design A is built.

## D10 — Real images: sources and roles (2026-10-09)

**Decision:** no own photos. Every real image comes from a public source with a clear license, and each source has exactly one role.

| Role | Source | License |
|---|---|---|
| Test | Wikimedia Commons, categories Pressure gauges, Manometers, Vacuum gauges, Gauges | CC0, public domain, CC BY, CC BY-SA (per file) |
| Real dev | Aalborg pressure gauge data (Kaggle); MeasureBench dial images | Aalborg CC BY 4.0 (Kaggle's listing; first recorded here as CC BY-SA 4.0 by mistake); MeasureBench CC BY-SA 4.0 |
| Training (Design A) | Roboflow Universe datasets with keypoint labels | CC BY 4.0, verified per dataset before use |

**Contamination:** VLMs were trained on web images and may have seen these photos.
- The test uses photos whose readings were never published, so no model can recall an answer.
- Commons files that MeasureBench uses are excluded, because their readings are public.
- The README reports this as a limitation.

**Duplicates:** images are compared across sources by perceptual hash. If a duplicate appears in the test set and elsewhere, it stays in the test set and is removed from the other source.

**Protocol:**
- Test candidates are chosen by metadata only: category, license, format, and size.
- Eligibility and readings are labeled by the user, following `docs/labeling-guideline.md`.
- No evaluated reader or model looks at the test images before the labels and image hashes are committed. The assistant's role is set by D11.

## D11 — The assistant and the test images (2026-10-15)

**Decision:** the assistant may look at test candidates, both for rule questions and to review labels. The images stay in the test set.
- Labels are the author's. When the assistant disagrees with a reading, it asks the author to re-read the image instead of supplying a value, and the author decides.
- The author's intra-rater agreement (first pass vs. the 10% relabel) is computed before any review, so that number is untouched by the assistant.
- Images the assistant saw are logged in `data/raw/commons/shown_to_assistant.jsonl`.
- The test set's documentation describes the labeling process in one factual sentence, for example: "labeled by the author; an AI assistant, which is not one of the evaluated systems, was consulted on rule questions and reviewed the labels."

**Why:**
- The evaluated systems never see the test images, so their results stay pre-registered.
- The review catches rule mistakes early.
- The assistant reads gauges poorly, so readings stay with the author.

## D12 — Uneven scales stay in the test set (2026-10-17)

**Decision:** a dial whose numbers are evenly spaced except in one section, most often a shortened first section, is eligible. Its label is marked `uneven_scale`, and results are reported with and without these dials. Scales whose spacing changes along the whole sweep (logarithmic, square-root) stay out of scope.

**Why:**
- The pattern is common on ordinary gauges: in a check of the eligible dials among the first 53 candidates, 8 of the 15 whose marks could be counted had it, from cheap 0–11 bar regulator gauges to old brass ones.
- A user would not notice it, so to them these are normal gauges. Excluding them would make the test set easier than real use.
- A reader that spaces values evenly between min and max has a built-in error on these dials, about 1.5–2% FS near the low end on the ones measured. That is the kind of weakness the comparison should show, not hide.

**Unprinted start:** on a pressure gauge, an unprinted start mark is labeled 0 and marked both `inferred_end` and `uneven_scale`, even when counting back with even spacing lands above 0. A pressure scale starts at 0 unless another start number is printed. If that assumption is wrong for a dial, the error stays inside two subsets that are reported on their own.

**Rejected:** excluding dials whose shape error exceeds 1% FS. Measuring that from a photo needs the dial's true center, and on tilted photos a few pixels of center error move the result by about 1% FS. The decision would then depend on the measurement more than on the dial. Counting gaps does not have this problem.

## D13 — Several photos of the same gauge (2026-11-03)

**Decision:** photos that show the same physical gauge all stay in the test set. Each is labeled on its own, and the notes of each say `same gauge: <pages>`. Confidence intervals come from a bootstrap that resamples gauges, not photos.

**Why:**
- Different photos of one gauge differ in angle, light, and often the reading, so each is a real test case. Keeping only one would throw away data.
- But they are not independent: a design that fails on a gauge's face tends to fail on all of its photos. Counting them as independent would make the intervals look narrower than the evidence supports.
- Resampling whole gauges keeps every photo and still gives honest intervals. Sample sizes are reported in images and in gauges.

**Seen so far:** 6/7, 33/34, 93/94/95 and 183/184 among the eligible images of the first 200 candidates.

## D14 — MeasureBench dial gauges as real dev data (2026-11-10)

**Decision:** the 139 dial pressure and blood pressure gauges of MeasureBench's real-world split are the first real dev set. The AI assistant labels eligibility and the scale (`min`, `max`, `unit`) with the test-set guideline; the author checks a sample. Readings are MeasureBench's intervals.

**Why:**
- D11 keeps readings with the author because the test set must stay unbiased. Dev data only steers tuning; a small label error shifts a setting slightly and is caught by the next dev run, never reported as a result.
- MeasureBench gives readings but no scale range, and every design needs the range as input (D8). Labeling the range is the missing step.
- The same guideline keeps dev and test comparable: a design tuned on gauges the test set would exclude would be tuned for the wrong task.

**Overlap with the test set:** checked by a 256-bit difference hash (`evaluation/dedup.py`). A 64-bit hash flagged two different dials as copies, because round white dials look alike at 9x8 pixels; at 256 bits copies differ by 0-1 bits and the closest different dials by 68. No copies were found. Two other photos of a test-set gauge were dropped.

## D15 — Aalborg test videos as real dev data, read by the assistant (2026-11-11)

**Decision:** the seven test videos of the Aalborg data (`man1` … `man7`, cropped to the dial) are the second real dev set. The data holds no readings: its numbers are the paper's own needle angles. The AI assistant picks 32 frames and labels them in full with the test-set guideline, reading included; the author checks a sample. Every frame of the four videos whose needle does not move also serves a label-free stability check.

**Why:**
- D14 took readings from MeasureBench; here there are none to take. The paper's angles cannot stand in: a design would be scored against another reader, that reader's errors included.
- The assistant is not one of the evaluated designs (Design C uses Qwen and Gemini), so its readings give none of them an answer. The risk is plain label error, which the author's check and dev-only use (D14) contain.
- Seven videos are seven gauges, not 20,000 samples: frames of one video are near-copies (D13). Frames are picked for spread: in the three videos where the pressure changes, the steady part and the fall at the end, skipping frames with a smeared needle where a sharp one is near; in the four where it does not, the first and the last frame.
- In a video whose needle does not move, readings that differ between frames are error, whatever the true value. The check cannot see a reader that is off by the same amount in every frame; the labeled frames cover that.

**Overlap with the test set:** no frame is within 25 bits of a test candidate (closest: 71). The gauges sit in sewer pump stations in Aalborg and were filmed by the paper's authors.
