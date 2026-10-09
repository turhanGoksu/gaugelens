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
