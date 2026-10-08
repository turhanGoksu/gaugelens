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
