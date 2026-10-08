# Prior Art and License Survey

_Last checked: 2026-10-04. Licenses change — re-verify every entry before each release._

## 1. Existing gauge readers

| Project | Approach | License | pip-installable | Notes |
|---|---|---|---|---|
| [ethz-asl/analog_gauge_reader](https://github.com/ethz-asl/analog_gauge_reader) (Reitsma et al., ICRA 2024) | Detection (YOLOv8) → needle segmentation → keypoints → MMOCR reads scale numbers → ellipse fit → value | MIT code, but depends on Ultralytics (AGPL-3.0) | No (Poetry/conda, pinned PyTorch 2.0.0) | Closest to our Design A + auto-range. Research code. |
| [Leon-Alcazar et al., WACV 2024](https://arxiv.org/abs/2308.14583) (KAUST) | Learned reader trained on synthetic data, tested on real | Repo [fuankarion/automatic-gauge-reading](https://github.com/fuankarion/automatic-gauge-reading) has only a README, no license file | No | Directly about the synthetic→real gap. |
| Hobby OpenCV readers, e.g. [kaiffeetasse/analog-gauge-reader](https://github.com/kaiffeetasse/analog-gauge-reader), [Romanmc72/gauge_reader](https://github.com/Romanmc72/gauge_reader) | Hough circle/line + manual calibration | Various | No | Equivalent to our A0 baseline. |
| PyPI | — | — | — | No maintained gauge-reading package found. |

## 2. VLM-focused studies

- **MeasureBench** ([Lin et al., CVPR 2026](https://arxiv.org/abs/2510.26865); data: [FlagEval/MeasureBench](https://huggingface.co/datasets/FlagEval/MeasureBench), CC BY-SA 4.0).
  1,272 real + 1,170 synthetic images, 26 instrument types. Best VLM: 30.3% correct on the real set.
  VLMs read the *unit* correctly >90% of the time; they fail at mapping pointer position to a scale value.
- **Mueez et al., arXiv [2608.17723](https://arxiv.org/abs/2608.17723) (Aug 2026)** — Qwen2.5-VL-7B only.
  - Zero-shot mean % error (range-normalized): 15.7–39.1%. Zero-shot error was *higher* with the range in the prompt on all three datasets.
  - After QLoRA fine-tuning: 2.4–4.4%; share within ±2% FS: 36–64%.
  - Reports high-confidence wrong answers and recommends abstention.
  - Code "will be released", but there was no link when checked. Paper license: CC BY-NC-ND 4.0.
- **Fu et al., "Lost in Motion" ([2604.22829](https://arxiv.org/abs/2604.22829), 2026)** — frontier API VLMs fail on gauge *video*. Video is out of scope for us.
- **SyncG** ([Deng et al., Scientific Data, Apr 2026](https://www.nature.com/articles/s41597-026-07308-x)) — 20k Blender-rendered gauges, 145 environments; boxes, keypoints, masks, OCR labels. **License not yet verified.**

## 3. How gaugelens differs

1. An installable, maintained Apache-2.0 library with no AGPL dependency anywhere.
2. Explicit statuses (`ok`, `low_confidence`, `unreadable`, `no_gauge`), with abstention vs. confidently-wrong reported as a first-class metric.
3. A head-to-head comparison of four families (A0 classical, A specialized, B foundation models, C VLM) on one pre-registered real test set. We found no prior comparison like this.
4. An ONNX-exportable specialized pipeline for CPU deployment.
5. A reusable synthetic gauge generator with exact ground truth.

## 4. Datasets

| Dataset | Content | License | Use |
|---|---|---|---|
| [Pressure Gauge Reader Data](https://www.kaggle.com/datasets/juliusgrassme/pressure-gauge-reader-data) (Aalborg sewer pump stations, 2018) | Real, video-derived, ~6.4 GB | CC BY-SA 4.0 | Dev-set candidate. ShareAlike: redistributed derivatives must stay CC BY-SA. |
| [MeasureBench](https://huggingface.co/datasets/FlagEval/MeasureBench) | Real + synthetic, many instrument types | CC BY-SA 4.0 | Dev / sanity check (dial subset). |
| [Synthetic Data for Precision Gauge Reading](https://www.kaggle.com/datasets/endava/synthetic-data-for-precision-gauge-reading) (Endava) | 501 synthetic images | CC BY-NC-SA 4.0 | **Excluded** (non-commercial). |
| Roboflow Universe gauge datasets | Mostly boxes/keypoints, no values | Per dataset | Check each one individually. |
| SyncG | 20k synthetic | To verify | — |
| [Poly Haven](https://polyhaven.com) HDRIs and textures | Background photos for synthetic scenes | CC0; the API asks for credit to Poly Haven | Synthetic backgrounds (see D6) |

## 5. Model candidates

| Role | Model | Family | License | Status |
|---|---|---|---|---|
| Gauge detector (A) | RT-DETR / RT-DETRv2 (`transformers`) | DETR (CNN backbone + transformer decoder) | Apache-2.0 | Default candidate |
| Gauge detector (A) | RF-DETR Nano–Large | DETR with DINOv2 backbone | Apache-2.0 | Default candidate |
| | RF-DETR XL / 2XL (`rfdetr[plus]`) | | PML 1.0 | Not allowed as default |
| Gauge detector (A) | D-FINE, DEIM | DETR variants | Apache-2.0 | Candidate |
| Gauge detector (A) | torchvision Faster R-CNN / SSDLite | CNN | Code BSD-3-Clause; check pretrained-weight terms | Candidate (CNN family) |
| Open-vocab detection (B) | [Grounding DINO tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) | DETR + text grounding | Apache-2.0 | `[foundation]` extra |
| Open-vocab detection (B) | [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) | CLIP-based (ViT) | Apache-2.0 | `[foundation]` extra |
| Segmentation (B) | [SAM 2.1](https://github.com/facebookresearch/sam2) | Promptable segmentation (Hiera ViT) | Apache-2.0 | `[foundation]` extra |
| Segmentation (B) | [SAM 3](https://github.com/facebookresearch/sam3) | Promptable + concept segmentation | SAM License (custom, use restrictions) | Eval only |
| VLM (C) | Qwen2.5-VL-3B | VLM | Qwen Research License (non-commercial) | Eval only |
| VLM (C) | Qwen2.5-VL-7B | VLM | Apache-2.0 | `[vlm]` extra |
| VLM (C) | Qwen2.5-VL-72B | VLM | Qwen License | Eval only (also too large for 2×T4) |
| VLM (C) | Qwen3-VL 2B/4B/8B/32B | VLM | Reported Apache-2.0; verify each model card | Candidate |
| VLM (C) | Qwen3.5 small dense models | Natively multimodal | Reported Apache-2.0; verify each model card | Candidate |
| API reference (C) | Gemini Flash, free tier | Proprietary API | Google API terms; free-tier inputs may be used to improve Google products | Eval only; send only public or own images |

Ultralytics / YOLO is excluded from this project entirely, including evaluation scripts (project rule).
