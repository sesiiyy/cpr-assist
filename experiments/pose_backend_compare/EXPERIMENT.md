# Experiment: 2D pose backends on a CPR detector crop

## Purpose

Compare **YOLOv8-Pose** and **Torchvision Keypoint R-CNN** (COCO 17 keypoints) on a single **detector crop** that shows a rescuer performing chest compressions on a mannequin. The goal is to choose a backend that tracks the **human caregiver** with a usable skeleton (torso and arms), for downstream readiness logic.

**Out of scope for this run:** RTMPose, HRNet (MMPose), and AlphaPose require separate heavy installs; see `results/metrics/OTHER_BACKENDS.txt`.

## Input

| Item | Description |
|------|-------------|
| Source | Readiness pipeline detector crop `crop_00` from run `ch1_updated_mppose` |
| Path in repo | `runs/ch1_updated_mppose/intermediate/detector_crops/crop_00.png` |
| Copy in this package | `results/images/01_input_detector_crop.png` |

The crop is tight on the rescuer and mannequin; foreshortening and hand–mannequin contact make wrists a hard case for any model.

## Methods

| Backend | Implementation | Notes |
|---------|----------------|--------|
| YOLOv8-Pose | Ultralytics `yolov8n-pose`, default weights | Single-stage pose; draws box + skeleton via `result.plot()` |
| Torchvision | `keypointrcnn_resnet50_fpn` with COCO weights | Two-stage detector + keypoints; **not** HRNet. Visualization uses the **single best-scoring** COCO “person” among all proposals (see code). |

Dependencies: Python 3 with `ultralytics`, `opencv-python`, `torch`, `torchvision`. First Torchvision run downloads ResNet50-FPN Keypoint R-CNN weights (~340MB).

## How to reproduce

From the **repository root** (`cpr/`):

```powershell
.\.venv\Scripts\python.exe experiments\pose_backend_compare\code\compare_pose_backends_on_image.py `
  --image runs\ch1_updated_mppose\intermediate\detector_crops\crop_00.png `
  --out-dir experiments\pose_backend_compare\results\regen `
  --torchvision
```

Or use `code/run_experiment.ps1`. The `results/regen/` folder is scratch output (gitignored); the **frozen** figures for slides stay under `results/images/`.

To regenerate only YOLOv8 overlays, omit `--torchvision`.

## Results (frozen snapshot)

### Numerical summary

See `results/metrics/summary.json` and `results/metrics/metrics_table.md`.

High level:

- **YOLOv8-Pose:** one person detection; keypoint tensor shape `1 × 17 × 3` (batch × COCO joints × x/y/conf).
- **Torchvision:** many person *proposals* on the crop; the script **selects one** (highest score, ~0.95). The overlay is still visually noisy on this pose compared to YOLOv8.

### Figures (this folder)

| File | Content |
|------|---------|
| `results/images/01_input_detector_crop.png` | Input crop only |
| `results/images/02_yolov8n_pose_overlay.png` | YOLOv8-Pose overlay |
| `results/images/03_torchvision_keypointrcnn_overlay.png` | Torchvision Keypoint R-CNN overlay |

## Conclusion (for presentation)

- **Prefer YOLOv8-Pose** for this use case: cleaner skeleton on the rescuer, plausible shoulders–arms–torso, and the model does not latch onto the mannequin as a second body.
- **Torchvision Keypoint R-CNN** on this crop produces a cluttered / less reliable skeleton despite a high person score; two-stage COCO R-CNN is weaker than a dedicated pose head for awkward CPR posture.

## Artifacts layout

```
experiments/pose_backend_compare/
  EXPERIMENT.md              ← this document
  README.md
  code/
    compare_pose_backends_on_image.py
    run_experiment.ps1
    README.md
  config/
    README.txt
    baseline_mediapipe.yaml
    baseline_yolov8.yaml
    baseline_hybrid.yaml
  results/
    images/                   ← frozen PNGs for slides
    metrics/                  ← summary JSON + table + OTHER_BACKENDS.txt
    compare_run_summary.json  ← optional raw script output
    related_*_ch1.json        ← optional snapshots from runs/ch1_updated_mppose
  results/regen/              ← scratch (gitignored)
```

**Main product** (outside this folder): Stage 1 readiness + frozen configs in `configs/experiments/` at repo root — not duplicated in `scripts/` for this compare tool.

## Revision

- **Date:** 2026-03-31  
- **Run:** snapshot taken from `runs/pose_compare` outputs; metrics JSON paths in `results/metrics/summary.json` are relative to this experiment folder for portability.
