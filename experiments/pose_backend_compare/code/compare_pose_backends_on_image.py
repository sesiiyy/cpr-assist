#!/usr/bin/env python3
"""
Compare **available** 2D pose backends on one image — writes one PNG per backend + ``summary.json``.

**Runs in this repo without extra installs**

  * **YOLOv8-Pose** (Ultralytics, COCO 17 keypoints) — always.
  * **Torchvision Keypoint R-CNN** (COCO 17, ResNet50-FPN) — if ``--torchvision``; first run downloads ~340MB weights.

**Not bundled** (heavy / separate projects — see ``OTHER_BACKENDS.txt`` in output):

  * **RTMPose** / **HRNet** (MMPose / OpenMMLab): ``pip install openmim`` + ``mim install mmcv mmpose`` …
  * **AlphaPose**: standalone repo + models.

Run from **repository root** (this file lives under ``experiments/pose_backend_compare/code/``)::

  python experiments/pose_backend_compare/code/compare_pose_backends_on_image.py \\
    --image runs/ch1_updated_mppose/intermediate/detector_crops/crop_00.png \\
    --out-dir runs/pose_compare

See ``experiments/pose_backend_compare/EXPERIMENT.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for d in (here.parent, *here.parents):
        if (d / "src" / "readiness").is_dir():
            return d
    return here.parents[1]


_ROOT = _repo_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# COCO 17 keypoint indices: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles.
COCO_SKELETON = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)


def _run_yolov8_pose(image_bgr: np.ndarray, model_name: str) -> tuple[np.ndarray, dict]:
    from ultralytics import YOLO

    weights = _ROOT / "models" / f"{model_name}.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights) if weights.is_file() else model_name)
    r = model(image_bgr, verbose=False)[0]
    vis = r.plot()
    meta: dict = {
        "backend": "yolov8_pose",
        "model": model_name,
        "num_detections": len(r.boxes) if r.boxes is not None else 0,
    }
    if r.keypoints is not None and len(r.keypoints):
        meta["keypoints_shape"] = list(r.keypoints.data.shape)
    return vis, meta


def _run_torchvision_keypointrcnn(image_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    import torch
    from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    t = torch.as_tensor(rgb).permute(2, 0, 1).float() / 255.0
    model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    with torch.no_grad():
        out = model([t])[0]
    vis = image_bgr.copy()
    labels = out["labels"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    kpts_all = out["keypoints"].cpu().numpy()
    person_idx = np.where(labels == 1)[0]
    meta: dict = {
        "backend": "torchvision_keypointrcnn_resnet50_fpn",
        "num_person_candidates": int(len(person_idx)),
    }
    if len(person_idx):
        best = person_idx[np.argmax(scores[person_idx])]
        if scores[best] >= 0.5:
            meta["person_score"] = float(scores[best])
            i = best
            pts = kpts_all[i]
        else:
            pts = None
    else:
        pts = None
    if pts is not None:
        for (a, b) in COCO_SKELETON:
            if a < len(pts) and b < len(pts):
                pa = (int(pts[a, 0]), int(pts[a, 1]))
                pb = (int(pts[b, 0]), int(pts[b, 1]))
                if pts[a, 2] > 0 and pts[b, 2] > 0:
                    cv2.line(vis, pa, pb, (0, 255, 255), 2)
        for j in range(len(pts)):
            if pts[j, 2] > 0:
                cv2.circle(vis, (int(pts[j, 0]), int(pts[j, 1])), 3, (255, 128, 0), -1)
    cv2.putText(
        vis,
        "torchvision Keypoint R-CNN (COCO 17) — not HRNet",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
    )
    return vis, meta


OTHER_BACKENDS = """RTMPose / HRNet (MMPose)
  Install OpenMMLab stack (see https://github.com/open-mmlab/mmpose ), then run their
  demo scripts on this image path. Models are not included in this repo.

AlphaPose
  Separate repository (https://github.com/MVIG-SJTU/AlphaPose ) with its own weights
  and CUDA setup; not invoked from this script.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare pose backends on one image (YOLOv8 + optional torchvision).")
    parser.add_argument("--image", "-i", type=Path, required=True)
    parser.add_argument("--out-dir", "-o", type=Path, default=Path("runs/pose_compare"))
    parser.add_argument("--yolo-model", type=str, default="yolov8n-pose", help="Ultralytics pose weights name or path.")
    parser.add_argument("--torchvision", action="store_true", help="Also run torchvision Keypoint R-CNN (large download first time).")
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"Not found: {args.image}", file=sys.stderr)
        return 1

    img = cv2.imread(str(args.image))
    if img is None:
        print(f"Could not read: {args.image}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"input": str(args.image.resolve()), "outputs": {}}

    vis_y, meta_y = _run_yolov8_pose(img, args.yolo_model)
    p_y = args.out_dir / "crop_00_yolov8_pose.png"
    cv2.imwrite(str(p_y), vis_y)
    summary["outputs"]["yolov8_pose"] = {"path": str(p_y), "meta": meta_y}

    if args.torchvision:
        vis_t, meta_t = _run_torchvision_keypointrcnn(img)
        p_t = args.out_dir / "crop_00_torchvision_keypointrcnn.png"
        cv2.imwrite(str(p_t), vis_t)
        summary["outputs"]["torchvision_keypointrcnn"] = {"path": str(p_t), "meta": meta_t}

    (args.out_dir / "OTHER_BACKENDS.txt").write_text(OTHER_BACKENDS, encoding="utf-8")
    summary["note"] = "RTMPose, HRNet (MMPose), AlphaPose are not run here — see OTHER_BACKENDS.txt"

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
