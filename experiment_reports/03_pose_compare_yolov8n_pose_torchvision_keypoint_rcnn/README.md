# Experiment: Pose backends — YOLOv8n-pose vs Torchvision Keypoint R-CNN

**Task:** Compare two pose estimation backends on **one** CPR **detector crop** (numeric meta and qualitative notes). **Not** a full-dataset benchmark.

## At a glance

The CPR pipeline often runs **person detection first**, then **pose estimation** on a **tight crop** around the rescuer or the pair. This mini-experiment asks: on **one real detector crop** from that workflow, which off-the-shelf pose backend gives **more usable keypoints** for downstream logic (e.g. arm/torso geometry)?

**What ran:** The **same RGB crop** was fed to two models: **Ultralytics YOLOv8n-pose** (lightweight COCO pose) and **Torchvision’s Keypoint R-CNN** with a **ResNet50-FPN** backbone trained on COCO. For each method the run recorded **how many persons were proposed**, the **shape of the keypoint tensor**, and (for R-CNN) the **confidence of the best “person” box**. No retraining was done — these are **pretrained** weights.

**What the numbers mean:** **YOLOv8n-pose** returned **one** detection and a **17-keypoint** map in shape **`[1, 17, 3]`** (x, y, and per-keypoint confidence in the third channel — COCO skeleton). **Keypoint R-CNN** produced **many** person proposals (**18**); the **highest box score** was about **0.95**. That **high classification score does not imply** a neat skeleton on this CPR-like pose: in the original write-up, the **R-CNN overlay looked noisy** for this crop, while **YOLOv8-Pose** was **judged visually better** for **caregiver body layout**. This is a **qualitative + structural** comparison, not a proof that YOLO always wins.

**Limits:** **Single image** only — no mAP, no failure-rate statistics, no timing study. **RTMPose / HRNet (MMPose)** and **AlphaPose** were **not** run. **This folder** has no input PNG or overlay images — text and tables only.

---

## 1. Summary

| Field | Value |
|-------|--------|
| experiment | pose_backend_compare |
| title | YOLOv8-Pose vs Torchvision Keypoint R-CNN on CPR detector crop |
| date | 2026-03-31 |
| recommendation (author note) | YOLOv8-Pose for caregiver skeleton quality on this crop. |
| not evaluated here | RTMPose, HRNet (MMPose), AlphaPose |

---

## 2. Comparison table

| Backend | Model | Person / detections | Keypoints / notes |
|---------|-------|---------------------|-------------------|
| YOLOv8-Pose | `yolov8n-pose` | 1 box | Shape `[1, 17, 3]` — COCO 17 keypoints, x/y + confidence |
| Torchvision Keypoint R-CNN | ResNet50-FPN (COCO) | 18 person proposals; **best person score** used: **0.9496407508850098** | COCO 17; overlay described as visually noisy on this CPR pose in source notes |

**Interpretation:** A high Torchvision **person classification score** does not imply a clean skeleton on this frame; YOLOv8-Pose was judged a better visual fit for caregiver geometry on this crop.

---

## 3. YOLOv8-Pose — meta

| Key | Value |
|-----|--------|
| backend | yolov8_pose |
| model | yolov8n-pose |
| num_detections | 1 |
| keypoints_shape | [1, 17, 3] |

---

## 4. Torchvision Keypoint R-CNN — meta

| Key | Value |
|-----|--------|
| backend | torchvision_keypointrcnn_resnet50_fpn |
| num_person_candidates | 18 |
| person_score (best proposal) | 0.9496407508850098 |

---

## 5. Backends not evaluated

**RTMPose / HRNet (MMPose)**  
Install OpenMMLab stack (see https://github.com/open-mmlab/mmpose ), then run their demo scripts on the crop. Models are not bundled with this experiment.

**AlphaPose**  
Separate project (https://github.com/MVIG-SJTU/AlphaPose ) with its own weights and CUDA setup.

---

## 6. Product defaults (conceptual)

Default product pose stack uses a **hybrid** backend with **yolov8n-pose** together with MediaPipe / Tasks-style landmarking. This experiment supports preferring YOLOv8-Pose for caregiver geometry on tight detector crops.
