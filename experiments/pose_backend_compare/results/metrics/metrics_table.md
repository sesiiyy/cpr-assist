# Numerical summary — pose backend comparison (crop_00)

| Backend | Model | Person / detections | Keypoints / notes |
|---------|-------|---------------------|-------------------|
| YOLOv8-Pose | `yolov8n-pose` | 1 box | Shape `[1, 17, 3]` (COCO 17, x/y + conf) |
| Torchvision Keypoint R-CNN | ResNet50-FPN COCO | 18 person proposals; **best score** used: **0.950** | COCO 17; overlay still visually noisy on this CPR pose |

**Interpretation:** High Torchvision *classification score* does not imply a clean skeleton here; YOLOv8-Pose is the better visual fit for caregiver geometry.
