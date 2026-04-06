# Code — pose backend comparison

- **`compare_pose_backends_on_image.py`** — self-contained in this experiment folder (not under `scripts/`).

**Run from repo root** (`cpr/`):

```powershell
.\.venv\Scripts\python.exe experiments\pose_backend_compare\code\compare_pose_backends_on_image.py `
  --image runs\ch1_updated_mppose\intermediate\detector_crops\crop_00.png `
  --out-dir experiments\pose_backend_compare\results\regen `
  --torchvision
```

Or: `.\experiments\pose_backend_compare\code\run_experiment.ps1`

`--torchvision` downloads COCO Keypoint R-CNN weights on first use (~340MB).
