Frozen Stage 1 YAML copies (same as configs/experiments/baseline_*.yaml at repo root).

The compare script in ../code/ runs standalone YOLOv8-Pose vs Torchvision on one image.
The full CPR readiness pipeline uses pose.backend from these files.

If you change baselines for the main product, update configs/experiments/ first, then copy
the three baseline_*.yaml files here again so this experiment package stays self-contained.
