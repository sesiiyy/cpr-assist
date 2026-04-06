# CPR Assist

Monorepo for a real-time mobile CPR guidance system. **Assist is standalone:** vision and research code live in [`cpr_ml/`](cpr_ml/README.md) (vendored CPR ML bundle). No sibling `cpr/` checkout is required at runtime.

## Datasets

### Primary Dataset (Proposed Contribution)

A novel dataset was created for this research based on HRCT sagittal chest images giving patient specific CPR compression depth modeling.

Dataset link: https://www.kaggle.com/datasets/sesandirajapakse/high-resolution-ct-thoracic-cpr-depth-dataset

This dataset provides annotated thoracic regions and anatomical boundaries used to derive compression depth ranges.


### Secondary Dataset (Benchmark / Validation)

The system was also evaluated using publicly available CPR related datasets for comparison and validation purposes.

CPR Coach Dataset:https://huggingface.co/datasets/ShunliWang/CPR-Coach


## Dataset Contribution

The primary HRCT dataset introduced in this work represents a novel contribution to CPR research by enabling anatomically grounded depth estimation rather than relying solely on generalized guidelines.

## Projects

- `backend`: FastAPI + MongoDB API
- `cpr_ml`: vendored `src/`, `api/cpr_api/`, `configs/`, and experiment checkpoints (see `cpr_ml/README.md`)
- `mobile`: Expo React Native app
- `shared`: shared API contracts and types
- `infra`: local infrastructure helpers
- `docs`: architecture and operations documentation

## Quick Start

### Backend

1. Copy **model weights** into `cpr_ml/experiments/...` (they are not in Git). See [`cpr_ml/README.md`](cpr_ml/README.md) for paths and `scp`/`rsync` examples.
2. `cd backend` (from repo root: `cpr-assist/backend`)
3. `python3 -m venv .venv && source .venv/bin/activate`
4. `pip install -e .`
5. `cp .env.example .env` — MongoDB + JWT; optional `CPR_ML_ROOT`, `CPR_CONFIG_PATH`, `CPR_FORCE_DEVICE`, `CPR_HARNESS_TTL_SECONDS`, `INFERENCE_TIMEOUT_SEC`
6. `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`

Optional seed data:
- `python3 scripts/seed_demo_data.py`

### Mobile

1. `cd mobile`
2. `npm install`
3. `npx expo start`
4. Optional checks:
   - `npm run typecheck`
   - `npm test`
5. For a clean simulator run when UI seems stale:
   - stop Metro
   - `watchman watch-del-all || true`
   - `rm -rf mobile/.expo`
   - `npx expo start --clear`

Use demo credentials after seeding:
- `caregiver@example.com / password123`

  ## System Architecture

The system consists of two main components:

1. Motion Analysis Module  
   - Extracts body movement from video input  
   - Tracks compression cycles and rate  

2. Anatomical Modeling Module  
   - Uses HRCT sagittal chest data  
   - Defines patient-specific compression depth range  

These modules are integrated to provide real-time CPR feedback.

---

## How to Run

1. Navigate to the implementation folder
2. Install dependencies
3. Run the main application file

(Note: Full deployment requires mobile application setup and camera input.)

---

## Evaluation

System performance was evaluated using:
- Simulated CPR scenarios
- Variation in lighting, angles, and participants
- Comparison with standard CPR guidelines (100–120 CPM)

Results and analysis are included in the `testing-report/` directory.

---

## Repository Link (For Thesis)

This repository is submitted as part of the undergraduate thesis requirement.
