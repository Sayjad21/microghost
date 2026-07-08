---
title: MicroGhost Thermal Inference
colorFrom: green
colorTo: cyan
sdk: docker
pinned: false
license: mit
---

# MicroGhost Thermal Inference Space

This folder is the Hugging Face Space backend for MicroGhost.

It exposes:

- `GET /health`
- `POST /analyze`

`POST /analyze` accepts multipart form uploads:

- `thermal_image`: optional thermal image file
- `rgb_image`: optional RGB image file
- `conf_thresh`: optional float confidence threshold
- `lap_thresh`: optional float Laplacian texture threshold, default `80`
- `lap_bypass_conf`: optional float confidence value that bypasses the Laplacian gate

At least one of `thermal_image` or `rgb_image` is required. If only one image is provided, the missing modality is replaced with a blank branch, matching the local thermal-only workflow.

## Local Run

```powershell
cd backend/model
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload --port 7860
```

Then open:

```text
http://localhost:7860/health
```

## Hugging Face Deployment

Create a new Hugging Face Space with SDK set to `Docker`, then upload the contents of this `backend/model` folder to that Space.

The checkpoint expected by default is:

```text
checkpoints/best_microghost_thermal_v3.pth
```

If you rename or move it, set this Space environment variable:

```text
MODEL_PATH=/app/checkpoints/your_model_file.pth
```

For a public Vercel app, also set:

```text
CORS_ORIGINS=https://your-vercel-domain.vercel.app
```

For local frontend testing:

```text
CORS_ORIGINS=http://localhost:3000
```
