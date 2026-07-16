---
title: MicroGhost Thermal Inference
emoji: 🔥
colorFrom: green
colorTo: cyan
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
---

# MicroGhost Thermal Inference Space

This is the free Hugging Face Gradio Space backend for MicroGhost.

It gives you:

- A small Gradio UI at `/ui`
- `GET /health`
- `POST /analyze` for the Vercel frontend

`POST /analyze` accepts multipart form uploads:

- `thermal_image`: optional thermal image file
- `rgb_image`: optional RGB image file
- `conf_thresh`: optional float confidence threshold
- `lap_thresh`: optional float Laplacian texture threshold, default `80`
- `lap_bypass_conf`: optional float confidence value that bypasses the Laplacian gate

At least one of `thermal_image` or `rgb_image` is required. If only one image is provided, the missing modality is replaced with a blank branch, matching the local thermal-only workflow.

## Hugging Face Deployment

Create a new Hugging Face Space with:

```text
SDK: Gradio
Hardware: CPU Basic
Visibility: Public
```

Then upload the contents of this `backend/model` folder to that Space.

The checkpoint expected by default is:

```text
checkpoints/best_microghost_thermal_v3.pth
```

If you rename or move it, set this Space variable:

```text
MODEL_PATH=/home/user/app/checkpoints/your_model_file.pth
```

For a public Vercel app, optional stricter CORS:

```text
CORS_ORIGINS=https://your-vercel-domain.vercel.app
```

For local frontend testing:

```text
CORS_ORIGINS=http://localhost:3000
```
