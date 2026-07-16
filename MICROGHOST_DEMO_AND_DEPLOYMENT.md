# MicroGhost Demo and Deployment Guide

## Project Stack

- Frontend: Next.js 15 App Router, React 19, TypeScript, CSS.
- Backend on Vercel: Python FastAPI serverless function in `frontend/api/analyze.py`.
- Model runtime: ONNX Runtime CPU provider.
- Image processing: OpenCV, NumPy.
- Runtime telemetry: `psutil` plus Python process timing.
- Hosting target: Vercel, with `frontend/vercel.json` configuring the Python function.

## What Was Added

- Redesigned the website into a polished one-page inference console.
- Kept a light/dark theme button in the top-right header.
- Added a MicroGhost logo area. Put your final logo at `frontend/public/logo.png`; the app automatically uses it. If that file is missing, it shows the built-in MicroGhost fallback mark.
- Added RGB, thermal, and paired-image upload cards with stable previews.
- Added live inference progress with elapsed time and browser-side device hints.
- Added metric widgets for detections, mode, thresholds, wall time, CPU load, CPU clock, RAM, threads, and cores.
- Added recent inference log rows for demo narration.
- Added download buttons for annotated primary and thermal detection images.
- Added threshold controls for Laplacian and optional confidence override.
- Added a merge-related-boxes toggle.
- Added backend performance metrics in the API response.
- Added upload size control and image downscaling to reduce memory pressure on Vercel.
- Added a thermal-only hot-body retention rule so strong human detections are not discarded just because the thermal blob is smooth.
- Added an RGB-only inverted-grayscale proxy channel plus conservative RGB post-filtering to reduce weak false positives.

## Threshold Locations for Experiments

Main backend threshold constants:

- `frontend/api/analyze.py:34` - `CONFIDENCE_THRESHOLD = 0.20`
  - Used when paired RGB + thermal mode does not send a custom confidence threshold.
- `frontend/api/analyze.py:35` - `DEFAULT_LAPLACIAN_THRESHOLD = 80.0`
  - Backend default for `lap_thresh`.
- `frontend/api/analyze.py:36` - `THERMAL_ONLY_CONFIDENCE_THRESHOLD = 0.20`
  - Default confidence when only thermal is uploaded.
- `frontend/api/analyze.py:37` - `RGB_ONLY_CONFIDENCE_THRESHOLD = 0.16`
  - Default confidence when only RGB is uploaded.
- `frontend/api/analyze.py:38` - `RGB_ONLY_MIN_OBJECTNESS = 0.18`
  - Extra RGB-only objectness filter used after inference.
- `frontend/api/analyze.py:613` - API form default: `lap_thresh: float = Form(default=DEFAULT_LAPLACIAN_THRESHOLD)`.
- `frontend/api/analyze.py:641` - thermal-only confidence is selected here.
- `frontend/api/analyze.py:649` - RGB-only confidence is selected here.

Frontend default slider value:

- `frontend/app/page.tsx:63` - `DEFAULT_LAP_THRESHOLD = 80`
- `frontend/app/page.tsx:269` - React state initializes the Laplacian slider from `DEFAULT_LAP_THRESHOLD`.

Current requested behavior:

- RGB inference uses Laplacian threshold `80` by default.
- Thermal inference uses Laplacian threshold `80` by default.
- Paired RGB + thermal inference uses Laplacian threshold `80` by default.
- RGB-only inference uses an inverted grayscale proxy as the thermal channel because the model was trained around thermal signal. This reduces random RGB-only false positives and improves RGB-only detection on the provided `rgb/fp1.png` sample.
- Confidence can be hard-coded in the backend constants above, or changed during a demo by opening "Tune thresholds" and enabling "Override confidence threshold."

## Performance and Optimization Notes

- `frontend/api/analyze.py:382` creates ONNX Runtime session options.
- `frontend/api/analyze.py:383` sets `intra_op_num_threads = 1`.
- `frontend/api/analyze.py:384` sets `inter_op_num_threads = 1`.
- The single-threaded ONNX setup helps avoid serverless CPU oversubscription during live demo traffic.
- `frontend/api/analyze.py:42` limits uploads to 8 MB.
- `frontend/api/analyze.py:43` limits the longest image side to 1600 px.
- `frontend/api/analyze.py:75` builds the RGB-only thermal proxy.
- `frontend/api/analyze.py:493` contains the resize helper used before inference.
- `frontend/api/analyze.py:522` collects runtime CPU/RAM telemetry.
- `frontend/vercel.json:5` gives the Python function up to 300 seconds.
- `frontend/vercel.json:6` sets function memory to 1024 MB.
- `frontend/vercel.json:7` excludes frontend build files from the Python function bundle.

For heavier traffic, the next upgrade would be moving inference to a dedicated API/GPU service and keeping Vercel as the frontend gateway. For the live demo, the current version is optimized for low setup complexity, bounded uploads, smaller image processing, singleton model loading, and visible telemetry.

## Local Verification

From the `frontend` folder:

```bash
npm install
npm run build
```

The production build passed after installing dependencies.

## Deploy This Version to Vercel

This downloaded folder is not currently a Git checkout, so there is no `.git` history or remote configured here. Since your existing repo is already connected to Vercel, the best workflow is to put these edited files into the real Git repo and push a new commit.

### Recommended Git-Connected Workflow

1. Clone your real GitHub repo again, or open the original local clone that has `.git`.

```bash
git clone YOUR_REPO_URL microghost-vercel-update
cd microghost-vercel-update
```

2. Copy these changed files/folders from this downloaded project into that Git checkout:

```text
frontend/app/page.tsx
frontend/app/globals.css
frontend/app/layout.tsx
frontend/api/analyze.py
frontend/requirements.txt
frontend/public/README.md
MICROGHOST_DEMO_AND_DEPLOYMENT.md
```

3. If your Vercel project root is `frontend`, test from inside `frontend`:

```bash
cd frontend
npm install
npm run build
cd ..
```

4. Check the changed files:

```bash
git status
```

5. Commit the update:

```bash
git add frontend/app/page.tsx frontend/app/globals.css frontend/app/layout.tsx frontend/api/analyze.py frontend/requirements.txt frontend/public/README.md MICROGHOST_DEMO_AND_DEPLOYMENT.md
git commit -m "Polish MicroGhost web demo and inference telemetry"
```

6. Push to the branch that Vercel deploys from, usually `main`:

```bash
git push origin main
```

7. Open the Vercel dashboard and watch the new production deployment. If your Vercel project is connected to Git, Vercel automatically builds a deployment from the push.

### CLI Production Deploy Alternative

Use this if you want to deploy without waiting for Git integration, or if the repo is not connected.

```bash
cd frontend
npm install
npm i -g vercel
vercel login
vercel link
vercel deploy --prod
```

If Vercel asks for the project directory, choose the `frontend` directory. If it asks whether to link to an existing project, choose the current live MicroGhost project.

## Demo Talking Points for Judges

- "The frontend is a one-page Next.js inference console for RGB, thermal, or paired inputs."
- "The model runs through ONNX Runtime in a Vercel Python serverless function."
- "The default Laplacian texture threshold is fixed at 80 for RGB, thermal, and paired inference."
- "Thermal-only mode keeps high-confidence hot human bodies even when the body texture is smooth, which improves the 190003/190006/190007 demo frames."
- "RGB-only mode uses an inverted grayscale proxy channel and conservative filtering because the model is thermal-first."
- "The UI exposes threshold tuning so we can demonstrate how texture filtering affects false positives."
- "The live progress bar gives feedback during serverless inference."
- "The metric widgets show runtime, CPU estimate, CPU clock where available, RAM, thread count, and system memory."
- "The detection outputs are downloadable immediately as annotated images."
- "For reliability, uploads are capped, oversized images are resized, and the ONNX session is reused instead of reloading the model each request."

## Vercel References

- Vercel CLI deploy command: https://vercel.com/docs/cli/deploy
- Vercel deployments overview: https://vercel.com/docs/deployments
- Vercel CLI project deployment workflow: https://vercel.com/docs/projects/deploy-from-cli
