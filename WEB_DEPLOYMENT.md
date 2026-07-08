# MicroGhost Web Deployment

This repo now has the requested web structure:

```text
backend/
  model/
    app.py
    Dockerfile
    requirements.txt
    checkpoints/
    microghost/
frontend/
  app/
  package.json
```

The model runs on Hugging Face. The Vercel app is only the user interface plus a small proxy route, so Vercel does not need to install PyTorch or load the checkpoint.

## 1. Deploy the Hugging Face backend

Create a Hugging Face Space:

- SDK: `Docker`
- Hardware: start with the default CPU Basic tier
- Files to upload: everything inside `backend/model`

The backend starts with:

```text
uvicorn app:app --host 0.0.0.0 --port 7860
```

The default model path is:

```text
checkpoints/best_microghost_thermal_v3.pth
```

If you put the checkpoint somewhere else, add this Hugging Face Space variable:

```text
MODEL_PATH=/app/checkpoints/best_microghost_thermal_v3.pth
```

After the Space builds, check:

```text
https://your-user-your-space.hf.space/health
```

## 2. Deploy the Vercel frontend

Create a Vercel project from this repo and set:

```text
Root Directory: frontend
Framework: Next.js
Build Command: npm run build
Output Directory: .next
```

Add this Vercel environment variable:

```text
HF_SPACE_URL=https://your-user-your-space.hf.space
```

If the Hugging Face Space is private, also add:

```text
HF_TOKEN=hf_your_token
```

## 3. CORS

For a public Space, the backend defaults to allowing all origins. For a stricter deployment, add this Hugging Face Space variable:

```text
CORS_ORIGINS=https://your-vercel-app.vercel.app
```

For local frontend development:

```text
CORS_ORIGINS=http://localhost:3000
```

## 4. Local development

Backend:

```powershell
cd backend/model
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload --port 7860
```

Frontend:

```powershell
cd frontend
Copy-Item .env.example .env.local
```

Edit `.env.local`:

```text
HF_SPACE_URL=http://localhost:7860
```

Then run:

```powershell
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

## Notes

- Upload thermal + RGB for best false-positive filtering.
- Upload only thermal or only RGB if that is all you have. The missing modality is replaced with a blank branch.
- `lap_thresh` defaults to `80`.
- Single-image inference uses a more sensitive confidence default of `0.20`.
