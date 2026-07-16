# MicroGhost Free Web Deployment

Use this deployment path when you want everything on free tiers:

- Hugging Face Gradio Space on CPU Basic or ZeroGPU for inference
- Vercel free project for the frontend

Project structure:

```text
backend/
  model/
    app.py
    requirements.txt
    README.md
    checkpoints/
    microghost/
frontend/
  app/
  package.json
```

The model runs on Hugging Face. Vercel only hosts the user interface and forwards uploads to Hugging Face.

## 1. Push this repo to GitHub

```powershell
cd "D:\Hackathons\Ibtida_Saif_Sayjad_trio\FINAL\microghost"
git add .gitignore inference.py main.py run_infer.py WEB_DEPLOYMENT.md backend frontend checkpoints/best_microghost_thermal_v3.pth
git commit -m "Add free web app deployment"
git push origin main
```

## 2. Create the free Hugging Face backend

Go to:

```text
https://huggingface.co/new-space
```

Create the Space:

```text
Name: microghost-thermal
SDK: Gradio
Visibility: Public
Hardware: CPU Basic, or ZeroGPU if CPU Basic is locked
```

Upload or push the contents of:

```text
backend/model
```

to the root of that Hugging Face Space repo.

The Space root should contain:

```text
app.py
requirements.txt
README.md
checkpoints/best_microghost_thermal_v3.pth
microghost/config.py
microghost/inference.py
microghost/model.py
microghost/preprocessing.py
```

After the build finishes, test:

```text
https://YOUR_USERNAME-microghost-thermal.hf.space/health
```

Expected:

```json
{"ok": true, "sdk": "gradio", "model_exists": true}
```

If you use ZeroGPU, the backend includes `@spaces.GPU(duration=60)` so the Space passes the ZeroGPU startup check.

## 3. Deploy the Vercel frontend

Go to Vercel and import your GitHub repo.

Use:

```text
Root Directory: frontend
Framework Preset: Next.js
Build Command: npm run build
Install Command: npm install
```

Add this Vercel environment variable:

```text
HF_SPACE_URL=https://YOUR_USERNAME-microghost-thermal.hf.space
```

Deploy.

## 4. Test the live app

Open the Vercel URL.

Try:

- Thermal only
- RGB only
- RGB + thermal

Use advanced tuning:

- `lap_thresh=80` as the normal starting point
- `conf_thresh=0.18` to `0.20` for small or crouched people
- `lap_thresh=100` to `120` for hot bonnet false positives

## 5. Updating later

Frontend changes:

```powershell
cd "D:\Hackathons\Ibtida_Saif_Sayjad_trio\FINAL\microghost"
git add frontend
git commit -m "Update frontend"
git push origin main
```

Backend changes:

Copy the changed files from `backend/model` into your Hugging Face Space repo, then:

```powershell
git add .
git commit -m "Update backend"
git push
```

Hugging Face rebuilds the Space after every push.
