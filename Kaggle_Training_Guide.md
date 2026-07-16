# MicroGhost-V2: Kaggle Training & Execution Guide

This guide provides step-by-step instructions on how to set up, train, and evaluate the **MicroGhost-V2** model using Kaggle Notebooks. Kaggle provides free GPU access (e.g., P100, T4x2), which is perfect for executing our multi-phase training pipeline.

---

## 1. Setting Up the Kaggle Notebook

1. **Create a New Notebook**: Go to [Kaggle](https://www.kaggle.com/) -> **Create** -> **New Notebook**.
2. **Enable GPU**: 
   - On the right-hand panel, under **Notebook options**, click **Accelerator**.
   - Select **GPU T4 x2** or **GPU P100**.
3. **Enable Internet**:
   - In the same panel, ensure **Internet** is toggled **On**. This is critical for auto-downloading HuggingFace datasets and cloning the repository.
4. **Persistence (Optional)**:
   - Set **Persistence** to **Files only** if you want your downloaded weights to persist across notebook sessions.

---

## 2. Setting Up the Codebase

Because this local copy contains patches that are not being pushed to GitHub, upload this patched project folder to Kaggle as a dataset and attach it to the notebook. The updated `kaggle_DRIVER.ipynb` will copy the attached patched folder into `/kaggle/working/microghost`. It only falls back to GitHub if no attached patched copy is found.

```python
# Cell 1 in kaggle_DRIVER.ipynb copies the attached patched project.
# Then install dependencies:
%cd /kaggle/working/microghost
!pip install -r requirements.txt -q
!pip install onnxscript -q
```

---

## 3. Dataset Preparation

The V2 pipeline requires multiple datasets across its 3 phases. 

### A. HuggingFace Datasets (ForestPersons & ForestPersonsIR)
**CRITICAL:** Do NOT attempt to auto-download these directly into the Kaggle notebook using the HuggingFace CLI. These datasets contain nearly 100,000 tiny image files. Writing that many small files directly to `/kaggle/working/` will trigger a severe Kaggle Disk I/O bottleneck, stalling the notebook for hours.

Instead, follow this workflow to mount them as read-only, high-speed Kaggle Datasets:

1. **Download Locally**: Use the provided `download_subset.py` script on your local PC to download exactly 5,000 images from both `ForestPersons` and `ForestPersonsIR`. (Ensure your `HF_TOKEN` is set).
2. **Extract Annotations**: On your PC, extract the `annotations.zip` inside each folder so that the `annotations` folder sits next to the `images` folder. Delete the `.zip` file afterwards.
3. **Zip and Upload**: Zip the main `ForestPersons` and `ForestPersonsIR` folders, and upload them to Kaggle as a **New Dataset** via the Kaggle web UI.
4. **Attach to Notebook**: In your Kaggle notebook, click **Add Data** -> **Your Work**, and attach the two datasets you just created.

*Because of our updated `config.py`, the pipeline will automatically discover these datasets in `/kaggle/input/` and load them at maximum speed!*

### B. Kaggle Datasets (LLVIP & Camo-M3FD)
For **LLVIP** (Phase 1) and **Camo-M3FD** (Phase 3), it is highly recommended to use Kaggle Datasets to avoid downloading gigabytes of data every session.

1. On the right-hand panel of your notebook, click **Add Data**.
2. Search for the datasets:
   - Search for **LLVIP** (e.g., uploaded by a community member) and click **+**.
   - Search for **M3FD** or **CAMO-M3FD** and click **+**.
3. Kaggle mounts these datasets in the `/kaggle/input/` directory.

### C. Linking Datasets to the Code
Because I have just added **Kaggle Auto-Discovery** to the pipeline, you do **NOT** need to create symbolic links manually anymore!

As long as you clicked "Add Data" and the datasets are somewhere in `/kaggle/input/`, the code will automatically scan the input directory, find the `visible`/`infrared` folders for LLVIP, and automatically link them. 

You can completely skip the `!ln -s` commands in Cell 2.


---

## 4. Running the V2 Multi-Phase Training

### Pipeline Dry Run (Testing)
Before launching a multi-hour training session, it is highly recommended to run a quick test to verify all dataset paths and shapes are correct.
Place this in your notebook to run a single batch/single epoch test:

```python
# 1. Clear out the persistent storage from previous failed attempts
!rm -rf /kaggle/working/*

import sys
sys.path.append('/kaggle/working/microghost')
import config

# TURN THIS ON TO TEST 1 BATCH / 1 EPOCH
config.DEBUG_MODE = True
print(f"DEBUG_MODE is set to: {config.DEBUG_MODE}")
```

### Full Training
Once the debug run succeeds, set `config.DEBUG_MODE = False` (or remove the line entirely) and execute the actual training script. The script will automatically handle anchor optimization, phase switching, dataset concatenation, and CMM single-modality generation.

```bash
# Start V2 Training with safe Kaggle settings
!python main.py train --batch-size 16 --num-workers 2
```

> **Note on Workers:** Kaggle CPU cores are limited. If you encounter shared memory errors or hanging DataLoader threads, set `--num-workers 0`.
> **Note on Stability:** The code now clips gradients using `MICROGHOST_MAX_GRAD_NORM` (default `2.0`) and skips non-finite batches/optimizer steps instead of corrupting the run.

### What to Expect:
- **Phase 1 (LLVIP)**: Trains the foundational shape representations.
- **Phase 2 (ForestPersons + LLVIP)**: Introduces single-modality jungle data. Triggers the auto-download from HuggingFace.
- **Phase 3 (Full Mix Polish)**: Polishes LLVIP, ForestPersons RGB, and ForestPersonsIR thermal together with gate regularization.

Saved weights are written to `/kaggle/working/microghost/checkpoints/`, exported models to `/kaggle/working/microghost/exports/`, and visual inference outputs to `/kaggle/working/microghost/runs/`.

---

## 5. Evaluation & Inference

Once training is complete, you can evaluate the model's mAP score and run inference on test images.

### Evaluate mAP
```bash
# Cell 4: Evaluate Model
!python main.py evaluate --model-path checkpoints/best_microghost_thermal_v3.pth --dataset llvip
!python main.py evaluate --model-path checkpoints/best_microghost_thermal_v3.pth --dataset forestpersons --conf-threshold 0.05
```

### Run Inference on a Sample Image
```bash
# Cell 5: Run Inference
!python main.py infer \
    --model-path checkpoints/best_microghost_thermal_v3.pth \
    --image-rgb data/llvip/visible/test/190001.jpg \
    --image-thermal data/llvip/infrared/test/190001.jpg
```

For ForestPersons RGB-only inference, omit `--image-thermal`; the script creates a blank thermal branch automatically:

```bash
!python main.py infer \
    --model-path checkpoints/best_microghost_thermal_v3.pth \
    --image-rgb /kaggle/input/your-forestpersons-dataset/ForestPersons/images/example.jpg \
    --conf-thresh 0.05 \
    --lap-thresh 0
```

For ForestPersonsIR thermal-only inference, omit `--image-rgb`; the script creates a blank RGB branch automatically:

```bash
!python main.py infer \
    --model-path checkpoints/best_microghost_thermal_v3.pth \
    --image-thermal /kaggle/input/your-forestpersonsir-dataset/ForestPersonsIR/images/example.jpg \
    --conf-thresh 0.10 \
    --lap-thresh 0
```

The annotated images will be saved in the `runs/` folder. You can view them directly in the Kaggle notebook:

```python
# Cell 6: View Inference Results
import matplotlib.pyplot as plt
import cv2

# Load and convert from BGR to RGB
img = cv2.imread('runs/190001_det.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis('off')
plt.show()
```

---

## 6. Exporting the Model

To deploy the trained model to edge hardware, export it to ONNX format:

```bash
# Cell 7: Export to ONNX
!python main.py export --model-path checkpoints/best_microghost_thermal_v3.pth --format onnx
```

### Downloading your Files
Kaggle automatically saves anything placed in `/kaggle/working/`. 
To download your final weights and ONNX model:
1. Look at the **Output** section in the right-hand panel.
2. Navigate to `microghost/checkpoints/` and `microghost/exports/`.
3. Click the **three dots (...)** next to `best_microghost_thermal_v3.pth` and `microghost_thermal.onnx` to download them directly to your local machine.
