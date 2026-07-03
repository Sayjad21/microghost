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

In your first notebook cell, clone the `microghost` repository and install the required dependencies.

```python
# Cell 1: Clone Repository and Install Dependencies
!git clone https://github.com/Sayjad21/microghost.git
%cd microghost

# Install required packages
!pip install opencv-python-headless
!pip install huggingface_hub
!pip install onnx
```

---

## 3. Dataset Preparation

The V2 pipeline requires multiple datasets across its 4 phases. 

### A. HuggingFace Datasets (Gated Auto-Download)
The `ForestPersons` (RGB) and `ForestPersonsIR` (Thermal) datasets are automatically downloaded via `huggingface_hub`. However, because they are **Gated Datasets**, you must provide a HuggingFace Access Token.

**1. Accept the Dataset Terms:**
Go to the HuggingFace dataset page (e.g., `etri-vilab/ForestPersons`) in your browser, log in, and click the button to accept the repository terms.

**2. Set up your Kaggle Secret:**
1. In Kaggle, click **Add-ons -> Secrets** in the top menu.
2. Create a new secret with the Label: `HF_TOKEN`
3. Paste your HuggingFace Access Token (Read) in the Value field.
4. Check the box next to `HF_TOKEN` to attach it to your notebook.

**3. Inject the Secret into the Environment:**
Add this cell to your notebook *before* you run the training script:

```python
# Cell 2: Load HuggingFace Token
from kaggle_secrets import UserSecretsClient
import os

user_secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")
```

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

Now you can start the 4-Phase V2 curriculum. The script will automatically handle anchor optimization, phase switching, dataset concatenation, and CMM single-modality generation.

```bash
# Cell 3: Start V2 Training
!python main.py train --batch-size 16 --num-workers 2
```

> **Note on Workers:** Kaggle CPU cores are limited. If you encounter shared memory errors or hanging DataLoader threads, set `--num-workers 0`.

### What to Expect:
- **Phase 1 (LLVIP)**: Trains the foundational shape representations.
- **Phase 2 (ForestPersons + LLVIP)**: Introduces single-modality jungle data. Triggers the auto-download from HuggingFace.
- **Phase 3 (Camo-M3FD + LLVIP)**: Fine-tunes on camouflage thermal targets using `findContours` mask-to-bbox conversion.
- **Phase 4 (Full Mix)**: Polishes the model across all data.

*Saved weights and logs will be written to `/kaggle/working/microghost/runs/` and `/kaggle/working/microghost/weights/`.*

---

## 5. Evaluation & Inference

Once training is complete, you can evaluate the model's mAP score and run inference on test images.

### Evaluate mAP
```bash
# Cell 4: Evaluate Model
!python main.py evaluate --model-path weights/best_model.pth --dataset llvip
```

### Run Inference on a Sample Image
```bash
# Cell 5: Run Inference
!python main.py infer \
    --model-path weights/best_model.pth \
    --image-rgb data/llvip/visible/test/190001.jpg \
    --image-thermal data/llvip/infrared/test/190001.jpg
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
!python main.py export --model-path weights/best_model.pth --format onnx
```

### Downloading your Files
Kaggle automatically saves anything placed in `/kaggle/working/`. 
To download your final weights and ONNX model:
1. Look at the **Output** section in the right-hand panel.
2. Navigate to `microghost/weights/` and `microghost/export/`.
3. Click the **three dots (...)** next to `best_model.pth` and `microghost.onnx` to download them directly to your local machine.
