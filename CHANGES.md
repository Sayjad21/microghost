# MicroGhost-Thermal — Change Log (V2 Improvement)

## Diagnosis (from your screenshot)

| | V1 | V2 |
|---|---|---|
| Precision | 0.6068 | 0.6549 (+5%) |
| Recall | 0.6011 | **0.5829 (-3%)** |
| F1 | 0.6039 | 0.6168 (+1.3%) |
| FP | 3270 | 2565 |
| FN | 3348 | **3483** |

V2 is more conservative (fewer predictions = fewer FPs, but MORE missed targets).  
**Root cause:** V1+V2 were both trained on **LLVIP** (nighttime pedestrians only) but  
the val set (`get_val_base_dataset`) is also LLVIP test split — yet the scores are only ~0.62.  
This means the model generalizes poorly even within LLVIP. Training directly on CAMO M3FD  
removes the domain gap and is the **single biggest lever**.

---

## Changes — MINIMAL, TARGETED

### Summary of 4 files changed

| File | What changes | Why |
|---|---|---|
| `config.py` | Add CAMO M3FD config + bump `OBJ_WEIGHT` 5→7 | Enable dataset + improve recall |
| `data_loading.py` | Add `CAMOD3FDDataset` class + wire into factory | Core: loads new dataset |
| `main.py` | Add `camod3fd` to choices + `--resume-v1` flag | Enable fine-tuning from V1 |
| `kaggle_DRIVER.ipynb` | Point to CAMO M3FD dataset | Correct Kaggle paths |

---

## Change 1 — `config.py`

### 1a. In `DATASET_SUBDIRS` dict, add one line:
```python
DATASET_SUBDIRS = {
    'llvip': 'llvip',
    'kaist': 'kaist-multispectral',
    'flirv2': 'FLIR_ADAS_v2',
    'camod3fd': 'camo-m3fd',          # ← ADD THIS LINE
}
```

### 1b. In `DATASET_CONFIGS` dict, add the new block BEFORE the closing `}`:
```python
    'camod3fd': {
        'name': 'CAMO M3FD',
        'description': 'Multi-Modal Multi-Spectral Detection — CAMO variant',
        'train_imgs':    'train/Imgs',
        'train_thermal': 'train/Thermal',
        'train_gt':      'train/GT',
        'val_imgs':      'val/Imgs',
        'val_thermal':   'val/Thermal',
        'val_gt':        'val/GT',
        'annotation_format': 'voc_xml_or_yolo',
        'classes': ['People', 'Car', 'Bus', 'Motorcycle', 'Lamp', 'Truck'],
        'image_ext': ['.jpg', '.png', '.jpeg'],
        'native_resolution': (640, 480),
    },
```

### 1c. Bump OBJ_WEIGHT to improve recall (line ~177):
```python
# BEFORE
OBJ_WEIGHT = 5.0
# AFTER
OBJ_WEIGHT = 7.0          # Raised: V2 recall was 0.58, need more objectness signal
```

---

## Change 2 — `data_loading.py`

### 2a. Insert the new dataset class BEFORE the `ThermalIntrusionDataset` class (before line 503).

Paste this entire block between the FLIRv2Dataset and ThermalIntrusionDataset:

```python
# ============================================================================
# CAMO M3FD DATASET LOADER
# ============================================================================

class CAMOD3FDDataset(Dataset):
    """
    CAMO M3FD: Multi-Modal Multi-Spectral Detection Dataset.

    Structure expected:
    ```
    camo-m3fd/
    ├── train/
    │   ├── GT/        <- VOC XML (.xml) or YOLO txt (.txt) annotations
    │   ├── Imgs/      <- RGB images
    │   └── Thermal/   <- Thermal images
    ├── val/
    │   └── ...
    └── test/
        └── ...
    ```

    Class mapping:
      People / Person / Pedestrian → person_visible (class 1)
      Car / Bus / Motorcycle / Truck → vehicle_boat (class 3)
      Lamp → ignored
    """

    # M3FD label → MicroGhost class_id
    _NAME_TO_ID = {
        'people':       CLASS_MAP['person_visible'],
        'person':       CLASS_MAP['person_visible'],
        'pedestrian':   CLASS_MAP['person_visible'],
        'human':        CLASS_MAP['person_visible'],
        'car':          CLASS_MAP['vehicle_boat'],
        'bus':          CLASS_MAP['vehicle_boat'],
        'motorcycle':   CLASS_MAP['vehicle_boat'],
        'motorbike':    CLASS_MAP['vehicle_boat'],
        'truck':        CLASS_MAP['vehicle_boat'],
        'van':          CLASS_MAP['vehicle_boat'],
        'lamp':         None,   # intentionally ignored
    }

    # YOLO class index → MicroGhost class_id (M3FD ordering)
    # 0=People, 1=Car, 2=Bus, 3=Motorcycle, 4=Lamp, 5=Truck
    _YOLO_IDX_TO_ID = {
        0: CLASS_MAP['person_visible'],
        1: CLASS_MAP['vehicle_boat'],
        2: CLASS_MAP['vehicle_boat'],
        3: CLASS_MAP['vehicle_boat'],
        4: None,
        5: CLASS_MAP['vehicle_boat'],
    }

    def __init__(self, root_dir, split='train', verbose=True):
        self.root_dir = root_dir
        self.split = split
        self.verbose = verbose

        self.img_dir     = os.path.join(root_dir, split, 'Imgs')
        self.thermal_dir = os.path.join(root_dir, split, 'Thermal')
        self.gt_dir      = os.path.join(root_dir, split, 'GT')

        self.parse_errors    = []
        self.unknown_classes = set()
        self.paired_paths    = []

        if not os.path.exists(self.img_dir):
            if verbose:
                print(f"⚠️  CAMO M3FD Imgs dir not found: {self.img_dir}")
            return

        for ext in ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'):
            for img_path in sorted(glob(os.path.join(self.img_dir, ext))):
                stem = os.path.splitext(os.path.basename(img_path))[0]

                # Match thermal (same stem, any common image extension)
                thermal_path = None
                for t_ext in ('.jpg', '.png', '.jpeg', '.JPG', '.PNG'):
                    tp = os.path.join(self.thermal_dir, stem + t_ext)
                    if os.path.exists(tp):
                        thermal_path = tp
                        break

                if thermal_path is None:
                    continue  # skip if no thermal pair

                # Match annotation: prefer XML, fall back to YOLO txt
                gt_path = None
                for g_ext in ('.xml', '.txt'):
                    gp = os.path.join(self.gt_dir, stem + g_ext)
                    if os.path.exists(gp):
                        gt_path = gp
                        break

                self.paired_paths.append((img_path, thermal_path, gt_path))

        if verbose:
            found = sum(1 for _, _, g in self.paired_paths if g is not None)
            print(f"📁 CAMO M3FD [{split}]: {len(self.paired_paths)} pairs, "
                  f"{found} annotations found")

    # ------------------------------------------------------------------
    def _parse_annotation(self, gt_path, w_orig, h_orig):
        if gt_path is None or not os.path.exists(gt_path):
            return []
        if gt_path.endswith('.xml'):
            return self._parse_xml(gt_path, w_orig, h_orig)
        return self._parse_yolo(gt_path, w_orig, h_orig)

    def _clamp_box(self, xmin, ymin, xmax, ymax, w, h):
        xmin = max(0, min(int(xmin), w - 1))
        ymin = max(0, min(int(ymin), h - 1))
        xmax = max(xmin + 1, min(int(xmax), w))
        ymax = max(ymin + 1, min(int(ymax), h))
        return xmin, ymin, xmax, ymax

    def _parse_xml(self, xml_path, w_orig, h_orig):
        annotations = []
        try:
            root = ET.parse(xml_path).getroot()
            for obj in root.findall('object'):
                name_el = obj.find('name')
                if name_el is None:
                    continue
                name = name_el.text.strip().lower()
                class_id = self._NAME_TO_ID.get(name)
                if class_id is None:
                    if name != 'lamp':
                        self.unknown_classes.add(name)
                    continue
                bb = obj.find('bndbox')
                if bb is None:
                    continue
                xmin, ymin, xmax, ymax = self._clamp_box(
                    float(bb.find('xmin').text), float(bb.find('ymin').text),
                    float(bb.find('xmax').text), float(bb.find('ymax').text),
                    w_orig, h_orig,
                )
                annotations.append({
                    'class_id': class_id,
                    'xmin': xmin, 'ymin': ymin,
                    'xmax': xmax, 'ymax': ymax,
                })
        except ET.ParseError as e:
            self.parse_errors.append((xml_path, str(e)))
        except Exception as e:
            self.parse_errors.append((xml_path, str(e)))
        return annotations

    def _parse_yolo(self, txt_path, w_orig, h_orig):
        """YOLO format: class_idx cx cy w h (all normalized 0-1)."""
        annotations = []
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_idx = int(parts[0])
                    class_id = self._YOLO_IDX_TO_ID.get(cls_idx)
                    if class_id is None:
                        continue
                    cx, cy, w, h = (float(x) for x in parts[1:5])
                    xmin, ymin, xmax, ymax = self._clamp_box(
                        (cx - w / 2) * w_orig, (cy - h / 2) * h_orig,
                        (cx + w / 2) * w_orig, (cy + h / 2) * h_orig,
                        w_orig, h_orig,
                    )
                    annotations.append({
                        'class_id': class_id,
                        'xmin': xmin, 'ymin': ymin,
                        'xmax': xmax, 'ymax': ymax,
                    })
        except Exception as e:
            self.parse_errors.append((txt_path, str(e)))
        return annotations

    # ------------------------------------------------------------------
    def report_issues(self):
        if self.parse_errors:
            print(f"  ⚠️  {len(self.parse_errors)} annotation parse errors")
        if self.unknown_classes:
            print(f"  ℹ️  Unknown classes skipped: {self.unknown_classes}")

    def iter_annotations(self):
        """Yield (annotations, (h, w)) without loading images — for anchor K-Means."""
        for img_path, _, gt_path in self.paired_paths:
            img = cv2.imread(img_path)
            h_orig, w_orig = img.shape[:2] if img is not None else (480, 640)
            yield self._parse_annotation(gt_path, w_orig, h_orig), (h_orig, w_orig)

    def __len__(self):
        return len(self.paired_paths)

    def __getitem__(self, idx):
        """Returns: (image_rgb, image_thermal), annotations, (h_orig, w_orig)"""
        img_path, thermal_path, gt_path = self.paired_paths[idx]

        # Load RGB
        image_rgb = cv2.imread(img_path)
        if image_rgb is None:
            raise ValueError(f"Failed to load RGB: {img_path}")
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image_rgb.shape[:2]

        # Load Thermal (grayscale)
        image_thermal = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
        if image_thermal is None:
            # Try color load + convert
            tmp = cv2.imread(thermal_path)
            if tmp is not None:
                image_thermal = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
            else:
                # Last resort: dark gray frame
                image_thermal = np.zeros((h_orig, w_orig), dtype=np.uint8)

        annotations = self._parse_annotation(gt_path, w_orig, h_orig)
        return (image_rgb, image_thermal), annotations, (h_orig, w_orig)
```

### 2b. In `create_dataloaders()` — add the `camod3fd` branch

Find the block:
```python
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Supported: llvip, kaist, flirv2")
```

Replace it with:
```python
    elif dataset_name == 'camod3fd':
        raw_train = CAMOD3FDDataset(dataset_root, split='train')
        raw_val   = CAMOD3FDDataset(dataset_root, split='val')

        train_dataset = ThermalIntrusionDataset(raw_train, preprocessor, augment=True)
        val_dataset   = ThermalIntrusionDataset(raw_val,   preprocessor, augment=False)

        raw_train.report_issues()

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Supported: llvip, kaist, flirv2, camod3fd")
```

### 2c. In `get_val_base_dataset()` — add the `camod3fd` case

Find:
```python
    raise ValueError(f"Unknown dataset: {dataset_name}")
```

Add BEFORE it:
```python
    if dataset_name == 'camod3fd':
        return CAMOD3FDDataset(dataset_root, split='val', verbose=False)

```

---

## Change 3 — `main.py`

### 3a. In `train_parser` args, extend `--dataset` choices:
```python
# BEFORE
choices=['llvip', 'kaist', 'flirv2']
# AFTER (both occurrences in train_parser and eval_parser)
choices=['llvip', 'kaist', 'flirv2', 'camod3fd']
```

### 3b. Add `--resume-v1` flag to `train_parser` (after the `--no-kmeans` line):
```python
    train_parser.add_argument('--resume-v1', action='store_true',
                              help='Load V1 checkpoint weights as starting point for V2 training')
```

### 3c. In `main()`, after `model = MicroGhostThermal()` (around line 116), add:
```python
        # --- Fine-tune from V1 if requested ---
        if args.resume_v1:
            from config import BEST_MODEL_V1_PATH, DEVICE as _DEVICE
            if os.path.exists(BEST_MODEL_V1_PATH):
                ckpt = torch.load(BEST_MODEL_V1_PATH, map_location=_DEVICE)
                model.load_state_dict(ckpt['model_state_dict'], strict=False)
                print(f"✅ Loaded V1 weights from {BEST_MODEL_V1_PATH} (fine-tune mode)")
            else:
                print(f"⚠️  --resume-v1 set but {BEST_MODEL_V1_PATH} not found — training from scratch")
```

---

## Change 4 — `kaggle_DRIVER.ipynb` (the notebook cell that runs training)

Replace the big cell that searches for LLVIP with:

```python
import os

data_root = None
base_input = '/kaggle/input'

# Auto-detect CAMO M3FD dataset
if os.path.exists(base_input):
    for item in os.listdir(base_input):
        candidate = os.path.join(base_input, item)
        if os.path.isdir(os.path.join(candidate, 'train', 'Imgs')):
            data_root = candidate
            break

if data_root:
    print(f"✅ Found CAMO M3FD dataset at: {data_root}")
    # Train V2 fine-tuned from V1 on CAMO M3FD
    !python main.py train \
        --dataset camod3fd \
        --data-root {data_root} \
        --batch-size 32 \
        --num-workers 2 \
        --resume-v1

    print("\n🚀 Training Complete! Running Model Comparison...")
    !python compare_models.py --dataset camod3fd --data-root {data_root}
else:
    print("❌ CAMO M3FD dataset not found in /kaggle/input")
    print("Mount the dataset: hvelesaca/camo-m3fd")
    for item in os.listdir(base_input):
        print(f" - {item}")
```

---

## How to run on Kaggle

```bash
# Train V2 from V1 weights on CAMO M3FD
python main.py train \
    --dataset camod3fd \
    --data-root /kaggle/input/camo-m3fd \
    --batch-size 32 \
    --num-workers 2 \
    --resume-v1

# Compare V1 vs V2
python compare_models.py --dataset camod3fd --data-root /kaggle/input/camo-m3fd
```

---

## Expected Score Impact

| Change | Impact | Reason |
|---|---|---|
| CAMO M3FD training data | **+++ F1** | Eliminates domain gap — biggest lever |
| Fine-tune from V1 | **++ F1** | Faster convergence, better initialization |
| OBJ_WEIGHT 5→7 | **+ Recall** | More objectness signal, reduces FN |
| Together | ~F1 0.68–0.72 target | Conservative estimate |

> If CAMO M3FD's `val/` folder is small, also train on it using `--no-kmeans` once you have  
> anchor stats from the train set. The test set should remain untouched for final eval.
