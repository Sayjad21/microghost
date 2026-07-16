"""
MicroGhost-Thermal: Configuration Module (V2)
===============================================
Central configuration for MicroGhost V2 dual-branch architecture.

V2 Changes:
- Dual independent branches (RGB + Thermal) with separate weights
- EnergyGate fusion at Scale 2
- BiFusion Neck (replaces FPN)
- 3 anchors per cell
- Phase-based training with CMM
- Accuracy-first design (optimize for ESP32 later)
"""

import os
import torch

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    torch.backends.cudnn.benchmark = True

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.environ.get('MICROGHOST_DATA_ROOT', os.path.join(_PROJECT_DIR, 'data'))

# Per-dataset subdirectory names under DATASET_ROOT
DATASET_SUBDIRS = {
    'llvip': 'llvip',
    'forestpersons': 'ForestPersons',
    'forestpersonsir': 'ForestPersonsIR',
    'camod3fd': 'camo-m3fd',
    # Legacy (loaders kept for backward compat, not used in V2 training)
    'kaist': 'kaist-multispectral',
    'flirv2': 'FLIR_ADAS_v2',
}

# HuggingFace dataset identifiers (for auto-download)
HUGGINGFACE_DATASETS = {
    'forestpersons': 'etri/ForestPersons',
    'forestpersonsir': 'etri/ForestPersonsIR',
}


def get_dataset_path(dataset_name):
    print(f'[DEBUG] Resolving path for {dataset_name}...')
    """Resolve dataset root path (env override or default under DATASET_ROOT)."""
    env_key = f'MICROGHOST_{dataset_name.upper()}_PATH'
    if env_key in os.environ:
        return os.environ[env_key]
    subdir = DATASET_SUBDIRS.get(dataset_name, dataset_name)
    path = os.path.join(DATASET_ROOT, subdir)
    if os.path.isdir(path):
        print(f'[DEBUG] Fallback path for {dataset_name}: {path}')
        return path
    alt = os.path.join(_PROJECT_DIR, subdir.upper() if dataset_name == 'llvip' else subdir)
    if os.path.isdir(alt):
        return alt
        
    # Kaggle Auto-Discovery (Makes symlinks optional on Kaggle)
    kaggle_input = '/kaggle/input'
    if os.path.exists(kaggle_input):
        for root, dirs, _ in os.walk(kaggle_input):
            if root.count(os.sep) - kaggle_input.count(os.sep) > 6: continue
            
            lower_root = root.lower()
            lower_dirs = [d.lower() for d in dirs]
            
            if dataset_name == 'llvip' and 'llvip' in lower_root:
                if 'infrared' in lower_dirs or 'visible' in lower_dirs:
                    print(f'[DEBUG] Found {dataset_name} at {root}')
                    return root
                    
            if dataset_name == 'camod3fd' and ('m3fd' in lower_root or 'camo' in lower_root):
                if 'train' in lower_dirs or 'images' in lower_dirs:
                    print(f'[DEBUG] Found {dataset_name} at {root}')
                    return root
                    
            if dataset_name == 'forestpersons' and 'forest' in lower_root and 'ir' not in lower_root.split('_')[-1] and 'ir' not in lower_root.split('/')[-1]:
                if 'images' in lower_dirs or 'annotations' in lower_dirs or 'train.json' in os.listdir(root):
                    print(f'[DEBUG] Found {dataset_name} at {root}')
                    return root
                    
            if dataset_name == 'forestpersonsir' and 'forest' in lower_root and ('ir' in lower_root.split('_')[-1] or 'ir' in lower_root.split('/')[-1]):
                if 'images' in lower_dirs or 'annotations' in lower_dirs or 'train.json' in os.listdir(root):
                    print(f'[DEBUG] Found {dataset_name} at {root}')
                    return root
    return path

# Supported dataset configurations
DATASET_CONFIGS = {
    'llvip': {
        'name': 'LLVIP',
        'description': 'Low-Light Vision Infrared-Visible Paired Dataset',
        'infrared_train': 'infrared/train',
        'infrared_test': 'infrared/test',
        'visible_train': 'visible/train',
        'visible_test': 'visible/test',
        'annotations_train': 'Annotations',
        'annotations_test': 'Annotations',
        'annotation_format': 'voc_xml',
        'classes': ['Pedestrian'],
        'image_ext': ['.jpg', '.png', '.jpeg'],
        'native_resolution': (1280, 1024),
        'modality': 'paired',  # synchronized RGB+Thermal
    },
    'forestpersons': {
        'name': 'ForestPersons (RGB)',
        'description': 'Under-canopy SAR person detection — RGB only',
        'huggingface_id': 'etri/ForestPersons',
        'annotation_format': 'coco_json',
        'classes': ['person'],
        'image_ext': ['.jpg', '.png', '.jpeg'],
        'native_resolution': (640, 480),
        'modality': 'rgb_only',  # CMM-RXTO: thermal zeroed
    },
    'forestpersonsir': {
        'name': 'ForestPersons IR (Thermal)',
        'description': 'Under-canopy SAR person detection — Thermal/IR only',
        'huggingface_id': 'etri/ForestPersonsIR',
        'annotation_format': 'coco_json',
        'classes': ['person'],
        'image_ext': ['.jpg', '.png', '.jpeg'],
        'native_resolution': (640, 480),
        'modality': 'thermal_only',  # CMM-ROTX: RGB zeroed
    },
    'camod3fd': {
        'name': 'CAMO M3FD',
        'description': 'Cross-spectral camouflaged pedestrian detection (mask→box)',
        'train_imgs':    'train/Imgs',
        'train_thermal': 'train/Thermal',
        'train_gt':      'train/GT',
        'val_imgs':      'val/Imgs',
        'val_thermal':   'val/Thermal',
        'val_gt':        'val/GT',
        'annotation_format': 'mask_to_box',  # pixel masks → bounding boxes
        'classes': ['People', 'Car', 'Bus', 'Motorcycle', 'Lamp', 'Truck'],
        'image_ext': ['.jpg', '.png', '.jpeg'],
        'native_resolution': (640, 480),
        'modality': 'paired',
    },
    # Legacy datasets (loaders available, not in V2 training pipeline)
    'kaist': {
        'name': 'KAIST Multispectral Pedestrian',
        'description': 'KAIST — urban driving, NOT used in V2 training',
        'image_pattern': 'set{:02d}/V{:03d}/lwir/I{:05d}.png',
        'annotation_format': 'kaist_txt',
        'classes': ['person', 'cyclist', 'people'],
        'native_resolution': (640, 480),
        'modality': 'paired',
    },
    'flirv2': {
        'name': 'FLIR ADAS v2',
        'description': 'FLIR Thermal v2 — automotive, NOT used in V2 training',
        'annotation_format': 'coco_json',
        'classes': ['person', 'bike'],
        'native_resolution': (640, 512),
        'modality': 'paired',
    },
}

# Active dataset (change this based on what you upload to Kaggle)
ACTIVE_DATASET = 'llvip'

# ============================================================================
# MODEL INPUT CONFIGURATION
# ============================================================================
INPUT_WIDTH = 160
INPUT_HEIGHT = 128
INPUT_SIZE = (INPUT_HEIGHT, INPUT_WIDTH)  # (128, 160)

INPUT_CHANNELS = 4   # 3 (RGB) + 1 (Thermal)

# ============================================================================
# TARGET CLASSES
# ============================================================================
CLASS_MAP = {
    'background': 0,
    'person_visible': 1,
    'person_camouflaged': 2,
    'vehicle_boat': 3
}
NUM_CLASSES = len(CLASS_MAP)  # 4

# Anchor configuration
NUM_ANCHORS = 3

# Default anchor aspect ratios (h/w). These are the all-dataset K-Means anchors
# from the successful LLVIP + ForestPersons + ForestPersonsIR Kaggle run.
DEFAULT_ANCHOR_RATIOS = [1.7217747953704292, 3.340500600298943, 5.433984452466083]

# Default anchor sizes (relative to image, sqrt of area).
DEFAULT_ANCHOR_SIZES = [0.0717509437717046, 0.13346257948562704, 0.4044514489729058]

# Grid sizes
SMALL_GRID_H = INPUT_HEIGHT // 8    # 16
SMALL_GRID_W = INPUT_WIDTH // 8     # 20
LARGE_GRID_H = INPUT_HEIGHT // 16   # 8
LARGE_GRID_W = INPUT_WIDTH // 16    # 10

# Detection thresholds (split for different use cases)
CONFIDENCE_THRESHOLD = 0.20       # Default for inference; artifact filters reduce vehicle/glare false positives
EVAL_CONFIDENCE_THRESHOLD = 0.10  # For mAP evaluation (high recall)
NMS_IOU_THRESHOLD = 0.35          # V2: tighter NMS for adjacent person detection
MAX_DETECTIONS = 10

# Log clamp range for bbox encoding/decoding
LOG_CLAMP_MIN = -4.5
LOG_CLAMP_MAX = 4.5

# Objectness metric threshold for recall computation
OBJ_METRIC_THRESHOLD = 0.25

# Post-decode filters
MIN_BOX_WIDTH_NORM = 0.03
MIN_BOX_HEIGHT_NORM = 0.05
MIN_BOX_AREA_NORM = 0.002
CORNER_FILTER_X = 0.12
CORNER_FILTER_Y = 0.10
CORNER_MAX_WIDTH = 0.10

# ============================================================================
# V1 MODEL ARCHITECTURE (kept for backward compatibility)
# ============================================================================
RGB_STEM_CHANNELS = 16
THERMAL_STEM_CHANNELS = 16
STEM_CHANNELS = RGB_STEM_CHANNELS + THERMAL_STEM_CHANNELS  # 32

SCALE1_CHANNELS = 32
SCALE2_CHANNELS = 48
SCALE3_CHANNELS = 96
FPN_CHANNELS = 64
CLASSIFIER_HIDDEN_DIM = 64
EXPAND_RATIO = 3

# ============================================================================
# V2 MODEL ARCHITECTURE (Dual-Branch, Accuracy-First)
# ============================================================================
# Per-branch channel widths (each branch processes one modality independently)
V2_STEM_CHANNELS = 16           # Per-branch stem output
V2_SCALE1_CHANNELS = 24         # Per-branch Scale 1 (Ghost Bottleneck)
V2_SCALE2_CHANNELS = 32         # Per-branch Scale 2 → EnergyGate
V2_SCALE3_CHANNELS = 48         # Per-branch Scale 3 → BiFusion Neck
V2_BIFUSION_CHANNELS = 48       # BiFusion Neck output (feeds heads + classifier)
V2_CLASSIFIER_HIDDEN_DIM = 64   # Classifier FC hidden dim

# Expand ratio sweet spot: 4 balances accuracy vs size
# (3 = lean/V1, 6 = heavy/maximum capacity)
V2_EXPAND_RATIO = 4

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
EPOCHS = 50
PATIENCE = 10
NUM_WORKERS = 0

# Loss weights
BBOX_WEIGHT = 3.0
OBJ_WEIGHT = 7.0
CLASS_WEIGHT = 0.5

# V2 auxiliary loss weights
CONTRAST_WEIGHT = 0.1           # TFDet-style contrast loss (Phase 2+)
GATE_REG_WEIGHT = 0.05          # EnergyGate entropy regularizer (Phase 3)

# Learning rate schedule
LR_SCHEDULER = 'cosine_warm_restarts'
LR_T0 = 10
LR_T_MULT = 2
LR_MIN = 1e-6

# Data split
VAL_RATIO = 0.2
RANDOM_SEED = 42

# Debugging / Testing
DEBUG_MODE = False  # Set to True to run only 1 batch per epoch and 1 epoch total for testing

# ============================================================================
# V2 PHASE TRAINING CONFIGURATION
# ============================================================================
TRAINING_PHASES = {
    1: {
        'name': 'Human Shape Foundation (LLVIP)',
        'datasets': {'llvip': 1.0},
        'epochs': 50,
        'lr': 1e-3,
        'cmm_enabled': False,
        'cmm_alpha': 0.0,
        'contrast_loss': False,
        'gate_regularizer': False,
        'freeze_layers': [],
        'batch_size': 32,
    },
    2: {
        'name': 'Jungle Domain Transfer (ForestPersons + LLVIP)',
        'datasets': {
            'forestpersons': 0.40,     # CMM-RXTO (thermal zeroed)
            'forestpersonsir': 0.40,   # CMM-ROTX (RGB zeroed)
            'llvip': 0.20,            # Replay (prevents forgetting)
        },
        'epochs': 60,
        'lr': 2e-4,
        'cmm_enabled': True,
        'cmm_alpha_start': 0.3,
        'cmm_alpha_end': 0.5,
        'contrast_loss': True,
        'gate_regularizer': False,
        'freeze_layers': [],
        'batch_size': 32,
    },
    3: {
        'name': 'Full Mix Polish',
        'datasets': {
            'llvip': 0.34,
            'forestpersons': 0.33,
            'forestpersonsir': 0.33,
        },
        'epochs': 25,
        'lr': 1e-5,
        'cmm_enabled': True,
        'cmm_alpha': 0.5,
        'contrast_loss': True,
        'gate_regularizer': True,
        'freeze_layers': [],
        'batch_size': 32,
    },
}

# ============================================================================
# DEPLOYMENT CONSTRAINTS (ESP32-S3 with 8MB PSRAM) — optimize later
# ============================================================================
ESP32_S3 = {
    'sram_kb': 512,
    'psram_mb': 8,
    'flash_mb': 8,
    'clock_mhz': 240,
    'max_model_flash_kb': 4000,
    'max_arena_sram_kb': 4000,
    'target_fps': 10,
    'target_model_fp16_kb': 2048,
    'target_model_int8_kb': 2048,
}

# ============================================================================
# ALERT CONFIGURATION
# ============================================================================
ALERT_CONFIG = {
    'include_confidence': True,
    'include_bbox': True,
    'include_thermal_score': True,
    'include_timestamp': True,
    'include_intrusion_count': True,
    'compact_format': True,
}

# ============================================================================
# FILE PATHS
# ============================================================================
MODEL_SAVE_DIR = 'checkpoints'
BEST_MODEL_V1_PATH = os.path.join(MODEL_SAVE_DIR, 'best_microghost_thermal_v1.pth')
BEST_MODEL_V2_PATH = os.path.join(MODEL_SAVE_DIR, 'best_microghost_thermal_v3.pth')
BEST_MODEL_PATH = BEST_MODEL_V2_PATH
LAST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'last_microghost_thermal.pth')
EXPORT_DIR = 'exports'
ONNX_PATH = os.path.join(EXPORT_DIR, 'microghost_thermal.onnx')
TFLITE_PATH = os.path.join(EXPORT_DIR, 'microghost_thermal.tflite')
TFLITE_FP16_PATH = os.path.join(EXPORT_DIR, 'microghost_thermal_fp16.tflite')
LOG_DIR = 'logs'


def print_config():
    """Print current configuration summary."""
    print("=" * 65)
    print("  MicroGhost-Thermal Configuration (V2)")
    print("=" * 65)
    print(f"  Device:           {DEVICE}")
    print(f"  Input:            {INPUT_SIZE}×{INPUT_CHANNELS}")
    print(f"  Classes:          {NUM_CLASSES} → {list(CLASS_MAP.keys())}")
    print(f"  Anchors/cell:     {NUM_ANCHORS}")
    print(f"  Grids:            {SMALL_GRID_H}×{SMALL_GRID_W} (small), "
          f"{LARGE_GRID_H}×{LARGE_GRID_W} (large)")
    print(f"  V2 Branch:        {V2_STEM_CHANNELS}→{V2_SCALE1_CHANNELS}→"
          f"{V2_SCALE2_CHANNELS}→{V2_SCALE3_CHANNELS}")
    print(f"  V2 BiFusion:      {V2_BIFUSION_CHANNELS}")
    print(f"  V2 Expand Ratio:  {V2_EXPAND_RATIO}")
    print(f"  Batch size:       {BATCH_SIZE}")
    print(f"  Learning rate:    {LEARNING_RATE}")
    print(f"  Epochs:           {EPOCHS} (patience={PATIENCE})")
    print(f"  Active dataset:   {ACTIVE_DATASET}")
    print(f"  Log clamp:        [{LOG_CLAMP_MIN}, {LOG_CLAMP_MAX}]")
    print(f"  NMS IoU:          {NMS_IOU_THRESHOLD}")
    print(f"  Conf (inference): {CONFIDENCE_THRESHOLD}")
    print(f"  Conf (eval):      {EVAL_CONFIDENCE_THRESHOLD}")
    print("=" * 65)


if __name__ == '__main__':
    print_config()
