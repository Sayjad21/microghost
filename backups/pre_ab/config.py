"""
MicroGhost-Thermal: Configuration Module
==========================================
Central configuration for the ESP32-S3 thermal intrusion detection system.

Target Hardware: ESP32-S3 (512KB SRAM, 8MB Flash, 240MHz dual-core)
Sensor: FLIR Lepton 3.5 (160×120) or FLIR Lepton 2.5 (80×60) via SPI
Model Framework: TensorFlow Lite Micro (deployed), PyTorch (training)
"""

import os
import torch

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# Default local data directory (override with MICROGHOST_DATA_ROOT env var)
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.environ.get('MICROGHOST_DATA_ROOT', os.path.join(_PROJECT_DIR, 'data'))

# Per-dataset subdirectory names under DATASET_ROOT
DATASET_SUBDIRS = {
    'llvip': 'llvip',
    'kaist': 'kaist-multispectral',
    'flirv2': 'FLIR_ADAS_v2',
}


def get_dataset_path(dataset_name):
    """Resolve dataset root path (env override or default under DATASET_ROOT)."""
    env_key = f'MICROGHOST_{dataset_name.upper()}_PATH'
    if env_key in os.environ:
        return os.environ[env_key]
    subdir = DATASET_SUBDIRS.get(dataset_name, dataset_name)
    path = os.path.join(DATASET_ROOT, subdir)
    if os.path.isdir(path):
        return path
    # Fallback: dataset folder at project root (e.g. ./LLVIP)
    alt = os.path.join(_PROJECT_DIR, subdir.upper() if dataset_name == 'llvip' else subdir)
    if os.path.isdir(alt):
        return alt
    return path

# Supported dataset configurations
DATASET_CONFIGS = {
    'llvip': {
        'name': 'LLVIP',
        'description': 'Low-Light Vision Infrared-Visible Paired Dataset',
        'infrared_train': 'infrared/train',
        'infrared_test': 'infrared/test',
        'visible_train': 'visible/train',   # Not used, but available
        'visible_test': 'visible/test',      # Not used, but available
        'annotations_train': 'Annotations',  # VOC XML format
        'annotations_test': 'Annotations',
        'annotation_format': 'voc_xml',
        'classes': ['Pedestrian'],
        'image_ext': ['.jpg', '.png', '.jpeg'],
        'native_resolution': (1280, 1024),   # W x H
    },
    'kaist': {
        'name': 'KAIST Multispectral Pedestrian',
        'description': 'KAIST Multispectral Pedestrian Detection Benchmark',
        'image_pattern': 'set{:02d}/V{:03d}/lwir/I{:05d}.png',
        'annotation_pattern': 'set{:02d}/V{:03d}/annotations/I{:05d}.txt',
        'annotation_format': 'kaist_txt',
        'classes': ['person', 'cyclist', 'people'],  # Mapped to person_visible
        'native_resolution': (640, 480),     # W x H
    },
    'flirv2': {
        'name': 'FLIR ADAS v2',
        'description': 'FLIR Thermal Dataset v2',
        'train_dir': 'images_thermal_train',
        'val_dir': 'images_thermal_val',
        'rgb_train_dir': 'images_rgb_train',
        'rgb_val_dir': 'images_rgb_val',
        'annotations_train': 'annotations/thermal_annotations_train.json',
        'annotations_val': 'annotations/thermal_annotations_val.json',
        'annotation_format': 'coco_json',
        'classes': ['person', 'bike'],  # Mapped to person_visible
        'native_resolution': (640, 512),
    },
}

# Active dataset (change this based on what you upload to Kaggle)
ACTIVE_DATASET = 'llvip'

# ============================================================================
# MODEL INPUT CONFIGURATION
# ============================================================================
# Input size for ESP32-S3 deployment
# FLIR Lepton: 160x120. We pad 4 pixels top/bottom to get exactly 160x128
# which is perfectly divisible by 32 for the 4 FPN downsampling stages.
INPUT_WIDTH = 160
INPUT_HEIGHT = 128
INPUT_SIZE = (INPUT_HEIGHT, INPUT_WIDTH) # (128, 160)

INPUT_CHANNELS = 4   # 3 (RGB) + 1 (Thermal)

# ============================================================================
# TARGET CLASSES (BGB MISSION CRITICAL)
# ============================================================================
CLASS_MAP = {
    'background': 0,
    'person_visible': 1,
    'person_camouflaged': 2,
    'vehicle_boat': 3
}
NUM_CLASSES = len(CLASS_MAP)  # 4: background + 3 intrusion types

# Anchor configuration — Phase 2: 3 anchors (rider / standing / tall-distant)
NUM_ANCHORS = 3

# Default anchor aspect ratios (h/w)
# - Bike/scooter rider: wider (~1.6 h/w)
# - Standing pedestrian: ~2.5 h/w
# - Tall / distant person: ~3.5 h/w
DEFAULT_ANCHOR_RATIOS = [1.6, 2.5, 3.5]

# Default anchor sizes (relative to image, sqrt of area)
DEFAULT_ANCHOR_SIZES = [0.095, 0.127, 0.145]  # small / medium / large

# Grid sizes derived from input height/width
SMALL_GRID_H = INPUT_HEIGHT // 8    # 16
SMALL_GRID_W = INPUT_WIDTH // 8     # 20
LARGE_GRID_H = INPUT_HEIGHT // 16   # 8
LARGE_GRID_W = INPUT_WIDTH // 16    # 10

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.12     # Lower for small-model objectness scores (previous value was 0.15)
OBJ_METRIC_THRESHOLD = 0.25     # Threshold when computing objectness recall
NMS_IOU_THRESHOLD = 0.45        # Standard NMS (0.35 was too aggressive on crowds)
MAX_DETECTIONS = 15             # Allow more boxes in dense groups

# Post-decode filters (remove corner artifacts and tiny false positives)
MIN_BOX_WIDTH_NORM = 0.03       # Min box width relative to image
MIN_BOX_HEIGHT_NORM = 0.05      # Min box height relative to image
MIN_BOX_AREA_NORM = 0.002       # Min box area (width * height)
CORNER_FILTER_X = 0.12          # Reject thin boxes in top-left corner band
CORNER_FILTER_Y = 0.10
CORNER_MAX_WIDTH = 0.10         # Corner band only applies below this width
# Wider band for medium-height foliage FPs (still passes thin-box rule above)
CORNER_FILTER_X_WIDE = 0.13
CORNER_FILTER_Y_WIDE = 0.30
CORNER_WIDE_MAX_WIDTH = 0.10
CORNER_LOW_CONF_MAX = 0.40      # Wide band only below this confidence
# Small-head grid cells that fire on trees/foliage in the top-left
SMALL_GRID_BLACKLIST_GX_MAX = 2
SMALL_GRID_BLACKLIST_GY_MAX = 1

# ============================================================================
# MODEL ARCHITECTURE CONFIGURATION
# ============================================================================
# Dual-Branch Stem
RGB_STEM_CHANNELS = 16     # RGB input (3) → 16
THERMAL_STEM_CHANNELS = 16 # Thermal input (1) → 16
STEM_CHANNELS = RGB_STEM_CHANNELS + THERMAL_STEM_CHANNELS # Fused → 32

# Backbone Stages
SCALE1_CHANNELS = 32       # Backbone stage 1
SCALE2_CHANNELS = 48       # Backbone stage 2
SCALE3_CHANNELS = 96       # Backbone stage 3
FPN_CHANNELS = 64          # FPN output
CLASSIFIER_HIDDEN_DIM = 64 # Classifier FC hidden

# Expansion ratio for InvertedResidual blocks
EXPAND_RATIO = 3           # Reduced to 3

# ============================================================================
# TRAINING CONFIGURATION (tuned for local CPU — Acer Aspire 3, 7GB RAM)
# ============================================================================
BATCH_SIZE = 8             # Low RAM; increase to 16 if training is stable
LEARNING_RATE = 1e-3       # Initial learning rate
WEIGHT_DECAY = 5e-4        # Slightly stronger L2 for val generalization
EPOCHS = 50                # Enough for demo; early stopping may finish sooner
PATIENCE = 10              # Early stopping patience
NUM_WORKERS = 0            # 0 avoids RAM spikes on 7GB systems

# Loss weights — emphasize bbox regression (main generalization bottleneck)
BBOX_WEIGHT = 3.0          # Bounding box regression
OBJ_WEIGHT = 5.0           # Objectness (was 10; grid is mostly negatives)
CLASS_WEIGHT = 0.5         # Image-level class (trivial on LLVIP)

# Learning rate schedule
LR_SCHEDULER = 'cosine_warm_restarts'
LR_T0 = 10                # Cosine restart period
LR_T_MULT = 2             # Period multiplier
LR_MIN = 1e-6             # Minimum learning rate

# Data split
VAL_RATIO = 0.2            # 20% validation split
RANDOM_SEED = 42           # For reproducible splits

# ============================================================================
# DEPLOYMENT CONSTRAINTS (ESP32-S3 with 8MB PSRAM)
# ============================================================================
ESP32_S3 = {
    'sram_kb': 512,
    'psram_mb': 8,                  # We have 8MB of PSRAM available
    'flash_mb': 8,
    'clock_mhz': 240,
    'max_model_flash_kb': 4000,     # 4 MB flash budget for FP16 weights
    'max_arena_sram_kb': 4000,      # Arena can spill into PSRAM easily (4MB limit)
    'target_fps': 10,
    'target_model_fp16_kb': 2048,   # 2 MB hard limit for ESP32 flash
    'target_model_int8_kb': 2048,   # INT8 weights budget (deploy target)
}

# ============================================================================
# ALERT CONFIGURATION
# ============================================================================
ALERT_CONFIG = {
    'include_confidence': True,
    'include_bbox': True,
    'include_temperature': True,
    'include_timestamp': True,
    'include_intrusion_count': True,
    'compact_format': True,         # Use compact binary for low-bandwidth TX
}

# ============================================================================
# FILE PATHS
# ============================================================================
MODEL_SAVE_DIR = 'checkpoints'
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_microghost_thermal.pth')
LAST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'last_microghost_thermal.pth')
EXPORT_DIR = 'exports'
ONNX_PATH = os.path.join(EXPORT_DIR, 'microghost_thermal.onnx')
TFLITE_PATH = os.path.join(EXPORT_DIR, 'microghost_thermal.tflite')
TFLITE_FP16_PATH = os.path.join(EXPORT_DIR, 'microghost_thermal_fp16.tflite')
LOG_DIR = 'logs'


def print_config():
    """Print current configuration summary."""
    print("=" * 65)
    print("  MicroGhost-Thermal Configuration")
    print("=" * 65)
    print(f"  Device:           {DEVICE}")
    print(f"  Input:            {INPUT_SIZE}×{INPUT_SIZE}×{INPUT_CHANNELS} (thermal)")
    print(f"  Classes:          {NUM_CLASSES} → {list(CLASS_MAP.keys())}")
    print(f"  Anchors/cell:     {NUM_ANCHORS}")
    print(f"  Grids:            {SMALL_GRID_H}×{SMALL_GRID_W} (small), "
          f"{LARGE_GRID_H}×{LARGE_GRID_W} (large)")
    print(f"  Backbone:         {STEM_CHANNELS}→{SCALE1_CHANNELS}→"
          f"{SCALE2_CHANNELS}→{SCALE3_CHANNELS}")
    print(f"  FPN channels:     {FPN_CHANNELS}")
    print(f"  Batch size:       {BATCH_SIZE}")
    print(f"  Learning rate:    {LEARNING_RATE}")
    print(f"  Epochs:           {EPOCHS} (patience={PATIENCE})")
    print(f"  Active dataset:   {ACTIVE_DATASET}")
    print(f"  Target:           ESP32-S3 ({ESP32_S3['sram_kb']}KB SRAM, "
          f"{ESP32_S3['flash_mb']}MB Flash)")
    print(f"  Target FP16:      <{ESP32_S3['target_model_fp16_kb']}KB")
    print(f"  Target FPS:       ≥{ESP32_S3['target_fps']}")
    print("=" * 65)


if __name__ == '__main__':
    print_config()
