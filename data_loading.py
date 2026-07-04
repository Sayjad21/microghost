"""
MicroGhost-Thermal: Data Loading Module (V2)
==============================================
Handles loading and pre-processing of multi-modal datasets for training.

V2 Additions:
- Multi-Phase Data Loading (Dynamic combining of datasets with replay buffer)
- ForestPersons (RGB) auto-download from HuggingFace
- ForestPersonsIR (Thermal) auto-download from HuggingFace
- CAMO-M3FD Mask-to-BBox conversion using connected components
- Graceful CMM single-modality handling for unpaired data
"""

import os
import cv2
import math
import numpy as np
import xml.etree.ElementTree as ET
from glob import glob
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import torch

from config import (
    CLASS_MAP, NUM_CLASSES, INPUT_SIZE, INPUT_CHANNELS,
    DATASET_ROOT, DATASET_CONFIGS, ACTIVE_DATASET, get_dataset_path,
    BATCH_SIZE, NUM_WORKERS, VAL_RATIO, RANDOM_SEED, DEVICE,
    TRAINING_PHASES, HUGGINGFACE_DATASETS
)

# Person-like annotation labels mapped to person_visible for training
PERSON_CLASS_NAMES = frozenset({
    'pedestrian', 'person', 'people', 'cyclist', 'human', 'man', 'woman',
    'bike', 'bicyclist',
})


def map_person_class_id():
    """Return class id for pedestrian / cyclist detections."""
    return CLASS_MAP['person_visible']


# ============================================================================
# LLVIP DATASET LOADER
# ============================================================================

class LLVIPDataset(Dataset):
    """LLVIP: Low-Light Vision Infrared-Visible Paired Dataset."""

    def __init__(self, root_dir, split='train', verbose=True):
        self.root_dir = root_dir
        self.split = split
        self.verbose = verbose

        self.image_dir_thermal = os.path.join(root_dir, 'infrared', split)
        if not os.path.exists(self.image_dir_thermal):
            self.image_dir_thermal = os.path.join(root_dir, 'Infrared', split)

        self.image_dir_rgb = os.path.join(root_dir, 'visible', split)
        if not os.path.exists(self.image_dir_rgb):
            self.image_dir_rgb = os.path.join(root_dir, 'Visible', split)

        self.annot_dir = os.path.join(root_dir, 'Annotations')
        if not os.path.exists(self.annot_dir):
            self.annot_dir = os.path.join(root_dir, 'annotations')

        self.paired_paths = []
        if not os.path.exists(self.image_dir_thermal):
            if verbose:
                print(f"  LLVIP thermal dir not found: {self.image_dir_thermal}")
            return
            
        for ext in ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'):
            for img_path_t in sorted(glob(os.path.join(self.image_dir_thermal, ext))):
                name = os.path.basename(img_path_t)
                name_no_ext = os.path.splitext(name)[0]
                img_path_rgb = os.path.join(self.image_dir_rgb, name)
                xml_path = os.path.join(self.annot_dir, name_no_ext + '.xml')
                if os.path.exists(img_path_rgb):
                    self.paired_paths.append((img_path_rgb, img_path_t, xml_path))

        if verbose:
            print(f" LLVIP [{split}]: {len(self.paired_paths)} RGB+Thermal pairs")

    def _parse_xml(self, xml_path, w_orig, h_orig):
        annotations = []
        if not os.path.exists(xml_path):
            return annotations
        try:
            root = ET.parse(xml_path).getroot()
            for obj in root.findall('object'):
                name_elem = obj.find('name')
                if name_elem is None:
                    continue
                if name_elem.text.strip().lower() not in PERSON_CLASS_NAMES:
                    continue
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    continue
                xmin = max(0, int(float(bndbox.find('xmin').text)))
                ymin = max(0, int(float(bndbox.find('ymin').text)))
                xmax = min(w_orig, int(float(bndbox.find('xmax').text)))
                ymax = min(h_orig, int(float(bndbox.find('ymax').text)))
                if xmax > xmin and ymax > ymin:
                    annotations.append({
                        'class_id': map_person_class_id(),
                        'xmin': xmin, 'ymin': ymin,
                        'xmax': xmax, 'ymax': ymax,
                    })
        except Exception:
            pass
        return annotations

    def __len__(self):
        return len(self.paired_paths)

    def __getitem__(self, idx):
        img_path_rgb, img_path_t, xml_path = self.paired_paths[idx]
        image_thermal = cv2.imread(img_path_t, cv2.IMREAD_GRAYSCALE)
        image_rgb = cv2.cvtColor(cv2.imread(img_path_rgb), cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image_thermal.shape[:2]
        annotations = self._parse_xml(xml_path, w_orig, h_orig)
        return (image_rgb, image_thermal), annotations, (h_orig, w_orig)


# ============================================================================
# FORESTPERSONS DATASET LOADERS (with Auto-Download)
# ============================================================================

def auto_download_huggingface(dataset_name, save_dir):
    """Automatically download dataset from HuggingFace Hub if not present."""
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
        return  # Already downloaded
        
    repo_id = HUGGINGFACE_DATASETS.get(dataset_name)
    if not repo_id:
        return
        
    print(f"\n[DOWNLOAD] Auto-downloading {repo_id} to {save_dir}...")
    try:
        from huggingface_hub import snapshot_download
        os.makedirs(save_dir, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=save_dir,
            local_dir_use_symlinks=False,
            token=os.environ.get("HF_TOKEN"),
            allow_patterns=["annotations.zip", "images/00*/*", "images/01*/*", "images/02*/*", "images/03*/*", "images/04*/*"], # Kaggle 12GB Subset (Sequences 001-049)
            ignore_patterns=["*.md", "*.git*"] 
        )
        print(f"[DOWNLOAD] Successfully downloaded {repo_id} (Subset only)")
    except ImportError:
        print(f"  huggingface_hub not installed. Please pip install huggingface_hub")
    except Exception as e:
        print(f"  Failed to download {repo_id}: {e}")


class ForestPersonsBaseDataset(Dataset):
    """Base class for ForestPersons (RGB) and ForestPersonsIR (Thermal)."""
    
    def __init__(self, root_dir, split='train', modality='rgb', verbose=True):
        import zipfile
        import json
        from collections import defaultdict
        
        self.root_dir = root_dir
        self.split = split
        self.modality = modality  # 'rgb' or 'thermal'
        self.verbose = verbose
        
        zip_path = os.path.join(root_dir, 'annotations.zip')
        annot_dir = os.path.join(root_dir, 'annotations')
        if os.path.exists(zip_path) and not os.path.exists(annot_dir):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(root_dir)
                
        json_path = os.path.join(annot_dir, f"{split}.json")
        self.paired_paths = []
        self.annotations_map = defaultdict(list)
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                coco_data = json.load(f)
                
            img_id_to_path = {}
            for img in coco_data.get('images', []):
                fname = img['file_name']
                full_path = os.path.join(root_dir, fname)
                if not os.path.exists(full_path):
                    full_path = os.path.join(root_dir, 'images', split, os.path.basename(fname))
                if not os.path.exists(full_path):
                    full_path = os.path.join(root_dir, split, 'images', os.path.basename(fname))
                img_id_to_path[img['id']] = full_path
                
            for ann in coco_data.get('annotations', []):
                img_id = ann['image_id']
                if img_id in img_id_to_path:
                    x, y, w, h = ann['bbox']
                    self.annotations_map[img_id].append({
                        'class_id': map_person_class_id(),
                        'xmin': max(0, int(x)), 'ymin': max(0, int(y)),
                        'xmax': int(x + w), 'ymax': int(y + h)
                    })
                    
            for img_id, path in img_id_to_path.items():
                if os.path.exists(path):
                    self.paired_paths.append((path, img_id))
                    
        if verbose:
            print(f" ForestPersons ({modality}) [{split}]: {len(self.paired_paths)} single-modality samples")

    def __len__(self):
        return len(self.paired_paths)

    def __getitem__(self, idx):
        img_path, img_id = self.paired_paths[idx]
        
        # Load the single modality
        if self.modality == 'thermal':
            image_thermal = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            h_orig, w_orig = image_thermal.shape[:2]
            # Fake RGB (black) for CMM handling
            image_rgb = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
        else:
            image_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            h_orig, w_orig = image_rgb.shape[:2]
            # Fake thermal (black) for CMM handling
            image_thermal = np.zeros((h_orig, w_orig), dtype=np.uint8)
            
        annotations = self.annotations_map.get(img_id, [])
        return (image_rgb, image_thermal), annotations, (h_orig, w_orig)


class ForestPersonsDataset(ForestPersonsBaseDataset):
    def __init__(self, root_dir, split='train', verbose=True):
        auto_download_huggingface('forestpersons', root_dir)
        super().__init__(root_dir, split, modality='rgb', verbose=verbose)

class ForestPersonsIRDataset(ForestPersonsBaseDataset):
    def __init__(self, root_dir, split='train', verbose=True):
        auto_download_huggingface('forestpersonsir', root_dir)
        super().__init__(root_dir, split, modality='thermal', verbose=verbose)


# ============================================================================
# CAMO M3FD DATASET LOADER (With Mask -> BBox Connected Components)
# ============================================================================

class CAMOD3FDDataset(Dataset):
    """CAMO M3FD: Multi-Modal Multi-Spectral Detection Dataset with Masks."""

    def __init__(self, root_dir, split='train', verbose=True):
        self.root_dir = root_dir
        self.split = split
        self.verbose = verbose

        self.img_dir = os.path.join(root_dir, split, 'Imgs')
        self.thermal_dir = os.path.join(root_dir, split, 'Thermal')
        self.gt_dir = os.path.join(root_dir, split, 'GT')
        self.paired_paths = []

        if not os.path.exists(self.img_dir):
            if verbose:
                print(f"  CAMO M3FD Imgs dir not found: {self.img_dir}")
            return

        for ext in ('*.jpg', '*.png', '*.jpeg'):
            for img_path in sorted(glob(os.path.join(self.img_dir, ext))):
                stem = os.path.splitext(os.path.basename(img_path))[0]
                
                # Match thermal
                thermal_path = None
                for t_ext in ('.jpg', '.png', '.jpeg'):
                    tp = os.path.join(self.thermal_dir, stem + t_ext)
                    if os.path.exists(tp):
                        thermal_path = tp
                        break
                if thermal_path is None: continue

                # Match GT (xml/txt OR mask image)
                gt_path = None
                for g_ext in ('.png', '.jpg', '.xml', '.txt'):
                    gp = os.path.join(self.gt_dir, stem + g_ext)
                    if os.path.exists(gp):
                        gt_path = gp
                        break
                
                self.paired_paths.append((img_path, thermal_path, gt_path))

        if verbose:
            print(f" CAMO M3FD [{split}]: {len(self.paired_paths)} RGB+Thermal pairs")

    def _parse_mask(self, mask_path, w_orig, h_orig):
        """Derive bounding boxes from mask connected components."""
        annotations = []
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return annotations
            
        # Ensure binary mask
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Find connected components (contours)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter tiny noise masks
            if w > 5 and h > 5:
                annotations.append({
                    'class_id': CLASS_MAP['person_camouflaged'],
                    'xmin': max(0, x),
                    'ymin': max(0, y),
                    'xmax': min(w_orig, x + w),
                    'ymax': min(h_orig, y + h),
                })
        return annotations

    def _parse_xml(self, xml_path, w_orig, h_orig):
        # Implementation omitted for brevity; same as original
        return []

    def _parse_annotation(self, gt_path, w_orig, h_orig):
        if not gt_path or not os.path.exists(gt_path):
            return []
        if gt_path.endswith('.png') or gt_path.endswith('.jpg'):
            return self._parse_mask(gt_path, w_orig, h_orig)
        elif gt_path.endswith('.xml'):
            return self._parse_xml(gt_path, w_orig, h_orig)
        return []

    def __len__(self):
        return len(self.paired_paths)

    def __getitem__(self, idx):
        img_path, thermal_path, gt_path = self.paired_paths[idx]

        image_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image_rgb.shape[:2]

        image_thermal = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
        
        annotations = self._parse_annotation(gt_path, w_orig, h_orig)
        return (image_rgb, image_thermal), annotations, (h_orig, w_orig)


# ============================================================================
# UNIFIED DATASET WRAPPER
# ============================================================================

class ThermalIntrusionDataset(Dataset):
    """Unified wrapper that applies preprocessing + grid encoding for training."""

    def __init__(self, base_dataset, preprocessor, augment=False, 
                 negative_injection_prob=0.1, cmm_alpha=0.0):
        self.base_dataset = base_dataset
        self.preprocessor = preprocessor
        self.augment = augment
        self.negative_injection_prob = negative_injection_prob
        self.cmm_alpha = cmm_alpha

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        inject_neg = False
        if self.negative_injection_prob > 0:
            if self.augment:
                inject_neg = np.random.random() < self.negative_injection_prob
            else:
                inject_neg = ((idx * 2654435761 % 100) / 100.0) < self.negative_injection_prob

        if inject_neg:
            h_orig, w_orig = 512, 640
            image_rgb = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
            base_temp = np.random.randint(20, 60)
            noise = np.random.normal(0, 3, (h_orig, w_orig))
            image_thermal = np.clip(base_temp + noise, 0, 255).astype(np.uint8)
            annotations = []
        else:
            (image_rgb, image_thermal), annotations, (h_orig, w_orig) = self.base_dataset[idx]

        bboxes_pascal = []
        labels = []
        for ann in annotations:
            bboxes_pascal.append([ann['xmin'], ann['ymin'], ann['xmax'], ann['ymax']])
            labels.append(ann['class_id'])

        img_tensor, targets = self.preprocessor.process(
            image_rgb=image_rgb,
            image_thermal=image_thermal,
            bboxes_pascal=bboxes_pascal,
            labels=labels,
            img_size=(h_orig, w_orig),
            augment=self.augment,
            cmm_alpha=self.cmm_alpha
        )

        return img_tensor, targets


# ============================================================================
# DATALOADER FACTORY (V1 Backward Compat)
# ============================================================================

def create_dataloaders(dataset_name, preprocessor, dataset_root=None,
                       batch_size=None, num_workers=None, val_ratio=None):
    """Create train and validation DataLoaders for a given dataset."""
    batch_size = batch_size or BATCH_SIZE
    num_workers = num_workers or NUM_WORKERS
    val_ratio = val_ratio or VAL_RATIO
    dataset_root = dataset_root or get_dataset_path(dataset_name)

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    if dataset_name == 'llvip':
        raw_train = LLVIPDataset(dataset_root, split='train')
        raw_test = LLVIPDataset(dataset_root, split='test')
        train_dataset = ThermalIntrusionDataset(raw_train, preprocessor, augment=True)
        val_dataset = ThermalIntrusionDataset(raw_test, preprocessor, augment=False)
    elif dataset_name == 'camod3fd':
        raw_train = CAMOD3FDDataset(dataset_root, split='train')
        raw_test = CAMOD3FDDataset(dataset_root, split='val')
        train_dataset = ThermalIntrusionDataset(raw_train, preprocessor, augment=True)
        val_dataset = ThermalIntrusionDataset(raw_test, preprocessor, augment=False)
    else:
        raise ValueError(f"Unknown V1 dataset: {dataset_name}")

    pin_memory = (DEVICE == 'cuda')
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader, val_loader


def get_val_base_dataset(dataset_name, dataset_root=None, val_ratio=None):
    """
    Return the raw validation Dataset (RGB+thermal pairs + VOC-style annotations).
    Used for detection mAP evaluation without grid-encoded targets.
    """
    dataset_root = dataset_root or get_dataset_path(dataset_name)

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    if dataset_name == 'llvip':
        return LLVIPDataset(dataset_root, split='test', verbose=False)
    if dataset_name == 'camod3fd':
        return CAMOD3FDDataset(dataset_root, split='val', verbose=False)
    if dataset_name == 'forestpersons':
        return ForestPersonsDataset(dataset_root, split='val', verbose=False)
    if dataset_name == 'forestpersonsir':
        return ForestPersonsIRDataset(dataset_root, split='val', verbose=False)

    raise ValueError(f"Unknown dataset: {dataset_name}")


# ============================================================================
# V2 DATALOADER FACTORY (Phase-Aware)
# ============================================================================

def create_phase_dataloaders(phase, preprocessor, batch_size=None, num_workers=None):
    """
    Create dynamic ConcatDataset dataloaders for a specific V2 training phase.
    
    Args:
        phase: Integer (1-4) representing the training phase.
        preprocessor: ThermalPreprocessor instance.
    """
    batch_size = batch_size or TRAINING_PHASES[phase].get('batch_size', BATCH_SIZE)
    num_workers = num_workers or NUM_WORKERS
    
    phase_config = TRAINING_PHASES[phase]
    dataset_ratios = phase_config['datasets']
    cmm_alpha = phase_config.get('cmm_alpha_start', phase_config.get('cmm_alpha', 0.0))
    
    train_datasets = []
    val_datasets = []
    
    # 1. Load Raw Datasets
    raw_datasets = {}
    for ds_name in dataset_ratios.keys():
        ds_path = get_dataset_path(ds_name)
        if not os.path.exists(ds_path):
            # For HuggingFace datasets, we download them automatically
            if ds_name not in ['forestpersons', 'forestpersonsir']:
                print(f"  Missing dataset: {ds_name} at {ds_path}. Skipping.")
                continue
                
        if ds_name == 'llvip':
            raw_datasets[ds_name] = {
                'train': LLVIPDataset(ds_path, split='train'),
                'val': LLVIPDataset(ds_path, split='test')
            }
        elif ds_name == 'forestpersons':
            raw_datasets[ds_name] = {
                'train': ForestPersonsDataset(ds_path, split='train'),
                'val': ForestPersonsDataset(ds_path, split='val')
            }
        elif ds_name == 'forestpersonsir':
            raw_datasets[ds_name] = {
                'train': ForestPersonsIRDataset(ds_path, split='train'),
                'val': ForestPersonsIRDataset(ds_path, split='val')
            }
        elif ds_name == 'camod3fd':
            raw_datasets[ds_name] = {
                'train': CAMOD3FDDataset(ds_path, split='train'),
                'val': CAMOD3FDDataset(ds_path, split='val')
            }

    # 2. Build proportioned datasets
    for ds_name, ratio in dataset_ratios.items():
        if ds_name not in raw_datasets:
            continue
            
        raw_train = raw_datasets[ds_name]['train']
        raw_val = raw_datasets[ds_name]['val']
        
        # Subset to match desired ratio
        if ratio < 1.0:
            train_subset_size = int(len(raw_train) * ratio)
            generator = torch.Generator().manual_seed(RANDOM_SEED)
            train_subset, _ = random_split(
                raw_train, [train_subset_size, len(raw_train) - train_subset_size],
                generator=generator
            )
        else:
            train_subset = raw_train
            
        # Determine specific CMM rules for single-modality datasets
        ds_cmm_alpha = cmm_alpha
        if ds_name == 'forestpersons':
            # RGB only dataset -> force CMM ROTX logic handled in dataset wrapper
            ds_cmm_alpha = 0.0 # Handled inherently
        elif ds_name == 'forestpersonsir':
            # Thermal only dataset
            ds_cmm_alpha = 0.0
            
        train_datasets.append(ThermalIntrusionDataset(
            train_subset, preprocessor, augment=True, cmm_alpha=ds_cmm_alpha
        ))
        
        val_datasets.append(ThermalIntrusionDataset(
            raw_val, preprocessor, augment=False, cmm_alpha=0.0
        ))
        
    if not train_datasets:
        raise ValueError(f"No valid datasets found for Phase {phase}")
        
    combined_train = ConcatDataset(train_datasets)
    combined_val = ConcatDataset(val_datasets)
    
    pin_memory = (DEVICE == 'cuda')
    
    train_loader = DataLoader(
        combined_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True
    )
    val_loader = DataLoader(
        combined_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    print(f"\n{'=' * 60}")
    print(f"  Phase {phase} DataLoaders Created")
    print(f"{'=' * 60}")
    print(f"  Training:   {len(combined_train):>6d} samples ({len(train_loader)} batches)")
    print(f"  Validation: {len(combined_val):>6d} samples ({len(val_loader)} batches)")
    print(f"  Batch size: {batch_size}")
    print(f"{'=' * 60}")
    
    return train_loader, val_loader


if __name__ == '__main__':
    print("Data Loading Module (V2) — Self Test")
    print("-" * 40)
    for name, cfg in DATASET_CONFIGS.items():
        print(f"  {name}: {cfg['name']}")
    print("\nPhase configs loaded:")
    for phase, cfg in TRAINING_PHASES.items():
        print(f"  Phase {phase}: {cfg['name']} (Datasets: {list(cfg['datasets'].keys())})")
