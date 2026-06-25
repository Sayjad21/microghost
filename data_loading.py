"""
MicroGhost-Thermal: Data Loading Module
=========================================
Handles loading thermal/infrared datasets for training intrusion detection.

Supported Datasets:
1. LLVIP (Low-Light Vision Infrared-Visible) — VOC XML annotations
2. KAIST Multispectral Pedestrian — Custom text annotations

All datasets are converted to a unified internal format:
- Image: single-channel thermal frame (H, W) as numpy uint8/uint16
- Annotations: list of dicts [{class_id, xmin, ymin, xmax, ymax}, ...]
"""

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from glob import glob
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, WeightedRandomSampler, Subset
import torch

from config import (
    CLASS_MAP, NUM_CLASSES, INPUT_SIZE, INPUT_CHANNELS,
    DATASET_ROOT, DATASET_CONFIGS, ACTIVE_DATASET, get_dataset_path,
    BATCH_SIZE, NUM_WORKERS, VAL_RATIO, RANDOM_SEED, DEVICE,
    MIXED_CAMO_SAMPLE_WEIGHT,
    MIXED_EPOCH_LLVIP_CAP,
    MIXED_VAL_LLVIP_CAP,
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
    """
    LLVIP: Low-Light Vision Infrared-Visible Paired Dataset.

    Structure expected on Kaggle:
    ```
    llvip/
    ├── infrared/
    │   ├── train/     # Infrared training images (.jpg/.png)
    │   └── test/      # Infrared test images
    ├── visible/
    │   ├── train/     # Visible training images (NOT used)
    │   └── test/
    └── Annotations/   # VOC XML annotations for all images
    ```

    We ONLY use infrared images. The visible pairs are ignored.
    All 'Pedestrian' annotations are mapped to class 1 (intrusion).
    """

    def __init__(self, root_dir, split='train', verbose=True):
        """
        Args:
            root_dir: Path to the LLVIP dataset root folder
            split: 'train' or 'test'
            verbose: Print loading statistics
        """
        self.root_dir = root_dir
        self.split = split
        self.verbose = verbose

        # Build paths
        self.image_dir_thermal = os.path.join(root_dir, 'infrared', split)
        self.image_dir_rgb = os.path.join(root_dir, 'visible', split)
        self.annot_dir = os.path.join(root_dir, 'Annotations')

        # Error tracking
        self.xml_errors = []
        self.missing_annotations = []
        self.unknown_classes = set()

        # Collect image paths
        self.image_paths = []
        self.annot_paths = []

        if not os.path.exists(self.image_dir_thermal):
            if verbose:
                print(f"⚠️  LLVIP thermal dir not found: {self.image_dir_thermal}")
            return
            
        for ext in ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'):
            self.image_paths.extend(
                sorted(glob(os.path.join(self.image_dir_thermal, ext)))
            )

        # Match annotations and RGB pairs
        valid_paths = []
        for img_path_t in self.image_paths:
            name = os.path.basename(img_path_t)
            name_no_ext = os.path.splitext(name)[0]
            
            img_path_rgb = os.path.join(self.image_dir_rgb, name)
            xml_path = os.path.join(self.annot_dir, name_no_ext + '.xml')
            
            # Strict pairing: require both
            if os.path.exists(img_path_rgb):
                valid_paths.append((img_path_rgb, img_path_t, xml_path))
                
        self.paired_paths = valid_paths

        if verbose:
            found_annots = sum(1 for p in self.paired_paths if os.path.exists(p[2]))
            print(f"📁 LLVIP [{split}]: {len(self.paired_paths)} valid RGB+Thermal pairs, "
                  f"{found_annots} annotations found")

    def _parse_xml(self, xml_path, w_orig, h_orig):
        """Parse VOC XML into annotation dicts (no image I/O)."""
        annotations = []
        if not os.path.exists(xml_path):
            return annotations
        try:
            root = ET.parse(xml_path).getroot()
            for obj in root.findall('object'):
                name_elem = obj.find('name')
                if name_elem is None:
                    continue
                name = name_elem.text.strip()
                if name.lower() not in PERSON_CLASS_NAMES:
                    self.unknown_classes.add(name)
                    continue
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    continue
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))
                xmin = max(0, min(xmin, w_orig - 1))
                ymin = max(0, min(ymin, h_orig - 1))
                xmax = max(xmin + 1, min(xmax, w_orig))
                ymax = max(ymin + 1, min(ymax, h_orig))
                annotations.append({
                    'class_id': map_person_class_id(),
                    'xmin': xmin, 'ymin': ymin,
                    'xmax': xmax, 'ymax': ymax,
                })
        except ET.ParseError as e:
            self.xml_errors.append((xml_path, str(e)))
        except Exception as e:
            self.xml_errors.append((xml_path, str(e)))
        return annotations

    def iter_annotations(self):
        """Yield (annotations, (h, w)) from XML only — no image loading."""
        for _, _, xml_path in self.paired_paths:
            h_orig, w_orig = 1024, 1280
            if os.path.exists(xml_path):
                try:
                    root = ET.parse(xml_path).getroot()
                    size = root.find('size')
                    if size is not None:
                        h_orig = int(float(size.find('height').text))
                        w_orig = int(float(size.find('width').text))
                except Exception:
                    pass
            yield self._parse_xml(xml_path, w_orig, h_orig), (h_orig, w_orig)

    def __len__(self):
        return len(self.paired_paths)

    def __getitem__(self, idx):
        """
        Returns:
            image_tuple: (image_rgb, image_thermal)
            annotations: list of dicts
            img_size: tuple (height, width) of original image
        """
        img_path_rgb, img_path_t, xml_path = self.paired_paths[idx]
        
        # Load Thermal
        image_thermal = cv2.imread(img_path_t, cv2.IMREAD_GRAYSCALE)
        if image_thermal is None:
            image_thermal = cv2.imread(img_path_t)
            if image_thermal is not None:
                image_thermal = cv2.cvtColor(image_thermal, cv2.COLOR_BGR2GRAY)
            else:
                raise ValueError(f"Failed to load thermal image: {img_path_t}")

        # Load RGB
        image_rgb = cv2.imread(img_path_rgb)
        if image_rgb is None:
            raise ValueError(f"Failed to load RGB image: {img_path_rgb}")
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB) # Convert BGR to RGB

        h_orig, w_orig = image_thermal.shape[:2]
        annotations = self._parse_xml(xml_path, w_orig, h_orig)
        if not os.path.exists(xml_path):
            self.missing_annotations.append(xml_path)

        return (image_rgb, image_thermal), annotations, (h_orig, w_orig)

    def report_issues(self):
        """Print summary of data loading issues."""
        if self.xml_errors:
            print(f"\n⚠️  XML ERRORS ({len(self.xml_errors)}):")
            for path, err in self.xml_errors[:5]:
                print(f"   {os.path.basename(path)}: {err}")
        if self.missing_annotations:
            print(f"\n⚠️  MISSING ANNOTATIONS: {len(self.missing_annotations)}")
        if self.unknown_classes:
            print(f"\n⚠️  UNKNOWN CLASSES (skipped): {self.unknown_classes}")


# ============================================================================
# CAMO-M3FD DATASET LOADER
# ============================================================================

def mask_to_boxes(mask, min_area=400, class_id=None):
    """
    Convert a binary segmentation mask to axis-aligned bounding boxes.

    Each connected foreground region becomes one detection target.
    """
    class_id = class_id if class_id is not None else CLASS_MAP['person_visible']
    if mask is None:
        return []

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotations = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh < min_area:
            continue
        annotations.append({
            'class_id': class_id,
            'xmin': x,
            'ymin': y,
            'xmax': x + bw,
            'ymax': y + bh,
        })
    return annotations


class CAMOM3FDDataset(Dataset):
    """
    CAMO-M3FD: Camouflaged pedestrian RGB + thermal pairs with segmentation masks.

    Structure:
    ```
    CAMO-M3FD/
    ├── train/
    │   ├── Imgs/      # Visible RGB (.png)
    │   ├── Thermal/   # Thermal (.png)
    │   └── GT/        # Binary masks (0 / 255)
    ├── val/
    └── test/
    ```

    Masks are converted to bounding boxes (one per connected region).
    """

    def __init__(self, root_dir, split='train', min_mask_area=None, verbose=True):
        self.root_dir = root_dir
        self.split = split
        self.verbose = verbose
        cfg = DATASET_CONFIGS.get('camo_m3fd', {})
        self.min_mask_area = min_mask_area or cfg.get('min_mask_area', 100)
        self.class_id = cfg.get('class_id', CLASS_MAP['person_visible'])

        split_dir = os.path.join(root_dir, split)
        self.rgb_dir = os.path.join(split_dir, cfg.get('rgb_subdir', 'Imgs'))
        self.thermal_dir = os.path.join(split_dir, cfg.get('thermal_subdir', 'Thermal'))
        self.gt_dir = os.path.join(split_dir, cfg.get('gt_subdir', 'GT'))

        self.paired_paths = []
        self.missing_gt = []

        if not os.path.isdir(self.rgb_dir):
            if verbose:
                print(f"⚠️  CAMO-M3FD RGB dir not found: {self.rgb_dir}")
            return

        for ext in ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG'):
            for rgb_path in sorted(glob(os.path.join(self.rgb_dir, ext))):
                stem = os.path.splitext(os.path.basename(rgb_path))[0]
                thermal_path = os.path.join(self.thermal_dir, stem + '.png')
                gt_path = os.path.join(self.gt_dir, stem + '.png')
                if os.path.exists(thermal_path):
                    self.paired_paths.append((rgb_path, thermal_path, gt_path))
                    if not os.path.exists(gt_path):
                        self.missing_gt.append(gt_path)

        if verbose:
            print(f"📁 CAMO-M3FD [{split}]: {len(self.paired_paths)} RGB+Thermal pairs, "
                  f"{len(self.paired_paths) - len(self.missing_gt)} masks found")

    def _parse_mask(self, gt_path, w_orig, h_orig):
        """Load GT mask and return bounding-box annotations."""
        if not os.path.exists(gt_path):
            return []
        mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return []
        if mask.shape[0] != h_orig or mask.shape[1] != w_orig:
            mask = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        return mask_to_boxes(mask, min_area=self.min_mask_area, class_id=self.class_id)

    def iter_annotations(self):
        """Yield (annotations, (h, w)) from GT masks only — for K-Means anchor analysis."""
        for rgb_path, _, gt_path in self.paired_paths:
            rgb = cv2.imread(rgb_path)
            if rgb is None:
                continue
            h_orig, w_orig = rgb.shape[:2]
            yield self._parse_mask(gt_path, w_orig, h_orig), (h_orig, w_orig)

    def __len__(self):
        return len(self.paired_paths)

    def __getitem__(self, idx):
        rgb_path, thermal_path, gt_path = self.paired_paths[idx]

        image_rgb = cv2.imread(rgb_path)
        if image_rgb is None:
            raise ValueError(f"Failed to load RGB image: {rgb_path}")
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

        image_thermal = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
        if image_thermal is None:
            raise ValueError(f"Failed to load thermal image: {thermal_path}")

        h_orig, w_orig = image_thermal.shape[:2]
        annotations = self._parse_mask(gt_path, w_orig, h_orig)

        return (image_rgb, image_thermal), annotations, (h_orig, w_orig)

    def report_issues(self):
        if self.missing_gt:
            print(f"\n⚠️  CAMO-M3FD missing GT masks: {len(self.missing_gt)}")


# ============================================================================
# KAIST MULTISPECTRAL DATASET LOADER
# ============================================================================

class KAISTDataset(Dataset):
    """
    KAIST Multispectral Pedestrian Detection Benchmark.

    Structure expected on Kaggle:
    ```
    kaist-multispectral/
    ├── images/
    │   ├── set00/
    │   │   └── V000/
    │   │       ├── lwir/        # Thermal (LWIR) images
    │   │       │   ├── I00000.png
    │   │       │   └── ...
    │   │       └── visible/     # Visible images (NOT used)
    │   └── set01/ ...
    └── annotations/
        ├── set00/
        │   └── V000/
        │       ├── I00000.txt
        │       └── ...
        └── set01/ ...
    ```

    Annotation format (per line):
        class_label x y w h (occluded) (ignore)
    """

    def __init__(self, root_dir, sets=None, verbose=True):
        """
        Args:
            root_dir: Path to the KAIST dataset root
            sets: List of set indices to load, e.g. [0,1,2]. None = all.
            verbose: Print loading statistics
        """
        self.root_dir = root_dir
        self.verbose = verbose

        self.paired_paths = []
        self.unknown_classes = set()

        # Auto-discover sets
        images_root = os.path.join(root_dir, 'images')
        annot_root = os.path.join(root_dir, 'annotations')

        if not os.path.exists(images_root):
            # Try flat structure
            images_root = root_dir
            annot_root = root_dir

        set_dirs = sorted(glob(os.path.join(images_root, 'set*')))
        if sets is not None:
            set_dirs = [d for d in set_dirs
                        if any(f'set{s:02d}' in d for s in sets)]

        for set_dir in set_dirs:
            set_name = os.path.basename(set_dir)
            video_dirs = sorted(glob(os.path.join(set_dir, 'V*')))

            for vid_dir in video_dirs:
                vid_name = os.path.basename(vid_dir)
                lwir_dir = os.path.join(vid_dir, 'lwir')

                if not os.path.exists(lwir_dir):
                    continue

                for img_file in sorted(os.listdir(lwir_dir)):
                    if not img_file.endswith(('.png', '.jpg')):
                        continue

                    lwir_path = os.path.join(lwir_dir, img_file)
                    vis_path = os.path.join(vid_dir, 'visible', img_file)
                    name_no_ext = os.path.splitext(img_file)[0]

                    # Find annotation
                    annot_path = os.path.join(
                        annot_root, set_name, vid_name, name_no_ext + '.txt'
                    )

                    if os.path.exists(vis_path):
                        self.paired_paths.append((vis_path, lwir_path, annot_path))

        if verbose:
            print(f"📁 KAIST: {len(self.paired_paths)} valid RGB+LWIR pairs loaded")

    def __len__(self):
        return len(self.paired_paths)

    def __getitem__(self, idx):
        """
        Returns:
            image_tuple: (image_rgb, image_thermal)
            annotations: list of dicts {class_id, xmin, ymin, xmax, ymax}
            img_size: tuple (height, width)
        """
        img_path_rgb, img_path_t, annot_path = self.paired_paths[idx]
        
        # Load Thermal
        image_thermal = cv2.imread(img_path_t, cv2.IMREAD_GRAYSCALE)
        if image_thermal is None:
            image_thermal = cv2.imread(img_path_t)
            if image_thermal is not None:
                image_thermal = cv2.cvtColor(image_thermal, cv2.COLOR_BGR2GRAY)
            else:
                raise ValueError(f"Failed to load thermal image: {img_path_t}")

        # Load RGB
        image_rgb = cv2.imread(img_path_rgb)
        if image_rgb is None:
            raise ValueError(f"Failed to load RGB image: {img_path_rgb}")
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB) # Convert BGR to RGB

        h_orig, w_orig = image_thermal.shape[:2]
        annotations = []

        if os.path.exists(annot_path):
            try:
                with open(annot_path, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('%'):
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                        continue

                    class_name = parts[0].lower()
                    if class_name in PERSON_CLASS_NAMES:
                        class_id = map_person_class_id()
                    else:
                        self.unknown_classes.add(class_name)
                        continue

                    x, y, w, h = (int(float(parts[1])), int(float(parts[2])),
                                  int(float(parts[3])), int(float(parts[4])))

                    # Check ignore flag if present
                    if len(parts) > 5:
                        ignore = int(parts[5]) if parts[5].isdigit() else 0
                        if ignore:
                            continue

                    xmin = max(0, x)
                    ymin = max(0, y)
                    xmax = min(w_orig, x + w)
                    ymax = min(h_orig, y + h)

                    if xmax > xmin and ymax > ymin:
                        annotations.append({
                            'class_id': class_id,
                            'xmin': xmin, 'ymin': ymin,
                            'xmax': xmax, 'ymax': ymax,
                        })

            except Exception as e:
                pass # Ignore malformed lines

        return (image_rgb, image_thermal), annotations, (h_orig, w_orig)


# ============================================================================
# FLIR ADAS v2 DATASET LOADER (COCO JSON FORMAT)
# ============================================================================

class FLIRv2Dataset(Dataset):
    """
    FLIR Thermal Dataset v2 with paired RGB + thermal frames.
    Uses COCO JSON format for thermal annotations.
    """
    def __init__(self, root_dir, split='train', verbose=True):
        import json
        self.root_dir = root_dir
        self.split = split
        self.verbose = verbose

        cfg = DATASET_CONFIGS['flirv2']
        img_folder = cfg['train_dir'] if split == 'train' else cfg['val_dir']
        rgb_folder = cfg['rgb_train_dir'] if split == 'train' else cfg['rgb_val_dir']
        annot_file = cfg['annotations_train'] if split == 'train' else cfg['annotations_val']

        self.image_dir = os.path.join(root_dir, img_folder)
        self.rgb_dir = os.path.join(root_dir, rgb_folder)
        annot_path = os.path.join(root_dir, annot_file)

        self.paired_paths = []  # (rgb_path, thermal_path, img_id)
        self.annotations = {}  # img_id -> list of dicts

        if not os.path.exists(annot_path):
            if verbose:
                print(f"⚠️  FLIRv2 annotation not found: {annot_path}")
            return

        with open(annot_path, 'r') as f:
            coco_data = json.load(f)

        cat_map = {cat['id']: cat['name'].lower() for cat in coco_data['categories']}

        target_cats = set()
        for cat_id, name in cat_map.items():
            if name in PERSON_CLASS_NAMES:
                target_cats.add(cat_id)

        img_id_to_thermal = {}
        for img in coco_data['images']:
            thermal_path = os.path.join(self.image_dir, img['file_name'])
            if not os.path.exists(thermal_path):
                thermal_path = os.path.join(
                    self.image_dir, os.path.basename(img['file_name'])
                )

            rgb_path = os.path.join(self.rgb_dir, os.path.basename(img['file_name']))
            if not os.path.exists(rgb_path):
                rgb_path = thermal_path.replace(img_folder, rgb_folder)

            img_id_to_thermal[img['id']] = (rgb_path, thermal_path)
            self.annotations[img['id']] = []

        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            cat_id = ann['category_id']

            if cat_id in target_cats and img_id in self.annotations:
                x, y, w, h = ann['bbox']
                self.annotations[img_id].append({
                    'class_id': map_person_class_id(),
                    'xmin': int(x),
                    'ymin': int(y),
                    'xmax': int(x + w),
                    'ymax': int(y + h),
                })

        for img_id, (rgb_path, thermal_path) in img_id_to_thermal.items():
            if os.path.exists(rgb_path) and os.path.exists(thermal_path):
                self.paired_paths.append((rgb_path, thermal_path, img_id))

        if verbose:
            print(f"📁 FLIRv2 [{split}]: {len(self.paired_paths)} valid RGB+Thermal pairs")

    def __len__(self):
        return len(self.paired_paths)

    def __getitem__(self, idx):
        img_path_rgb, img_path_t, img_id = self.paired_paths[idx]

        image_thermal = cv2.imread(img_path_t, cv2.IMREAD_GRAYSCALE)
        if image_thermal is None:
            raise ValueError(f"Failed to load thermal image: {img_path_t}")

        image_rgb = cv2.imread(img_path_rgb)
        if image_rgb is None:
            raise ValueError(f"Failed to load RGB image: {img_path_rgb}")
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

        h_orig, w_orig = image_thermal.shape[:2]
        annotations = self.annotations.get(img_id, [])

        clamped_annots = []
        for ann in annotations:
            xmin = max(0, ann['xmin'])
            ymin = max(0, ann['ymin'])
            xmax = min(w_orig, ann['xmax'])
            ymax = min(h_orig, ann['ymax'])
            if xmax > xmin and ymax > ymin:
                clamped_annots.append({
                    'class_id': ann['class_id'],
                    'xmin': xmin, 'ymin': ymin,
                    'xmax': xmax, 'ymax': ymax,
                })

        return (image_rgb, image_thermal), clamped_annots, (h_orig, w_orig)

# ============================================================================
# UNIFIED DATASET WRAPPER
# ============================================================================

class ThermalIntrusionDataset(Dataset):
    """
    Unified wrapper that takes raw data from any thermal dataset loader
    and applies preprocessing + grid encoding for training.

    This is the Dataset that DataLoaders consume.
    """

    def __init__(self, base_dataset, preprocessor, augment=False, verbose=True):
        """
        Args:
            base_dataset: An LLVIPDataset, KAISTDataset, or similar
            preprocessor: A ThermalPreprocessor instance (from preprocessing.py)
            augment: Whether to apply training augmentations
            verbose: Print loading statistics
        """
        self.base_dataset = base_dataset
        self.preprocessor = preprocessor
        self.augment = augment
        self.verbose = verbose

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        Returns:
            img_tensor: (4, H, W) float tensor (3 RGB + 1 Thermal), normalized
            targets: dict with grid-encoded detection targets
        """
        (image_rgb, image_thermal), annotations, (h_orig, w_orig) = self.base_dataset[idx]

        # Convert annotations to pascal_voc format for augmentation
        bboxes_pascal = []
        labels = []
        for ann in annotations:
            bboxes_pascal.append([
                ann['xmin'], ann['ymin'], ann['xmax'], ann['ymax']
            ])
            labels.append(ann['class_id'])

        # Apply preprocessing (resize, normalize, augment, encode)
        img_tensor, targets = self.preprocessor.process(
            image_rgb=image_rgb,
            image_thermal=image_thermal,
            bboxes_pascal=bboxes_pascal,
            labels=labels,
            img_size=(h_orig, w_orig),
            augment=self.augment,
        )

        return img_tensor, targets
# ============================================================================
# DATALOADER FACTORY
# ============================================================================

def create_dataloaders(dataset_name, preprocessor, dataset_root=None,
                       batch_size=None, num_workers=None, val_ratio=None):
    """
    Create train and validation DataLoaders for a given dataset.

    Args:
        dataset_name: 'llvip', 'kaist', or 'flirv2'
        dataset_root: Path to the dataset root folder (auto-resolved if None)
        preprocessor: ThermalPreprocessor instance
        batch_size: Override config batch size
        num_workers: Override config num workers
        val_ratio: Override config validation ratio

    Returns:
        train_loader, val_loader
    """
    batch_size = batch_size or BATCH_SIZE
    num_workers = num_workers or NUM_WORKERS
    val_ratio = val_ratio or VAL_RATIO
    dataset_root = dataset_root or get_dataset_path(dataset_name)

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_root}\n"
            f"Set MICROGHOST_DATA_ROOT or MICROGHOST_{dataset_name.upper()}_PATH, "
            f"or pass --data-root to main.py"
        )

    # Load raw dataset
    if dataset_name == 'llvip':
        # LLVIP has explicit train/test split
        raw_train = LLVIPDataset(dataset_root, split='train')
        raw_test = LLVIPDataset(dataset_root, split='test')

        # Use LLVIP test as our validation (it's already a separate split)
        train_dataset = ThermalIntrusionDataset(
            raw_train, preprocessor, augment=True
        )
        val_dataset = ThermalIntrusionDataset(
            raw_test, preprocessor, augment=False
        )

        raw_train.report_issues()

    elif dataset_name == 'kaist':
        # KAIST doesn't have a clean train/test split, so we split ourselves
        raw_dataset = KAISTDataset(dataset_root)

        full_dataset = ThermalIntrusionDataset(
            raw_dataset, preprocessor, augment=True
        )

        # Split into train/val
        total = len(full_dataset)
        val_size = int(total * val_ratio)
        train_size = total - val_size

        generator = torch.Generator().manual_seed(RANDOM_SEED)
        train_dataset, val_dataset_aug = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

        # Create a non-augmented version for validation
        val_no_aug = ThermalIntrusionDataset(
            raw_dataset, preprocessor, augment=False
        )
        val_indices = val_dataset_aug.indices
        val_dataset = torch.utils.data.Subset(val_no_aug, val_indices)

    elif dataset_name == 'flirv2':
        # FLIRv2 has explicit train/val split in COCO format
        raw_train = FLIRv2Dataset(dataset_root, split='train')
        raw_val = FLIRv2Dataset(dataset_root, split='val')
        
        train_dataset = ThermalIntrusionDataset(
            raw_train, preprocessor, augment=True
        )
        val_dataset = ThermalIntrusionDataset(
            raw_val, preprocessor, augment=False
        )

    elif dataset_name == 'camo_m3fd':
        raw_train = CAMOM3FDDataset(dataset_root, split='train')
        raw_val = CAMOM3FDDataset(dataset_root, split='val')

        train_dataset = ThermalIntrusionDataset(
            raw_train, preprocessor, augment=True
        )
        val_dataset = ThermalIntrusionDataset(
            raw_val, preprocessor, augment=False
        )
        raw_train.report_issues()

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Supported: llvip, kaist, flirv2, camo_m3fd")

    # Create DataLoaders
    pin_memory = (DEVICE == 'cuda')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"\n{'=' * 60}")
    print(f"  DataLoaders Created [{dataset_name.upper()}]")
    print(f"{'=' * 60}")
    print(f"  Training:   {len(train_dataset):>6d} samples "
          f"({len(train_loader)} batches)")
    print(f"  Validation: {len(val_dataset):>6d} samples "
          f"({len(val_loader)} batches)")
    print(f"  Batch size: {batch_size}")
    print(f"{'=' * 60}")

    return train_loader, val_loader


def create_mixed_dataloaders(preprocessor, llvip_root=None, camo_root=None,
                             batch_size=None, num_workers=None,
                             camo_weight=None):
    """
    Mixed LLVIP + CAMO-M3FD training with weighted sampling.

    ~70% LLVIP / ~30% CAMO batches by default to prevent forgetting while
    learning camouflaged pedestrians.
    """
    batch_size = batch_size or BATCH_SIZE
    num_workers = num_workers or NUM_WORKERS
    camo_weight = camo_weight if camo_weight is not None else MIXED_CAMO_SAMPLE_WEIGHT
    llvip_root = llvip_root or get_dataset_path('llvip')
    camo_root = camo_root or get_dataset_path('camo_m3fd')

    raw_llvip_train = LLVIPDataset(llvip_root, split='train')
    raw_llvip_val = LLVIPDataset(llvip_root, split='test', verbose=False)
    raw_camo_train = CAMOM3FDDataset(camo_root, split='train')
    raw_camo_val = CAMOM3FDDataset(camo_root, split='val', verbose=False)

    llvip_train = ThermalIntrusionDataset(raw_llvip_train, preprocessor, augment=True)
    camo_train = ThermalIntrusionDataset(raw_camo_train, preprocessor, augment=True)
    llvip_val = ThermalIntrusionDataset(raw_llvip_val, preprocessor, augment=False)
    camo_val = ThermalIntrusionDataset(raw_camo_val, preprocessor, augment=False)
    llvip_val_subset = Subset(llvip_val, range(min(len(llvip_val), MIXED_VAL_LLVIP_CAP)))

    n_llvip = len(llvip_train)
    n_camo = len(camo_train)
    llvip_w = (1.0 - camo_weight) / max(n_llvip, 1)
    camo_w = camo_weight / max(n_camo, 1)
    weights = [llvip_w] * n_llvip + [camo_w] * n_camo
    epoch_samples = min(n_llvip, MIXED_EPOCH_LLVIP_CAP) + n_camo
    mixed_train = ConcatDataset([llvip_train, camo_train])
    mixed_val = ConcatDataset([llvip_val_subset, camo_val])

    pin_memory = (DEVICE == 'cuda')
    sampler = WeightedRandomSampler(
        weights, num_samples=epoch_samples, replacement=True,
    )

    train_loader = DataLoader(
        mixed_train,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        mixed_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    raw_llvip_train.report_issues()
    raw_camo_train.report_issues()

    print(f"\n{'=' * 60}")
    print(f"  DataLoaders Created [MIXED LLVIP + CAMO_M3FD]")
    print(f"{'=' * 60}")
    print(f"  LLVIP train:  {n_llvip:>6d}  |  CAMO train: {n_camo:>4d}")
    print(f"  CAMO sample weight: {camo_weight:.0%}")
    print(f"  Val (LLVIP {len(llvip_val_subset)} + CAMO {len(camo_val)}): {len(mixed_val)} samples")
    print(f"  Epoch samples:  {epoch_samples} (~{epoch_samples // batch_size} batches)")
    print(f"  Batch size: {batch_size}")
    print(f"{'=' * 60}")

    return train_loader, val_loader


def get_val_base_dataset(dataset_name, dataset_root=None, val_ratio=None):
    """
    Return the raw validation Dataset (RGB+thermal pairs + VOC-style annotations).

    Used for detection mAP evaluation without grid-encoded targets.
    """
    dataset_root = dataset_root or get_dataset_path(dataset_name)
    val_ratio = val_ratio or VAL_RATIO

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_root}"
        )

    if dataset_name == 'llvip':
        return LLVIPDataset(dataset_root, split='test', verbose=False)

    if dataset_name == 'kaist':
        raw_dataset = KAISTDataset(dataset_root, verbose=False)
        total = len(raw_dataset)
        val_size = int(total * val_ratio)
        train_size = total - val_size
        generator = torch.Generator().manual_seed(RANDOM_SEED)
        _, val_subset = random_split(
            raw_dataset, [train_size, val_size], generator=generator
        )
        return val_subset

    if dataset_name == 'flirv2':
        return FLIRv2Dataset(dataset_root, split='val', verbose=False)

    if dataset_name == 'camo_m3fd':
        return CAMOM3FDDataset(dataset_root, split='test', verbose=False)

    raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == '__main__':
    # Quick test: check if dataset can be located
    print("Data Loading Module — Self Test")
    print("-" * 40)
    for name, cfg in DATASET_CONFIGS.items():
        print(f"  {name}: {cfg['name']}")
        print(f"    Classes: {cfg['classes']}")
    print(f"\n  Active: {ACTIVE_DATASET}")
