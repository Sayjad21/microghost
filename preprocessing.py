"""
MicroGhost-Thermal: Preprocessing Module
==========================================
Handles all data transformation, normalization, augmentation,
and grid-based target encoding for thermal intrusion detection.

Key Components:
- ThermalPreprocessor: Orchestrates the full preprocessing pipeline
- GridEncoder: Encodes bounding boxes into grid-based detection targets
- Augmentation pipelines: Thermal-specific data augmentation
"""

import cv2
import math
import torch
import numpy as np

from config import (
    INPUT_SIZE, INPUT_CHANNELS, NUM_CLASSES, NUM_ANCHORS,
    DEFAULT_ANCHOR_RATIOS, DEFAULT_ANCHOR_SIZES,
    SMALL_GRID_W, SMALL_GRID_H, LARGE_GRID_W, LARGE_GRID_H,
    INPUT_WIDTH, INPUT_HEIGHT
)


# ============================================================================
# GRID ENCODER
# ============================================================================

class GridEncoder:
    """
    Grid-based target encoder for dual-head detection.

    Encodes raw bounding boxes into grid targets for loss computation.
    Adapted from the MicroGhost-Hand GridEncoder with:
    - 2 anchors per cell (down from 3)
    - 64×64 input (down from 160×160)
    - Binary classification (intrusion vs background)
    - Log-space width/height encoding for scale-invariant learning

    Target Format:
    - bbox_small (NUM_ANCHORS*4, SMALL_GRID, SMALL_GRID): BBox offsets
    - obj_small  (NUM_ANCHORS, SMALL_GRID, SMALL_GRID): Objectness
    - bbox_large (NUM_ANCHORS*4, LARGE_GRID, LARGE_GRID): BBox offsets
    - obj_large  (NUM_ANCHORS, LARGE_GRID, LARGE_GRID): Objectness
    - label      (scalar): 0=background, 1=intrusion (dominant target)
    """

    def __init__(self, num_anchors=None, anchor_ratios=None, anchor_sizes=None):
        self.num_anchors = num_anchors or NUM_ANCHORS

        if anchor_ratios is not None:
            self.ratios = torch.tensor(anchor_ratios, dtype=torch.float32)
        else:
            self.ratios = torch.tensor(DEFAULT_ANCHOR_RATIOS, dtype=torch.float32)

        if anchor_sizes is not None:
            self.anchor_sizes = torch.tensor(anchor_sizes, dtype=torch.float32)
        else:
            self.anchor_sizes = torch.tensor(DEFAULT_ANCHOR_SIZES, dtype=torch.float32)

        # Soft margin thresholds for scale assignment
        self.small_threshold = 0.10   # Below → only small head
        self.large_threshold = 0.20   # Above → only large head
        # Between 0.10-0.20 → assign to BOTH heads (soft margin)

    def update_anchors(self, anchor_ratios, anchor_sizes):
        """Update anchors after K-Means analysis."""
        self.ratios = torch.tensor(anchor_ratios, dtype=torch.float32)
        self.anchor_sizes = torch.tensor(anchor_sizes, dtype=torch.float32)
        print(f"✅ GridEncoder anchors updated!")
        print(f"   Ratios: {anchor_ratios}")
        print(f"   Sizes: {anchor_sizes}")

    def encode(self, boxes, labels, input_size=None):
        """
        Convert raw boxes to grid targets.

        Args:
            boxes: List of [cx, cy, w, h] (normalized 0-1)
            labels: List of class_ids (0=bg, 1=intrusion)
            input_size: Override input size

        Returns:
            dict with bbox_small, obj_small, bbox_large, obj_large, label
        """
        targets = {
            'bbox_small': torch.zeros(self.num_anchors * 4,
                                      SMALL_GRID_H, SMALL_GRID_W),
            'obj_small': torch.zeros(self.num_anchors,
                                     SMALL_GRID_H, SMALL_GRID_W),
            'bbox_large': torch.zeros(self.num_anchors * 4,
                                      LARGE_GRID_H, LARGE_GRID_W),
            'obj_large': torch.zeros(self.num_anchors,
                                     LARGE_GRID_H, LARGE_GRID_W),
            'label': torch.tensor(0).long(),  # Default: background
        }

        if len(boxes) == 0:
            return targets

        # Dominant target: largest bounding box
        box_areas = [b[2] * b[3] for b in boxes]
        dominant_idx = box_areas.index(max(box_areas))
        targets['label'] = torch.tensor(labels[dominant_idx]).long()

        for box, label in zip(boxes, labels):
            cx, cy, w, h = box
            box_area = w * h

            # Soft margin scale assignment
            scale_assignments = []
            if box_area < self.large_threshold:
                scale_assignments.append('small')
            if box_area >= self.small_threshold:
                scale_assignments.append('large')
            if not scale_assignments:
                scale_assignments = ['small']

            for scale in scale_assignments:
                if scale == 'small':
                    grid_w, grid_h = SMALL_GRID_W, SMALL_GRID_H
                    key_bbox = 'bbox_small'
                    key_obj = 'obj_small'
                else:
                    grid_w, grid_h = LARGE_GRID_W, LARGE_GRID_H
                    key_bbox = 'bbox_large'
                    key_obj = 'obj_large'

                # Grid cell
                grid_x = min(int(cx * grid_w), grid_w - 1)
                grid_y = min(int(cy * grid_h), grid_h - 1)

                # Best anchor (match aspect ratio)
                box_ratio = h / (w + 1e-6)
                ratio_diffs = torch.abs(self.ratios - box_ratio)
                anchor_idx = torch.argmin(ratio_diffs).item()

                # Fill objectness
                targets[key_obj][anchor_idx, grid_y, grid_x] = 1.0

                # Fill bbox (log-space for w, h)
                anchor_size = self.anchor_sizes[anchor_idx].item()
                off = anchor_idx * 4
                targets[key_bbox][off + 0, grid_y, grid_x] = \
                    (cx * grid_w) - grid_x
                targets[key_bbox][off + 1, grid_y, grid_x] = \
                    (cy * grid_h) - grid_y
                targets[key_bbox][off + 2, grid_y, grid_x] = \
                    torch.log(torch.tensor(w / anchor_size + 1e-6)).clamp(-3, 3)
                targets[key_bbox][off + 3, grid_y, grid_x] = \
                    torch.log(torch.tensor(h / anchor_size + 1e-6)).clamp(-3, 3)

        return targets


# ============================================================================
# THERMAL-SPECIFIC AUGMENTATIONS
# ============================================================================

class ThermalAugmentor:
    """
    Data augmentation pipeline designed for thermal/infrared images.

    Unlike RGB augmentation, thermal augmentation focuses on:
    - Sensor noise simulation (Gaussian noise)
    - Ambient temperature variation (brightness shifts)
    - Geometric transforms (flip, slight rotation)
    - Focus variation (Gaussian blur)
    - NO color augmentations (single channel!)

    All transforms preserve bounding box coordinates.
    """

    def __init__(self):
        self.input_h = INPUT_HEIGHT
        self.input_w = INPUT_WIDTH
        self._try_load_albumentations()

    def _try_load_albumentations(self):
        """Try to load albumentations; fall back to manual augmentation."""
        try:
            import albumentations as A
            self.A = A
            self.use_albumentations = True

            self.train_transform = A.Compose([
                # Geometric
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=10,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.6,
                ),
                # Thermal-specific: ambient temperature variation
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.7,
                ),
                # Sensor noise
                A.GaussNoise(var_limit=(5, 40), p=0.5),
                # Focus variation
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                # Resize
                A.Resize(self.input_h, self.input_w),
            ], additional_targets={'image_thermal': 'image'}, bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels'],
                min_visibility=0.3,
            ))

            self.val_transform = A.Compose([
                A.Resize(self.input_h, self.input_w),
            ], additional_targets={'image_thermal': 'image'}, bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels'],
            ))

        except ImportError:
            self.use_albumentations = False
            print("⚠️  albumentations not installed. Using basic augmentation.")
            print("   Install: pip install albumentations")

    def augment_train(self, image_rgb, image_thermal, bboxes, labels):
        if self.use_albumentations and bboxes:
            try:
                result = self.train_transform(
                    image=image_rgb, image_thermal=image_thermal, bboxes=bboxes, labels=labels
                )
                return result['image'], result['image_thermal'], result['bboxes'], result['labels']
            except Exception as e:
                # Fallback to simple resize
                image_rgb = cv2.resize(image_rgb, (self.input_w, self.input_h))
                image_thermal = cv2.resize(image_thermal, (self.input_w, self.input_h))
                return image_rgb, image_thermal, bboxes, labels
        else:
            return self._manual_augment(image_rgb, image_thermal, bboxes, labels)

    def augment_val(self, image_rgb, image_thermal, bboxes, labels):
        if self.use_albumentations and bboxes:
            try:
                result = self.val_transform(
                    image=image_rgb, image_thermal=image_thermal, bboxes=bboxes, labels=labels
                )
                return result['image'], result['image_thermal'], result['bboxes'], result['labels']
            except Exception as e:
                image_rgb = cv2.resize(image_rgb, (self.input_w, self.input_h))
                image_thermal = cv2.resize(image_thermal, (self.input_w, self.input_h))
                return image_rgb, image_thermal, bboxes, labels
        else:
            image_rgb = cv2.resize(image_rgb, (self.input_w, self.input_h))
            image_thermal = cv2.resize(image_thermal, (self.input_w, self.input_h))
            return image_rgb, image_thermal, bboxes, labels

    def _manual_augment(self, image_rgb, image_thermal, bboxes, labels):
        h, w = image_rgb.shape[:2]

        if np.random.random() < 0.5:
            image_rgb = np.fliplr(image_rgb).copy()
            image_thermal = np.fliplr(image_thermal).copy()
            new_bboxes = []
            for box in bboxes:
                xmin, ymin, xmax, ymax = box
                new_bboxes.append([w - xmax, ymin, w - xmin, ymax])
            bboxes = new_bboxes

        if np.random.random() < 0.5:
            shift = np.random.randint(-30, 31)
            image_thermal = np.clip(image_thermal.astype(np.int16) + shift, 0, 255).astype(np.uint8)

        if np.random.random() < 0.3:
            noise = np.random.normal(0, 10, image_thermal.shape).astype(np.int16)
            image_thermal = np.clip(image_thermal.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        image_rgb = cv2.resize(image_rgb, (self.input_w, self.input_h))
        image_thermal = cv2.resize(image_thermal, (self.input_w, self.input_h))
        return image_rgb, image_thermal, bboxes, labels


# ============================================================================
# THERMAL PREPROCESSOR (MAIN PIPELINE)
# ============================================================================

class ThermalPreprocessor:
    """
    Complete preprocessing pipeline for thermal intrusion detection.

    Pipeline:
    1. Load single-channel thermal image
    2. Normalize (min-max or histogram equalization)
    3. Apply augmentation (if training)
    4. Resize to INPUT_SIZE × INPUT_SIZE
    5. Convert to tensor
    6. Encode bounding boxes to grid targets via GridEncoder

    Usage:
        preprocessor = ThermalPreprocessor()
        img_tensor, targets = preprocessor.process(image, bboxes, labels)
    """

    def __init__(self, encoder=None, normalize_method='minmax'):
        """
        Args:
            encoder: GridEncoder instance (creates default if None)
            normalize_method: 'minmax', 'histogram', or 'clahe'
        """
        self.encoder = encoder or GridEncoder()
        self.augmentor = ThermalAugmentor()
        self.normalize_method = normalize_method

    def normalize(self, image):
        """
        Normalize thermal image.

        Args:
            image: (H, W) numpy uint8

        Returns:
            Normalized (H, W) numpy uint8
        """
        if self.normalize_method == 'histogram':
            return cv2.equalizeHist(image)
        elif self.normalize_method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        else:
            # min-max normalization to full 0-255 range
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                return ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            return image

    def process(self, image_rgb, image_thermal, bboxes_pascal, labels, img_size, augment=False):
        """
        Full preprocessing pipeline for dual-modality inputs.

        Args:
            image_rgb: (H, W, 3) numpy uint8 RGB
            image_thermal: (H, W) numpy uint8 single-channel thermal
            bboxes_pascal: list of [xmin, ymin, xmax, ymax] in pixel coords
            labels: list of class_ids
            img_size: (h_orig, w_orig)
            augment: Whether to apply training augmentation

        Returns:
            img_tensor: (4, INPUT_HEIGHT, INPUT_WIDTH) float tensor
            targets: dict of grid-encoded targets
        """
        h_orig, w_orig = img_size

        # Step 1: Normalize
        image_thermal = self.normalize(image_thermal)
        # RGB normalization is simple division by 255.0 later

        # Step 2: Augment (or just resize)
        if augment:
            image_rgb, image_thermal, bboxes_pascal, labels = self.augmentor.augment_train(
                image_rgb, image_thermal, bboxes_pascal, labels
            )
        else:
            image_rgb, image_thermal, bboxes_pascal, labels = self.augmentor.augment_val(
                image_rgb, image_thermal, bboxes_pascal, labels
            )

        # Step 3: Convert to float tensor (4, H, W)
        # Ensure sizes
        if image_thermal.shape[0] != self.augmentor.input_h or image_thermal.shape[1] != self.augmentor.input_w:
            image_thermal = cv2.resize(image_thermal, (self.augmentor.input_w, self.augmentor.input_h))
            image_rgb = cv2.resize(image_rgb, (self.augmentor.input_w, self.augmentor.input_h))

        # PyTorch expects channels first: (C, H, W)
        tensor_rgb = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        tensor_thermal = torch.from_numpy(image_thermal).unsqueeze(0).float() / 255.0
        
        # Stack into (4, H, W)
        img_tensor = torch.cat([tensor_rgb, tensor_thermal], dim=0)

        # Step 4: Convert bboxes to normalized [cx, cy, w, h]
        boxes_norm = []
        valid_labels = []
        h_new, w_new = self.augmentor.input_h, self.augmentor.input_w

        for bbox, lbl in zip(bboxes_pascal, labels):
            xmin, ymin, xmax, ymax = bbox

            # Scale bboxes if image was resized from original
            if h_orig != h_new or w_orig != w_new:
                scale_x = w_new / w_orig
                scale_y = h_new / h_orig
                xmin, xmax = xmin * scale_x, xmax * scale_x
                ymin, ymax = ymin * scale_y, ymax * scale_y

            cx = ((xmin + xmax) / 2) / w_new
            cy = ((ymin + ymax) / 2) / h_new
            w = (xmax - xmin) / w_new
            h = (ymax - ymin) / h_new

            # Clamp
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.01, min(1.0, w))
            h = max(0.01, min(1.0, h))

            boxes_norm.append([cx, cy, w, h])
            valid_labels.append(lbl)

        # Handle empty annotations
        if not boxes_norm:
            valid_labels = [0]  # Background
            boxes_norm = []

        # Step 5: Encode to grid targets
        targets = self.encoder.encode(boxes_norm, valid_labels, input_size=None) # handled via grid_h/w configs now

        return img_tensor, targets


# ============================================================================
# ANCHOR ANALYSIS UTILITIES
# ============================================================================

def analyze_dataset_anchors(dataset, num_anchors=2):
    """
    Run K-Means clustering on dataset bounding boxes to find
    optimal anchor ratios and sizes.

    Args:
        dataset: Any dataset with __getitem__ returning (image, annotations, size)
        num_anchors: Number of anchor clusters

    Returns:
        optimal_ratios: list of aspect ratios (h/w)
        optimal_sizes: list of relative sizes (sqrt of normalized area)
    """
    from sklearn.cluster import KMeans

    all_ratios = []
    all_sizes = []

    print(f"\n📊 Analyzing {len(dataset)} samples for anchor optimization...")

    for i in range(len(dataset)):
        try:
            _, annotations, (h_orig, w_orig) = dataset[i]
        except Exception:
            continue

        for ann in annotations:
            w = (ann['xmax'] - ann['xmin']) / w_orig
            h = (ann['ymax'] - ann['ymin']) / h_orig

            if w > 0.01 and h > 0.01:
                all_ratios.append(h / w)
                all_sizes.append(math.sqrt(w * h))

    if len(all_ratios) < num_anchors:
        print(f"⚠️  Not enough boxes ({len(all_ratios)}). Using defaults.")
        return DEFAULT_ANCHOR_RATIOS, DEFAULT_ANCHOR_SIZES

    ratios = np.array(all_ratios)
    sizes = np.array(all_sizes)

    print(f"   Found {len(ratios)} bounding boxes")
    print(f"   Aspect ratios: min={ratios.min():.2f}, "
          f"max={ratios.max():.2f}, mean={ratios.mean():.2f}")
    print(f"   Sizes: min={sizes.min():.3f}, "
          f"max={sizes.max():.3f}, mean={sizes.mean():.3f}")

    # K-Means for ratios
    km_ratios = KMeans(n_clusters=num_anchors, random_state=42, n_init=10)
    km_ratios.fit(ratios.reshape(-1, 1))
    optimal_ratios = sorted(km_ratios.cluster_centers_.flatten().tolist())

    # K-Means for sizes
    km_sizes = KMeans(n_clusters=num_anchors, random_state=42, n_init=10)
    km_sizes.fit(sizes.reshape(-1, 1))
    optimal_sizes = sorted(km_sizes.cluster_centers_.flatten().tolist())

    print(f"\n✅ OPTIMAL ANCHORS:")
    print(f"   Ratios (h/w): {[f'{r:.3f}' for r in optimal_ratios]}")
    print(f"   Sizes:        {[f'{s:.4f}' for s in optimal_sizes]}")

    return optimal_ratios, optimal_sizes


if __name__ == '__main__':
    # Quick test
    print("Preprocessing Module — Self Test")
    print("-" * 40)

    encoder = GridEncoder()
    preprocessor = ThermalPreprocessor(encoder=encoder)

    # Test with a dummy thermal frame
    dummy_image = np.random.randint(0, 255, (120, 160), dtype=np.uint8)
    dummy_bboxes = [[30, 20, 80, 100]]
    dummy_labels = [1]

    img_tensor, targets = preprocessor.process(
        dummy_image, dummy_bboxes, dummy_labels,
        img_size=(120, 160), augment=False
    )

    print(f"  Input image:  {dummy_image.shape}")
    print(f"  Output tensor: {img_tensor.shape}")
    print(f"  Targets:")
    for k, v in targets.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: {v.shape}")
        else:
            print(f"    {k}: {v}")
    print("✅ Preprocessing test passed!")
