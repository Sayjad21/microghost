"""
MicroGhost-Thermal: Training Module
=====================================
Complete training pipeline for thermal intrusion detection.

Components:
- CIoU Loss (Complete Intersection over Union)
- Multi-task loss (bbox + objectness + classification)
- Metrics tracker with real IoU computation
- Trainer class with LR scheduling, early stopping, checkpointing
- Visualization utilities
"""

import os
import math
import time
import psutil
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for Kaggle compatibility
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm

from config import (
    INPUT_SIZE, INPUT_WIDTH, INPUT_HEIGHT, NUM_CLASSES, NUM_ANCHORS,
    DEFAULT_ANCHOR_SIZES, CLASS_MAP, OBJ_METRIC_THRESHOLD,
    BBOX_WEIGHT, OBJ_WEIGHT, CLASS_WEIGHT,
    LEARNING_RATE, WEIGHT_DECAY, EPOCHS, PATIENCE,
    LR_T0, LR_T_MULT, LR_MIN,
    MODEL_SAVE_DIR, BEST_MODEL_PATH, LAST_MODEL_PATH, LOG_DIR, DEVICE,
)


def _to_intrusion_binary(class_ids):
    """Collapse multi-class labels to binary: 0=background, 1=any intrusion."""
    return (np.asarray(class_ids) > 0).astype(np.int32)


def _decode_grid_boxes(bbox_tensor, obj_mask, grid_w, grid_h, anchor_size):
    """Decode positive grid cells to normalized [x1, y1, x2, y2]."""
    pos = torch.nonzero(obj_mask, as_tuple=False)
    if pos.numel() == 0:
        return torch.zeros(0, 4)

    gy = pos[:, 0].float()
    gx = pos[:, 1].float()
    cx = (gx + bbox_tensor[0][pos[:, 0], pos[:, 1]]) / grid_w
    cy = (gy + bbox_tensor[1][pos[:, 0], pos[:, 1]]) / grid_h
    w = anchor_size * torch.exp(bbox_tensor[2][pos[:, 0], pos[:, 1]].clamp(-3, 3))
    h = anchor_size * torch.exp(bbox_tensor[3][pos[:, 0], pos[:, 1]].clamp(-3, 3))
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def _iou_xyxy(boxes_a, boxes_b):
    """Pairwise IoU for equal-length box tensors (N, 4)."""
    inter_x1 = torch.max(boxes_a[:, 0], boxes_b[:, 0])
    inter_y1 = torch.max(boxes_a[:, 1], boxes_b[:, 1])
    inter_x2 = torch.min(boxes_a[:, 2], boxes_b[:, 2])
    inter_y2 = torch.min(boxes_a[:, 3], boxes_b[:, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a + area_b - inter + 1e-7
    return (inter / union).cpu().numpy()


# ============================================================================
# 1. CIoU LOSS
# ============================================================================

def ciou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    Complete IoU Loss — the standard for modern object detection.

    CIoU = IoU - (ρ²(b, b_gt) / c²) - αv
    where:
        ρ² = squared Euclidean distance between centers
        c² = squared diagonal of smallest enclosing box
        v  = aspect ratio consistency
        α  = trade-off parameter

    Args:
        pred_boxes:   (N, 4) tensor [cx, cy, w, h]
        target_boxes: (N, 4) tensor [cx, cy, w, h]

    Returns:
        CIoU loss (1 - CIoU), lower is better
    """
    # Convert [cx, cy, w, h] to [x1, y1, x2, y2]
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    tgt_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    tgt_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    tgt_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    tgt_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

    # Intersection
    inter_x1 = torch.max(pred_x1, tgt_x1)
    inter_y1 = torch.max(pred_y1, tgt_y1)
    inter_x2 = torch.min(pred_x2, tgt_x2)
    inter_y2 = torch.min(pred_y2, tgt_y2)

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * \
                 (inter_y2 - inter_y1).clamp(min=0)

    # Union
    pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
    tgt_area = target_boxes[:, 2] * target_boxes[:, 3]
    union_area = pred_area + tgt_area - inter_area + eps

    iou = inter_area / union_area

    # Enclosing box
    enc_x1 = torch.min(pred_x1, tgt_x1)
    enc_y1 = torch.min(pred_y1, tgt_y1)
    enc_x2 = torch.max(pred_x2, tgt_x2)
    enc_y2 = torch.max(pred_y2, tgt_y2)

    c_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    # Center distance
    center_dist_sq = (pred_boxes[:, 0] - target_boxes[:, 0]) ** 2 + \
                     (pred_boxes[:, 1] - target_boxes[:, 1]) ** 2

    # Aspect ratio consistency
    pred_ratio = torch.atan(pred_boxes[:, 2] / (pred_boxes[:, 3] + eps))
    tgt_ratio = torch.atan(target_boxes[:, 2] / (target_boxes[:, 3] + eps))
    v = (4 / (math.pi ** 2)) * (pred_ratio - tgt_ratio) ** 2

    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou_val = iou - (center_dist_sq / c_sq) - (alpha * v)
    return (1 - ciou_val).mean()


class CIoUBBoxLoss(nn.Module):
    """
    CIoU bounding box loss for grid-based detection.

    Handles [cx_offset, cy_offset, log_w, log_h] format and
    computes CIoU only at positive (object-containing) cells.
    """

    def __init__(self, anchor_sizes=None):
        super().__init__()
        if anchor_sizes is not None:
            self.anchor_sizes = torch.tensor(anchor_sizes, dtype=torch.float32)
        else:
            self.anchor_sizes = torch.tensor(DEFAULT_ANCHOR_SIZES,
                                             dtype=torch.float32)

    def forward(self, pred_bbox, target_bbox, obj_mask, grid_w, grid_h):
        """
        Args:
            pred_bbox:   (B, A*4, H, W) predicted bbox params
            target_bbox: (B, A*4, H, W) target bbox params
            obj_mask:    (B, A, H, W)   objectness targets
            grid_w, grid_h:   int (width and height grids)

        Returns:
            CIoU loss (scalar)
        """
        device = pred_bbox.device
        self.anchor_sizes = self.anchor_sizes.to(device)

        batch_size = pred_bbox.shape[0]
        num_anchors = self.anchor_sizes.shape[0]
        total_loss = 0.0
        num_positive = 0

        for b in range(batch_size):
            for a in range(num_anchors):
                mask = obj_mask[b, a] > 0.5
                if not mask.any():
                    continue

                off = a * 4
                pred_cx = pred_bbox[b, off + 0][mask]
                pred_cy = pred_bbox[b, off + 1][mask]
                pred_lw = pred_bbox[b, off + 2][mask]
                pred_lh = pred_bbox[b, off + 3][mask]

                tgt_cx = target_bbox[b, off + 0][mask]
                tgt_cy = target_bbox[b, off + 1][mask]
                tgt_lw = target_bbox[b, off + 2][mask]
                tgt_lh = target_bbox[b, off + 3][mask]

                pos = torch.nonzero(mask, as_tuple=False)
                gy = pos[:, 0].float()
                gx = pos[:, 1].float()

                anchor_size = self.anchor_sizes[a]

                # Decode to absolute (normalized 0-1)
                abs_pred = torch.stack([
                    (gx + pred_cx) / grid_w,
                    (gy + pred_cy) / grid_h,
                    anchor_size * torch.exp(pred_lw.clamp(-3, 3)),
                    anchor_size * torch.exp(pred_lh.clamp(-3, 3)),
                ], dim=1)

                abs_tgt = torch.stack([
                    (gx + tgt_cx) / grid_w,
                    (gy + tgt_cy) / grid_h,
                    anchor_size * torch.exp(tgt_lw.clamp(-3, 3)),
                    anchor_size * torch.exp(tgt_lh.clamp(-3, 3)),
                ], dim=1)

                loss = ciou_loss(abs_pred, abs_tgt)
                total_loss += loss * mask.sum()
                num_positive += mask.sum()

        if num_positive > 0:
            return total_loss / num_positive
        return torch.tensor(0.0, device=device, requires_grad=True)


# ============================================================================
# 2. COMBINED LOSS
# ============================================================================

class OHEMBCEWithLogitsLoss(nn.Module):
    """
    Online Hard Example Mining (OHEM) for Objectness BCE Loss.
    Maintains a 1:3 ratio of positive to hard-negative samples.
    This naturally penalizes 'hot bonnet' false positives by dynamically
    mining the background cells with the highest loss.
    """
    def __init__(self, pos_ratio=3.0):
        super().__init__()
        self.pos_ratio = pos_ratio

    def forward(self, pred, target):
        # unreduced loss: (B, A, H, W)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Flatten
        loss = loss.view(-1)
        target = target.view(-1)
        
        pos_mask = target > 0.5
        num_pos = pos_mask.sum().item()
        
        if num_pos == 0:
            # If no positives, just backprop the worst N negatives
            num_neg = min(100, loss.shape[0]) 
            loss, _ = torch.topk(loss, num_neg)
            return loss.mean()
            
        num_neg = min(int(num_pos * self.pos_ratio), loss.shape[0] - num_pos)
        
        pos_loss = loss[pos_mask]
        neg_loss = loss[~pos_mask]
        
        if num_neg > 0 and len(neg_loss) > 0:
            neg_loss, _ = torch.topk(neg_loss, min(num_neg, len(neg_loss)))
            return (pos_loss.sum() + neg_loss.sum()) / (num_pos + num_neg)
        else:
            return pos_loss.mean()

class MicroGhostThermalLoss(nn.Module):
    """
    Combined loss for thermal intrusion detection.

    Components:
    1. BBox regression: CIoU (optimizes overlap directly)
    2. Objectness: BCE with logits
    3. Classification: Cross-entropy (binary intrusion)

    Loss = bbox_weight * L_bbox + obj_weight * L_obj + class_weight * L_class
    """

    def __init__(self, bbox_weight=None, obj_weight=None, class_weight=None,
                 anchor_sizes=None, input_size=None):
        super().__init__()
        self.bbox_weight = bbox_weight or BBOX_WEIGHT
        self.obj_weight = obj_weight or OBJ_WEIGHT
        self.class_weight = class_weight or CLASS_WEIGHT

        self.small_grid_w = INPUT_WIDTH // 8
        self.small_grid_h = INPUT_HEIGHT // 8
        self.large_grid_w = INPUT_WIDTH // 16
        self.large_grid_h = INPUT_HEIGHT // 16

        self.bbox_loss = CIoUBBoxLoss(anchor_sizes=anchor_sizes)
        self.obj_loss = OHEMBCEWithLogitsLoss(pos_ratio=3.0)
        self.cls_loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict from model.forward()
            targets: dict with grid-encoded targets

        Returns:
            dict with total, bbox, obj, cls losses
        """
        # BBox CIoU loss (both scales)
        loss_bbox_s = self.bbox_loss(
            predictions['bbox_small'], targets['bbox_small'],
            targets['obj_small'], self.small_grid_w, self.small_grid_h,
        )
        loss_bbox_l = self.bbox_loss(
            predictions['bbox_large'], targets['bbox_large'],
            targets['obj_large'], self.large_grid_w, self.large_grid_h,
        )

        # Objectness BCE loss
        loss_obj_s = self.obj_loss(predictions['obj_small'],
                                   targets['obj_small'])
        loss_obj_l = self.obj_loss(predictions['obj_large'],
                                   targets['obj_large'])

        # Classification loss
        loss_cls = self.cls_loss(predictions['label'], targets['label'])

        # Weighted sum
        total_bbox = self.bbox_weight * (loss_bbox_s + loss_bbox_l)
        total_obj = self.obj_weight * (loss_obj_s + loss_obj_l)
        total_cls = self.class_weight * loss_cls
        total = total_bbox + total_obj + total_cls

        return {
            'total': total,
            'bbox': total_bbox,
            'obj': total_obj,
            'cls': total_cls,
        }


# ============================================================================
# 3. METRICS TRACKER
# ============================================================================

class MetricsTracker:
    """
    Tracks detection-focused metrics per epoch.

    Metrics:
    - Losses: total, bbox, obj, cls
    - Detection: mean IoU on positive grid cells, objectness recall@threshold
    - Classification: binary intrusion F1 (background vs any intrusion)
    """

    def __init__(self, num_classes=None, anchor_sizes=None, obj_threshold=None):
        self.num_classes = num_classes or NUM_CLASSES
        self.obj_threshold = obj_threshold or OBJ_METRIC_THRESHOLD
        self.small_grid_w = INPUT_WIDTH // 8
        self.small_grid_h = INPUT_HEIGHT // 8
        self.large_grid_w = INPUT_WIDTH // 16
        self.large_grid_h = INPUT_HEIGHT // 16

        if anchor_sizes is not None:
            self.anchor_sizes = torch.tensor(anchor_sizes, dtype=torch.float32)
        else:
            self.anchor_sizes = torch.tensor(DEFAULT_ANCHOR_SIZES,
                                             dtype=torch.float32)

        self.reset()

    def reset(self):
        """Reset all metrics for new epoch."""
        self.losses = defaultdict(list)
        self.cls_preds = []
        self.cls_targets = []
        self.bbox_ious = []
        self.obj_pos_total = 0
        self.obj_pos_detected = 0

    def _update_detection_metrics(self, predictions, targets):
        """IoU and objectness recall on positive grid cells only."""
        scales = [
            ('bbox_small', 'obj_small', self.small_grid_w, self.small_grid_h),
            ('bbox_large', 'obj_large', self.large_grid_w, self.large_grid_h),
        ]

        with torch.no_grad():
            for key_bbox, key_obj, grid_w, grid_h in scales:
                pred_bbox = predictions[key_bbox]
                pred_obj = predictions[key_obj]
                tgt_bbox = targets[key_bbox]
                tgt_obj = targets[key_obj]

                for b in range(pred_bbox.shape[0]):
                    for a, anchor_size in enumerate(self.anchor_sizes):
                        mask = tgt_obj[b, a] > 0.5
                        if not mask.any():
                            continue

                        off = a * 4
                        pred_cells = pred_bbox[b, off:off + 4]
                        tgt_cells = tgt_bbox[b, off:off + 4]

                        pred_boxes = _decode_grid_boxes(
                            pred_cells, mask, grid_w, grid_h, anchor_size.item()
                        )
                        tgt_boxes = _decode_grid_boxes(
                            tgt_cells, mask, grid_w, grid_h, anchor_size.item()
                        )
                        if pred_boxes.shape[0] == 0:
                            continue

                        self.bbox_ious.extend(_iou_xyxy(pred_boxes, tgt_boxes))

                        obj_scores = torch.sigmoid(pred_obj[b, a][mask])
                        self.obj_pos_total += mask.sum().item()
                        self.obj_pos_detected += (
                            obj_scores > self.obj_threshold
                        ).sum().item()

    def update(self, predictions, targets, losses):
        """Update metrics with one batch of results."""
        for k, v in losses.items():
            self.losses[k].append(v.item())

        pred_cls = predictions['label'].argmax(dim=1).cpu().numpy()
        true_cls = targets['label'].cpu().numpy()
        self.cls_preds.extend(pred_cls)
        self.cls_targets.extend(true_cls)

        self._update_detection_metrics(predictions, targets)

    def compute(self):
        """Compute final metrics for epoch."""
        metrics = {}

        for k, v in self.losses.items():
            metrics[f'loss_{k}'] = np.mean(v)

        metrics['bbox_iou'] = float(np.mean(self.bbox_ious)) if self.bbox_ious else 0.0
        metrics['obj_recall'] = (
            self.obj_pos_detected / max(1, self.obj_pos_total)
        )

        preds_bin = _to_intrusion_binary(self.cls_preds)
        trues_bin = _to_intrusion_binary(self.cls_targets)

        if len(preds_bin) > 0:
            from sklearn.metrics import precision_recall_fscore_support
            metrics['intrusion_accuracy'] = (preds_bin == trues_bin).mean()
            prec, rec, f1, _ = precision_recall_fscore_support(
                trues_bin, preds_bin, average='binary', zero_division=0
            )
            metrics['intrusion_precision'] = prec
            metrics['intrusion_recall'] = rec
            metrics['intrusion_f1'] = f1

        return metrics

    def get_confusion_matrix(self):
        """Return binary intrusion confusion matrix."""
        from sklearn.metrics import confusion_matrix
        preds_bin = _to_intrusion_binary(self.cls_preds)
        trues_bin = _to_intrusion_binary(self.cls_targets)
        return confusion_matrix(trues_bin, preds_bin, labels=[0, 1])

    def get_classification_report(self):
        """Return binary intrusion classification report."""
        from sklearn.metrics import classification_report
        preds_bin = _to_intrusion_binary(self.cls_preds)
        trues_bin = _to_intrusion_binary(self.cls_targets)
        return classification_report(
            trues_bin, preds_bin,
            target_names=['Background', 'Intrusion'],
            zero_division=0,
        )


def print_kaggle_system_stats():
    """Prints live CPU RAM and GPU VRAM usage."""
    ram_percent = psutil.virtual_memory().percent
    ram_used_gb = psutil.virtual_memory().used / (1024**3)
    
    gpu_stats = ""
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated() / (1024**2)
        vram_reserved = torch.cuda.memory_reserved() / (1024**2)
        gpu_stats = f" | GPU VRAM Allocated: {vram_allocated:.1f} MB | Reserved: {vram_reserved:.1f} MB"
        
    print(f"💻 System Check -> CPU RAM: {ram_used_gb:.1f} GB ({ram_percent}%){gpu_stats}")


# ============================================================================
# 4. TRAINER
# ============================================================================

class Trainer:
    """
    Complete training pipeline for MicroGhost-Thermal.

    Features:
    - Cosine annealing with warm restarts LR schedule
    - Early stopping with patience
    - Best model checkpointing (by lowest validation loss)
    - Gradient clipping
    - Comprehensive metric tracking
    """

    def __init__(self, model, train_loader, val_loader, device=None,
                 lr=None, weight_decay=None, epochs=None, patience=None,
                 input_size=None, anchor_sizes=None):
        self.device = device or DEVICE
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs or EPOCHS
        self.patience = patience or PATIENCE
        self.input_size = input_size or INPUT_SIZE
        self.anchor_sizes = anchor_sizes

        # Loss function
        self.criterion = MicroGhostThermalLoss(
            anchor_sizes=anchor_sizes,
            input_size=self.input_size,
        )

        # Optimizer
        lr = lr or LEARNING_RATE
        wd = weight_decay or WEIGHT_DECAY
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                            weight_decay=wd)

        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=LR_T0, T_mult=LR_T_MULT, eta_min=LR_MIN,
        )

        # Metrics
        self.train_metrics = MetricsTracker(anchor_sizes=anchor_sizes)
        self.val_metrics = MetricsTracker(anchor_sizes=anchor_sizes)

        # Best model tracking (checkpoint saved on lowest val loss)
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.no_improve_count = 0

        # History — detection-focused metrics
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_bbox_loss': [], 'val_bbox_loss': [],
            'train_bbox_iou': [], 'val_bbox_iou': [],
            'train_obj_recall': [], 'val_obj_recall': [],
            'train_intrusion_f1': [], 'val_intrusion_f1': [],
            'lr': [],
        }

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()

        pbar = tqdm(self.train_loader, desc='  Training', leave=False)
        for images, targets in pbar:
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}

            self.optimizer.zero_grad()
            predictions = self.model(images)
            losses = self.criterion(predictions, targets)

            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                            max_norm=1.0)
            self.optimizer.step()

            self.train_metrics.update(predictions, targets, losses)
            pbar.set_postfix({'loss': f"{losses['total'].item():.3f}"})

        return self.train_metrics.compute()

    @torch.no_grad()
    def validate(self):
        """Validate model."""
        self.model.eval()
        self.val_metrics.reset()

        pbar = tqdm(self.val_loader, desc='  Validation', leave=False)
        for images, targets in pbar:
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}

            predictions = self.model(images)
            losses = self.criterion(predictions, targets)
            self.val_metrics.update(predictions, targets, losses)
            pbar.set_postfix({'loss': f"{losses['total'].item():.3f}"})

        return self.val_metrics.compute()

    def fit(self, save_path=None):
        """Full training loop."""
        save_path = save_path or BEST_MODEL_PATH
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

        print("=" * 70)
        print("  STARTING TRAINING — MicroGhost-Thermal")
        print("=" * 70)
        print(f"  Device:       {self.device}")
        print(f"  Epochs:       {self.epochs} (patience={self.patience})")
        print(f"  Train:        {len(self.train_loader)} batches")
        print(f"  Validation:   {len(self.val_loader)} batches")
        print("=" * 70)

        for epoch in range(self.epochs):
            t0 = time.time()

            # Train
            train_m = self.train_epoch()

            # Validate
            val_m = self.validate()

            # LR step
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']

            # Record history
            self.history['train_loss'].append(train_m['loss_total'])
            self.history['val_loss'].append(val_m['loss_total'])
            self.history['train_bbox_loss'].append(train_m.get('loss_bbox', 0))
            self.history['val_bbox_loss'].append(val_m.get('loss_bbox', 0))
            self.history['train_bbox_iou'].append(train_m.get('bbox_iou', 0))
            self.history['val_bbox_iou'].append(val_m.get('bbox_iou', 0))
            self.history['train_obj_recall'].append(train_m.get('obj_recall', 0))
            self.history['val_obj_recall'].append(val_m.get('obj_recall', 0))
            self.history['train_intrusion_f1'].append(
                train_m.get('intrusion_f1', 0))
            self.history['val_intrusion_f1'].append(
                val_m.get('intrusion_f1', 0))
            self.history['lr'].append(lr)

            # Save best checkpoint on val loss; always keep latest epoch weights
            improved = False
            val_loss = val_m['loss_total']

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                improved = True
                self._save_checkpoint(save_path, epoch, val_m)

            last_path = LAST_MODEL_PATH
            self._save_checkpoint(last_path, epoch, val_m)

            self.no_improve_count = 0 if improved else \
                self.no_improve_count + 1

            print_kaggle_system_stats()

            # Print progress
            elapsed = time.time() - t0
            marker = ' *BEST*' if improved else ''
            print(
                f"  Epoch {epoch + 1:3d}/{self.epochs} | "
                f"{elapsed:.1f}s | "
                f"LR: {lr:.1e} | "
                f"Loss: {train_m['loss_total']:.4f}/{val_m['loss_total']:.4f} | "
                f"IoU: {val_m.get('bbox_iou', 0):.3f} | "
                f"ObjRec: {val_m.get('obj_recall', 0) * 100:.1f}% | "
                f"IntrF1: {val_m.get('intrusion_f1', 0) * 100:.1f}%{marker}"
            )

            # Early stopping
            if self.no_improve_count >= self.patience:
                print(f"\n  ⏹ Early stopping — no improvement for "
                      f"{self.patience} epochs.")
                break

        print("\n" + "=" * 70)
        print("  TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Best Epoch:   {self.best_epoch + 1}")
        print(f"  Best Val Loss:{self.best_val_loss:.4f}")
        print(f"  Best saved:   {save_path}")
        print(f"  Last saved:   {LAST_MODEL_PATH}")
        print("=" * 70)

        return self.history

    def _save_checkpoint(self, path, epoch, metrics):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': {
                'num_classes': self.model.num_classes,
                'input_size': self.model.input_size,
                'classifier_hidden_dim': self.model.classifier_hidden_dim,
            },
        }, path)


# ============================================================================
# 5. VISUALIZATION
# ============================================================================

def plot_training_history(history, save_path='training_history.png'):
    """Plot comprehensive training curves."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax.plot(epochs, history['val_loss'], 'r-', label='Val')
    ax.set_title('Total Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # BBox loss
    ax = axes[0, 1]
    ax.plot(epochs, history['train_bbox_loss'], 'b-', label='Train')
    ax.plot(epochs, history['val_bbox_loss'], 'r-', label='Val')
    ax.set_title('BBox Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # BBox IoU on positive cells
    ax = axes[0, 2]
    ax.plot(epochs, history['train_bbox_iou'], 'b-', label='Train')
    ax.plot(epochs, history['val_bbox_iou'], 'r-', label='Val')
    ax.set_title('BBox IoU (positive cells)')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Objectness recall on positive cells
    ax = axes[1, 0]
    ax.plot(epochs, [x * 100 for x in history['train_obj_recall']], 'b-',
            label='Train')
    ax.plot(epochs, [x * 100 for x in history['val_obj_recall']], 'r-',
            label='Val')
    ax.set_title('Objectness Recall @ threshold (%)')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Binary intrusion F1
    ax = axes[1, 1]
    ax.plot(epochs, [x * 100 for x in history['train_intrusion_f1']], 'b-',
            label='Train')
    ax.plot(epochs, [x * 100 for x in history['val_intrusion_f1']], 'r-',
            label='Val')
    ax.set_title('Binary Intrusion F1 (%)')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 2]
    ax.plot(epochs, history['lr'], 'g-')
    ax.set_title('Learning Rate')
    ax.set_xlabel('Epoch')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.suptitle('MicroGhost-Thermal Training History', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved training history to {save_path}")


def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """Plot binary confusion matrix."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    class_names = ['Background', 'Intrusion']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Intrusion Detection — Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved confusion matrix to {save_path}")


if __name__ == '__main__':
    print("Training Module — Self Test")
    print("-" * 40)

    # Quick loss test
    from model import MicroGhostThermal

    model = MicroGhostThermal()
    loss_fn = MicroGhostThermalLoss()

    small_grid_w = INPUT_WIDTH // 8
    small_grid_h = INPUT_HEIGHT // 8
    large_grid_w = INPUT_WIDTH // 16
    large_grid_h = INPUT_HEIGHT // 16
    B = 4

    dummy_preds = {
        'bbox_small': torch.randn(B, NUM_ANCHORS * 4, small_grid_h, small_grid_w),
        'obj_small': torch.randn(B, NUM_ANCHORS, small_grid_h, small_grid_w),
        'bbox_large': torch.randn(B, NUM_ANCHORS * 4, large_grid_h, large_grid_w),
        'obj_large': torch.randn(B, NUM_ANCHORS, large_grid_h, large_grid_w),
        'label': torch.randn(B, NUM_CLASSES),
    }
    dummy_targets = {
        'bbox_small': torch.randn(B, NUM_ANCHORS * 4, small_grid_h, small_grid_w),
        'obj_small': torch.rand(B, NUM_ANCHORS, small_grid_h, small_grid_w),
        'bbox_large': torch.randn(B, NUM_ANCHORS * 4, large_grid_h, large_grid_w),
        'obj_large': torch.rand(B, NUM_ANCHORS, large_grid_h, large_grid_w),
        'label': torch.randint(0, NUM_CLASSES, (B,)),
    }

    losses = loss_fn(dummy_preds, dummy_targets)
    print("  Loss computation test:")
    for k, v in losses.items():
        print(f"    {k}: {v.item():.4f}")
    print("[OK] Training module test passed!")
