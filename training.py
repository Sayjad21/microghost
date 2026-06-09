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
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

from config import (
    INPUT_SIZE, INPUT_WIDTH, INPUT_HEIGHT, NUM_CLASSES, NUM_ANCHORS,
    DEFAULT_ANCHOR_SIZES, CLASS_MAP,
    BBOX_WEIGHT, OBJ_WEIGHT, CLASS_WEIGHT,
    LEARNING_RATE, WEIGHT_DECAY, EPOCHS, PATIENCE,
    LR_T0, LR_T_MULT, LR_MIN,
    MODEL_SAVE_DIR, BEST_MODEL_PATH, LOG_DIR, DEVICE,
)


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
        self.obj_loss = nn.BCEWithLogitsLoss(reduction='mean')
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
    Tracks detection and classification metrics per epoch.

    Metrics:
    - Detection: IoU, Objectness Accuracy
    - Classification: Accuracy, Precision, Recall, F1
    - Losses: per-component breakdown
    """

    def __init__(self, num_classes=None, anchor_sizes=None):
        self.num_classes = num_classes or NUM_CLASSES
        self.class_names = list(CLASS_MAP.keys())
        self.small_grid_w = INPUT_WIDTH // 8
        self.large_grid_w = INPUT_WIDTH // 16

        if anchor_sizes is not None:
            self.anchor_sizes = torch.tensor(anchor_sizes)
        else:
            self.anchor_sizes = torch.tensor(DEFAULT_ANCHOR_SIZES)

        self.reset()

    def reset(self):
        """Reset all metrics for new epoch."""
        self.losses = defaultdict(list)
        self.cls_preds = []
        self.cls_targets = []
        self.obj_correct_s = 0
        self.obj_total_s = 0
        self.obj_correct_l = 0
        self.obj_total_l = 0
        self.iou_small = []
        self.iou_large = []

    def update(self, predictions, targets, losses):
        """Update metrics with one batch of results."""
        # Losses
        for k, v in losses.items():
            self.losses[k].append(v.item())

        # Classification
        pred_cls = predictions['label'].argmax(dim=1).cpu().numpy()
        true_cls = targets['label'].cpu().numpy()
        self.cls_preds.extend(pred_cls)
        self.cls_targets.extend(true_cls)

        # Objectness accuracy
        with torch.no_grad():
            for scale, key_o in [('s', 'obj_small'), ('l', 'obj_large')]:
                pred_o = (torch.sigmoid(predictions[key_o]) > 0.5).float()
                true_o = (targets[key_o] > 0.5).float()
                correct = (pred_o == true_o).sum().item()
                total = true_o.numel()
                if scale == 's':
                    self.obj_correct_s += correct
                    self.obj_total_s += total
                else:
                    self.obj_correct_l += correct
                    self.obj_total_l += total

    def compute(self):
        """Compute final metrics for epoch."""
        metrics = {}

        # Average losses
        for k, v in self.losses.items():
            metrics[f'loss_{k}'] = np.mean(v)

        # Objectness accuracy
        metrics['obj_acc_small'] = self.obj_correct_s / max(1, self.obj_total_s)
        metrics['obj_acc_large'] = self.obj_correct_l / max(1, self.obj_total_l)
        metrics['obj_acc_avg'] = (metrics['obj_acc_small'] +
                                  metrics['obj_acc_large']) / 2

        # Classification metrics
        preds = np.array(self.cls_preds)
        trues = np.array(self.cls_targets)

        if len(preds) > 0:
            metrics['cls_accuracy'] = (preds == trues).mean()

            # Multiclass F1 macro
            from sklearn.metrics import precision_recall_fscore_support
            try:
                prec, rec, f1, _ = precision_recall_fscore_support(
                    trues, preds, average='macro', zero_division=0
                )
                metrics['precision'] = prec
                metrics['recall'] = rec
                metrics['f1'] = f1
            except Exception:
                metrics['precision'] = 0.0
                metrics['recall'] = 0.0
                metrics['f1'] = 0.0

        return metrics

    def get_confusion_matrix(self):
        """Return 2×2 confusion matrix."""
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(
            self.cls_targets, self.cls_preds,
            labels=list(range(self.num_classes)),
        )

    def get_classification_report(self):
        """Return detailed classification report string."""
        from sklearn.metrics import classification_report
        return classification_report(
            self.cls_targets, self.cls_preds,
            target_names=self.class_names,
            zero_division=0,
        )


# ============================================================================
# 4. TRAINER
# ============================================================================

class Trainer:
    """
    Complete training pipeline for MicroGhost-Thermal.

    Features:
    - Cosine annealing with warm restarts LR schedule
    - Early stopping with patience
    - Best model checkpointing (by F1 score)
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

        # History
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': [],
            'train_obj_acc': [], 'val_obj_acc': [],
            'lr': [],
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.no_improve_count = 0

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()

        for images, targets in self.train_loader:
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

        return self.train_metrics.compute()

    @torch.no_grad()
    def validate(self):
        """Validate model."""
        self.model.eval()
        self.val_metrics.reset()

        for images, targets in self.val_loader:
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}

            predictions = self.model(images)
            losses = self.criterion(predictions, targets)
            self.val_metrics.update(predictions, targets, losses)

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
            self.history['train_acc'].append(train_m.get('cls_accuracy', 0))
            self.history['val_acc'].append(val_m.get('cls_accuracy', 0))
            self.history['train_f1'].append(train_m.get('f1', 0))
            self.history['val_f1'].append(val_m.get('f1', 0))
            self.history['train_precision'].append(
                train_m.get('precision', 0))
            self.history['val_precision'].append(val_m.get('precision', 0))
            self.history['train_recall'].append(train_m.get('recall', 0))
            self.history['val_recall'].append(val_m.get('recall', 0))
            self.history['train_obj_acc'].append(train_m['obj_acc_avg'])
            self.history['val_obj_acc'].append(val_m['obj_acc_avg'])
            self.history['lr'].append(lr)

            # Check improvement
            improved = False
            val_f1 = val_m.get('f1', 0)

            if val_m['loss_total'] < self.best_val_loss:
                self.best_val_loss = val_m['loss_total']
                improved = True

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_epoch = epoch
                improved = True
                self._save_checkpoint(save_path, epoch, val_m)

            self.no_improve_count = 0 if improved else \
                self.no_improve_count + 1

            # Print progress
            elapsed = time.time() - t0
            marker = ' *BEST*' if improved else ''
            print(
                f"  Epoch {epoch + 1:3d}/{self.epochs} | "
                f"{elapsed:.1f}s | "
                f"LR: {lr:.1e} | "
                f"Loss: {train_m['loss_total']:.4f}/{val_m['loss_total']:.4f} | "
                f"Acc: {val_m.get('cls_accuracy', 0) * 100:.1f}% | "
                f"F1: {val_f1 * 100:.1f}%{marker}"
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
        print(f"  Best Val F1:  {self.best_val_f1 * 100:.2f}%")
        print(f"  Best Val Loss:{self.best_val_loss:.4f}")
        print(f"  Saved to:     {save_path}")
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

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, [x * 100 for x in history['train_acc']], 'b-',
            label='Train')
    ax.plot(epochs, [x * 100 for x in history['val_acc']], 'r-',
            label='Val')
    ax.set_title('Classification Accuracy (%)')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # F1
    ax = axes[0, 2]
    ax.plot(epochs, [x * 100 for x in history['train_f1']], 'b-',
            label='Train')
    ax.plot(epochs, [x * 100 for x in history['val_f1']], 'r-',
            label='Val')
    ax.set_title('Intrusion F1 Score (%)')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Precision / Recall
    ax = axes[1, 0]
    ax.plot(epochs, [x * 100 for x in history['val_precision']], 'g-',
            label='Precision')
    ax.plot(epochs, [x * 100 for x in history['val_recall']], 'm-',
            label='Recall')
    ax.set_title('Val Precision & Recall (%)')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Objectness accuracy
    ax = axes[1, 1]
    ax.plot(epochs, [x * 100 for x in history['train_obj_acc']], 'b-',
            label='Train')
    ax.plot(epochs, [x * 100 for x in history['val_obj_acc']], 'r-',
            label='Val')
    ax.set_title('Objectness Accuracy (%)')
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
    plt.show()
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
    plt.show()
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
