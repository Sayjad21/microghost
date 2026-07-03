"""
MicroGhost-Thermal: Training Module (V2)
==========================================
Complete training pipeline with V2 loss functions and phase-based training.

V2 Additions:
- ContrastLoss: TFDet-style foreground/background feature contrast
- GateRegularizer: Entropy regularizer for EnergyGate weights
- MicroGhostV2Loss: Combined loss with all V2 terms
- PhaseTrainer: Phase-aware training with CMM scheduling and layer freezing
"""

import os
import math
import time
import random
import psutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm

from config import (
    INPUT_SIZE, INPUT_WIDTH, INPUT_HEIGHT, NUM_CLASSES, NUM_ANCHORS,
    DEFAULT_ANCHOR_SIZES, CLASS_MAP, OBJ_METRIC_THRESHOLD,
    BBOX_WEIGHT, OBJ_WEIGHT, CLASS_WEIGHT, CONTRAST_WEIGHT, GATE_REG_WEIGHT,
    LEARNING_RATE, WEIGHT_DECAY, EPOCHS, PATIENCE,
    LR_T0, LR_T_MULT, LR_MIN,
    MODEL_SAVE_DIR, BEST_MODEL_PATH, LAST_MODEL_PATH, LOG_DIR, DEVICE,
    LOG_CLAMP_MIN, LOG_CLAMP_MAX, TRAINING_PHASES,
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
    w = anchor_size * torch.exp(
        bbox_tensor[2][pos[:, 0], pos[:, 1]].clamp(LOG_CLAMP_MIN, LOG_CLAMP_MAX))
    h = anchor_size * torch.exp(
        bbox_tensor[3][pos[:, 0], pos[:, 1]].clamp(LOG_CLAMP_MIN, LOG_CLAMP_MAX))
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
    Complete IoU Loss.
    CIoU = IoU - (rho^2(b, b_gt) / c^2) - alpha*v
    Args: (N, 4) [cx, cy, w, h] tensors
    """
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    tgt_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    tgt_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    tgt_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    tgt_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

    inter_x1 = torch.max(pred_x1, tgt_x1)
    inter_y1 = torch.max(pred_y1, tgt_y1)
    inter_x2 = torch.min(pred_x2, tgt_x2)
    inter_y2 = torch.min(pred_y2, tgt_y2)

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * \
                 (inter_y2 - inter_y1).clamp(min=0)

    pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
    tgt_area = target_boxes[:, 2] * target_boxes[:, 3]
    union_area = pred_area + tgt_area - inter_area + eps

    iou = inter_area / union_area

    enc_x1 = torch.min(pred_x1, tgt_x1)
    enc_y1 = torch.min(pred_y1, tgt_y1)
    enc_x2 = torch.max(pred_x2, tgt_x2)
    enc_y2 = torch.max(pred_y2, tgt_y2)
    c_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    center_dist_sq = (pred_boxes[:, 0] - target_boxes[:, 0]) ** 2 + \
                     (pred_boxes[:, 1] - target_boxes[:, 1]) ** 2

    pred_ratio = torch.atan(pred_boxes[:, 2] / (pred_boxes[:, 3] + eps))
    tgt_ratio = torch.atan(target_boxes[:, 2] / (target_boxes[:, 3] + eps))
    v = (4 / (math.pi ** 2)) * (pred_ratio - tgt_ratio) ** 2

    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou_val = iou - (center_dist_sq / c_sq) - (alpha * v)
    return (1 - ciou_val).mean()


class CIoUBBoxLoss(nn.Module):
    """CIoU bounding box loss for grid-based detection."""

    def __init__(self, anchor_sizes=None):
        super().__init__()
        if anchor_sizes is not None:
            self.anchor_sizes = torch.tensor(anchor_sizes, dtype=torch.float32)
        else:
            self.anchor_sizes = torch.tensor(DEFAULT_ANCHOR_SIZES,
                                             dtype=torch.float32)

    def forward(self, pred_bbox, target_bbox, obj_mask, grid_w, grid_h):
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

                abs_pred = torch.stack([
                    (gx + pred_cx) / grid_w,
                    (gy + pred_cy) / grid_h,
                    anchor_size * torch.exp(pred_lw.clamp(LOG_CLAMP_MIN, LOG_CLAMP_MAX)),
                    anchor_size * torch.exp(pred_lh.clamp(LOG_CLAMP_MIN, LOG_CLAMP_MAX)),
                ], dim=1)

                abs_tgt = torch.stack([
                    (gx + tgt_cx) / grid_w,
                    (gy + tgt_cy) / grid_h,
                    anchor_size * torch.exp(tgt_lw.clamp(LOG_CLAMP_MIN, LOG_CLAMP_MAX)),
                    anchor_size * torch.exp(tgt_lh.clamp(LOG_CLAMP_MIN, LOG_CLAMP_MAX)),
                ], dim=1)

                loss = ciou_loss(abs_pred, abs_tgt)
                total_loss += loss * mask.sum()
                num_positive += mask.sum()

        if num_positive > 0:
            return total_loss / num_positive
        return torch.tensor(0.0, device=device, requires_grad=True)


# ============================================================================
# 2. OBJECTNESS LOSS (OHEM)
# ============================================================================

class OHEMBCEWithLogitsLoss(nn.Module):
    """Online Hard Example Mining for objectness BCE loss."""
    def __init__(self, pos_ratio=3.0):
        super().__init__()
        self.pos_ratio = pos_ratio

    def forward(self, pred, target):
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        loss = loss.view(-1)
        target = target.view(-1)

        pos_mask = target > 0.5
        num_pos = pos_mask.sum().item()

        if num_pos == 0:
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


# ============================================================================
# 3. V2 AUXILIARY LOSSES
# ============================================================================

def contrast_loss(feat_p2, obj_target_small, lambda_c=None):
    """
    TFDet-style contrast loss.

    Encourages separation between foreground features (where obj_target > 0)
    and background features. Uses objectness targets as proxy GT mask.

    Args:
        feat_p2: (B, C, H, W) features from BiFusion p2 output
        obj_target_small: (B, A, H, W) objectness targets at small scale
        lambda_c: weight multiplier
    """
    lambda_c = lambda_c if lambda_c is not None else CONTRAST_WEIGHT

    # Create binary mask: 1 where any anchor has an object
    gt_mask = (obj_target_small.max(dim=1, keepdim=True)[0] > 0.5).float()  # (B,1,H,W)

    feat_norm = F.normalize(feat_p2, dim=1)
    mask_expanded = gt_mask.expand_as(feat_norm)

    pos_feats = feat_norm[mask_expanded > 0.5]
    neg_feats = feat_norm[mask_expanded < 0.5]

    if pos_feats.numel() == 0 or neg_feats.numel() == 0:
        return torch.tensor(0.0, device=feat_p2.device)

    # Push foreground features up, background down
    return lambda_c * (-pos_feats.mean() + neg_feats.mean())


def gate_regularizer(w_rgb, w_thm, lambda_g=None):
    """
    Entropy regularizer for EnergyGate.

    Prevents gate from collapsing to always-thermal or always-RGB
    by maximizing the entropy of the gate weight distribution.

    Args:
        w_rgb, w_thm: (B, 1, H, W) gate weights, sum to 1 after softmax
        lambda_g: weight multiplier
    """
    lambda_g = lambda_g if lambda_g is not None else GATE_REG_WEIGHT

    # Entropy: -sum(p * log(p))
    entropy = -(w_rgb * w_rgb.log().clamp(min=-10) +
                w_thm * w_thm.log().clamp(min=-10))
    # Negative sign: we want to MAXIMIZE entropy (minimize negative entropy)
    return -lambda_g * entropy.mean()


# ============================================================================
# 4. COMBINED LOSSES
# ============================================================================

class MicroGhostThermalLoss(nn.Module):
    """V1 combined loss (backward compat)."""

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
        loss_bbox_s = self.bbox_loss(
            predictions['bbox_small'], targets['bbox_small'],
            targets['obj_small'], self.small_grid_w, self.small_grid_h,
        )
        loss_bbox_l = self.bbox_loss(
            predictions['bbox_large'], targets['bbox_large'],
            targets['obj_large'], self.large_grid_w, self.large_grid_h,
        )
        loss_obj_s = self.obj_loss(predictions['obj_small'], targets['obj_small'])
        loss_obj_l = self.obj_loss(predictions['obj_large'], targets['obj_large'])
        loss_cls = self.cls_loss(predictions['label'], targets['label'])

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


class MicroGhostV2Loss(nn.Module):
    """
    V2 combined loss with contrast loss and gate regularizer.

    L_total = BBOX_W * L_bbox + OBJ_W * L_obj + CLS_W * L_cls
              + CONTRAST_W * L_contrast (Phase 2+)
              + GATE_REG_W * L_gate_reg (Phase 3+, negative = maximize entropy)
    """

    def __init__(self, bbox_weight=None, obj_weight=None, class_weight=None,
                 anchor_sizes=None, use_contrast=False, use_gate_reg=False):
        super().__init__()
        self.bbox_weight = bbox_weight or BBOX_WEIGHT
        self.obj_weight = obj_weight or OBJ_WEIGHT
        self.class_weight = class_weight or CLASS_WEIGHT
        self.use_contrast = use_contrast
        self.use_gate_reg = use_gate_reg

        self.small_grid_w = INPUT_WIDTH // 8
        self.small_grid_h = INPUT_HEIGHT // 8
        self.large_grid_w = INPUT_WIDTH // 16
        self.large_grid_h = INPUT_HEIGHT // 16

        self.bbox_loss = CIoUBBoxLoss(anchor_sizes=anchor_sizes)
        self.obj_loss = OHEMBCEWithLogitsLoss(pos_ratio=3.0)
        self.cls_loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, predictions, targets):
        # Standard losses
        loss_bbox_s = self.bbox_loss(
            predictions['bbox_small'], targets['bbox_small'],
            targets['obj_small'], self.small_grid_w, self.small_grid_h,
        )
        loss_bbox_l = self.bbox_loss(
            predictions['bbox_large'], targets['bbox_large'],
            targets['obj_large'], self.large_grid_w, self.large_grid_h,
        )
        loss_obj_s = self.obj_loss(predictions['obj_small'], targets['obj_small'])
        loss_obj_l = self.obj_loss(predictions['obj_large'], targets['obj_large'])
        loss_cls = self.cls_loss(predictions['label'], targets['label'])

        total_bbox = self.bbox_weight * (loss_bbox_s + loss_bbox_l)
        total_obj = self.obj_weight * (loss_obj_s + loss_obj_l)
        total_cls = self.class_weight * loss_cls
        total = total_bbox + total_obj + total_cls

        result = {
            'total': total,
            'bbox': total_bbox,
            'obj': total_obj,
            'cls': total_cls,
        }

        # V2: Contrast loss (uses aux_seg head output or p2 features)
        if self.use_contrast and 'aux_seg' in predictions:
            l_contrast = contrast_loss(
                predictions['aux_seg'],  # (B, 1, H, W)
                targets['obj_small'],
            )
            result['contrast'] = l_contrast
            result['total'] = result['total'] + l_contrast

        # V2: Gate regularizer
        if self.use_gate_reg and 'w_rgb' in predictions:
            l_gate = gate_regularizer(
                predictions['w_rgb'],
                predictions['w_thm'],
            )
            result['gate_reg'] = l_gate
            result['total'] = result['total'] + l_gate

        return result


# ============================================================================
# 5. METRICS TRACKER
# ============================================================================

class MetricsTracker:
    """Tracks detection-focused metrics per epoch."""

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
        self.losses = defaultdict(list)
        self.cls_preds = []
        self.cls_targets = []
        self.bbox_ious = []
        self.obj_pos_total = 0
        self.obj_pos_detected = 0
        # V2 gate tracking
        self.gate_rgb_means = []
        self.gate_thm_means = []

    def _update_detection_metrics(self, predictions, targets):
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
        for k, v in losses.items():
            self.losses[k].append(v.item() if torch.is_tensor(v) else v)

        pred_cls = predictions['label'].argmax(dim=1).cpu().numpy()
        true_cls = targets['label'].cpu().numpy()
        self.cls_preds.extend(pred_cls)
        self.cls_targets.extend(true_cls)

        self._update_detection_metrics(predictions, targets)

        # Track gate weights if available (V2)
        if 'w_rgb' in predictions:
            self.gate_rgb_means.append(
                predictions['w_rgb'].mean().item())
            self.gate_thm_means.append(
                predictions['w_thm'].mean().item())

    def compute(self):
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

        # V2 gate balance
        if self.gate_rgb_means:
            metrics['gate_rgb_mean'] = np.mean(self.gate_rgb_means)
            metrics['gate_thm_mean'] = np.mean(self.gate_thm_means)

        return metrics

    def get_confusion_matrix(self):
        from sklearn.metrics import confusion_matrix
        preds_bin = _to_intrusion_binary(self.cls_preds)
        trues_bin = _to_intrusion_binary(self.cls_targets)
        return confusion_matrix(trues_bin, preds_bin, labels=[0, 1])

    def get_classification_report(self):
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
        gpu_stats = (f" | GPU VRAM Allocated: {vram_allocated:.1f} MB"
                     f" | Reserved: {vram_reserved:.1f} MB")

    print(f"System -> CPU RAM: {ram_used_gb:.1f} GB ({ram_percent}%){gpu_stats}")


# ============================================================================
# 6. TRAINER (V1 backward compat)
# ============================================================================

class Trainer:
    """V1 training pipeline (kept for backward compatibility)."""

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

        self.criterion = MicroGhostThermalLoss(
            anchor_sizes=anchor_sizes,
            input_size=self.input_size,
        )

        lr = lr or LEARNING_RATE
        wd = weight_decay or WEIGHT_DECAY
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                            weight_decay=wd)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=LR_T0, T_mult=LR_T_MULT, eta_min=LR_MIN,
        )

        self.train_metrics = MetricsTracker(anchor_sizes=anchor_sizes)
        self.val_metrics = MetricsTracker(anchor_sizes=anchor_sizes)

        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.no_improve_count = 0

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_bbox_loss': [], 'val_bbox_loss': [],
            'train_bbox_iou': [], 'val_bbox_iou': [],
            'train_obj_recall': [], 'val_obj_recall': [],
            'train_intrusion_f1': [], 'val_intrusion_f1': [],
            'lr': [],
        }

    def train_epoch(self):
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.train_metrics.update(predictions, targets, losses)
            pbar.set_postfix({'loss': f"{losses['total'].item():.3f}"})

        return self.train_metrics.compute()

    @torch.no_grad()
    def validate(self):
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
        save_path = save_path or BEST_MODEL_PATH
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

        print("=" * 70)
        print("  STARTING TRAINING -- MicroGhost-Thermal")
        print("=" * 70)
        print(f"  Device:       {self.device}")
        print(f"  Epochs:       {self.epochs} (patience={self.patience})")
        print(f"  Train:        {len(self.train_loader)} batches")
        print(f"  Validation:   {len(self.val_loader)} batches")
        print("=" * 70)

        for epoch in range(self.epochs):
            t0 = time.time()
            train_m = self.train_epoch()
            val_m = self.validate()
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_m['loss_total'])
            self.history['val_loss'].append(val_m['loss_total'])
            self.history['train_bbox_loss'].append(train_m.get('loss_bbox', 0))
            self.history['val_bbox_loss'].append(val_m.get('loss_bbox', 0))
            self.history['train_bbox_iou'].append(train_m.get('bbox_iou', 0))
            self.history['val_bbox_iou'].append(val_m.get('bbox_iou', 0))
            self.history['train_obj_recall'].append(train_m.get('obj_recall', 0))
            self.history['val_obj_recall'].append(val_m.get('obj_recall', 0))
            self.history['train_intrusion_f1'].append(train_m.get('intrusion_f1', 0))
            self.history['val_intrusion_f1'].append(val_m.get('intrusion_f1', 0))
            self.history['lr'].append(lr)

            improved = False
            val_loss = val_m['loss_total']

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                improved = True
                self._save_checkpoint(save_path, epoch, val_m)

            self._save_checkpoint(LAST_MODEL_PATH, epoch, val_m)
            self.no_improve_count = 0 if improved else self.no_improve_count + 1

            print_kaggle_system_stats()

            elapsed = time.time() - t0
            marker = ' *BEST*' if improved else ''
            print(
                f"  Epoch {epoch + 1:3d}/{self.epochs} | "
                f"{elapsed:.1f}s | LR: {lr:.1e} | "
                f"Loss: {train_m['loss_total']:.4f}/{val_m['loss_total']:.4f} | "
                f"IoU: {val_m.get('bbox_iou', 0):.3f} | "
                f"ObjRec: {val_m.get('obj_recall', 0) * 100:.1f}% | "
                f"IntrF1: {val_m.get('intrusion_f1', 0) * 100:.1f}%{marker}"
            )

            if self.no_improve_count >= self.patience:
                print(f"\n  Early stopping -- no improvement for "
                      f"{self.patience} epochs.")
                break

        print("\n" + "=" * 70)
        print("  TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Best Epoch:   {self.best_epoch + 1}")
        print(f"  Best Val Loss:{self.best_val_loss:.4f}")
        print(f"  Best saved:   {save_path}")
        print("=" * 70)

        return self.history

    def _save_checkpoint(self, path, epoch, metrics):
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
# 7. V2 PHASE TRAINER
# ============================================================================

class PhaseTrainer:
    """
    Phase-aware trainer for MicroGhost-V2.

    Supports the 4-phase training pipeline:
    1. Human Shape Foundation (LLVIP)
    2. Jungle Domain Transfer (ForestPersons + CMM)
    3. Camouflage Fine-Tuning (Camo-M3FD + gate reg)
    4. Full Mix Polish

    Each phase has its own:
    - Learning rate
    - CMM alpha schedule
    - Layer freeze strategy
    - Loss configuration (contrast, gate reg)
    """

    def __init__(self, model, train_loader, val_loader, phase=1,
                 device=None, anchor_sizes=None):
        self.device = device or DEVICE
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.anchor_sizes = anchor_sizes
        self.phase = phase

        # Load phase config
        phase_cfg = TRAINING_PHASES[phase]
        self.phase_name = phase_cfg['name']
        self.epochs = phase_cfg['epochs']
        self.lr = phase_cfg['lr']
        self.cmm_enabled = phase_cfg.get('cmm_enabled', False)
        self.use_contrast = phase_cfg.get('contrast_loss', False)
        self.use_gate_reg = phase_cfg.get('gate_regularizer', False)

        # CMM alpha scheduling
        self.cmm_alpha_start = phase_cfg.get('cmm_alpha_start',
                                              phase_cfg.get('cmm_alpha', 0.0))
        self.cmm_alpha_end = phase_cfg.get('cmm_alpha_end', self.cmm_alpha_start)

        # Apply freeze strategy
        freeze_layers = phase_cfg.get('freeze_layers', [])
        if freeze_layers:
            self.model.freeze_early_layers()
        else:
            if hasattr(self.model, 'unfreeze_all'):
                self.model.unfreeze_all()

        # V2 Loss
        self.criterion = MicroGhostV2Loss(
            anchor_sizes=anchor_sizes,
            use_contrast=self.use_contrast,
            use_gate_reg=self.use_gate_reg,
        )

        # Optimizer (only trainable params)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=self.lr, weight_decay=WEIGHT_DECAY
        )

        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=max(1, self.epochs // 5),
            T_mult=2, eta_min=LR_MIN,
        )

        # Metrics
        self.train_metrics = MetricsTracker(anchor_sizes=anchor_sizes)
        self.val_metrics = MetricsTracker(anchor_sizes=anchor_sizes)

        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.no_improve_count = 0
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_bbox_loss': [], 'val_bbox_loss': [],
            'train_bbox_iou': [], 'val_bbox_iou': [],
            'train_obj_recall': [], 'val_obj_recall': [],
            'train_intrusion_f1': [], 'val_intrusion_f1': [],
            'lr': [],
        }

    def _get_cmm_alpha(self, epoch):
        """Linearly interpolate CMM alpha over training."""
        if not self.cmm_enabled:
            return 0.0
        progress = epoch / max(1, self.epochs - 1)
        return self.cmm_alpha_start + progress * (
            self.cmm_alpha_end - self.cmm_alpha_start)

    def train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        cmm_alpha = self._get_cmm_alpha(epoch)

        pbar = tqdm(self.train_loader,
                    desc=f'  Phase {self.phase} Training', leave=False)
        for images, targets in pbar:
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}

            self.optimizer.zero_grad()
            predictions = self.model(images)
            losses = self.criterion(predictions, targets)

            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.train_metrics.update(predictions, targets, losses)
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.3f}",
                'cmm': f"{cmm_alpha:.2f}",
            })

        return self.train_metrics.compute()

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        self.val_metrics.reset()

        pbar = tqdm(self.val_loader,
                    desc=f'  Phase {self.phase} Validation', leave=False)
        for images, targets in pbar:
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}

            predictions = self.model(images)
            losses = self.criterion(predictions, targets)
            self.val_metrics.update(predictions, targets, losses)

        return self.val_metrics.compute()

    def fit(self, save_path=None, patience=None):
        save_path = save_path or BEST_MODEL_PATH
        patience = patience or PATIENCE
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

        from model import count_parameters
        print("=" * 70)
        print(f"  PHASE {self.phase}: {self.phase_name}")
        print("=" * 70)
        print(f"  Device:       {self.device}")
        print(f"  Epochs:       {self.epochs} (patience={patience})")
        print(f"  LR:           {self.lr}")
        print(f"  CMM:          {'ON' if self.cmm_enabled else 'OFF'}")
        print(f"  Contrast:     {'ON' if self.use_contrast else 'OFF'}")
        print(f"  Gate Reg:     {'ON' if self.use_gate_reg else 'OFF'}")
        print(f"  Trainable:    {count_parameters(self.model):,}")
        print(f"  Train:        {len(self.train_loader)} batches")
        print(f"  Validation:   {len(self.val_loader)} batches")
        print("=" * 70)

        for epoch in range(self.epochs):
            t0 = time.time()
            train_m = self.train_epoch(epoch)
            val_m = self.validate()
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_m['loss_total'])
            self.history['val_loss'].append(val_m['loss_total'])
            self.history['train_bbox_loss'].append(train_m.get('loss_bbox', 0))
            self.history['val_bbox_loss'].append(val_m.get('loss_bbox', 0))
            self.history['train_bbox_iou'].append(train_m.get('bbox_iou', 0))
            self.history['val_bbox_iou'].append(val_m.get('bbox_iou', 0))
            self.history['train_obj_recall'].append(train_m.get('obj_recall', 0))
            self.history['val_obj_recall'].append(val_m.get('obj_recall', 0))
            self.history['train_intrusion_f1'].append(train_m.get('intrusion_f1', 0))
            self.history['val_intrusion_f1'].append(val_m.get('intrusion_f1', 0))
            self.history['lr'].append(lr)

            improved = False
            val_loss = val_m['loss_total']
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                improved = True
                self._save_checkpoint(save_path, epoch, val_m)

            self._save_checkpoint(LAST_MODEL_PATH, epoch, val_m)
            self.no_improve_count = 0 if improved else self.no_improve_count + 1

            print_kaggle_system_stats()

            elapsed = time.time() - t0
            marker = ' *BEST*' if improved else ''
            gate_info = ''
            if 'gate_rgb_mean' in val_m:
                gate_info = (f" | Gate: R={val_m['gate_rgb_mean']:.2f}"
                             f"/T={val_m['gate_thm_mean']:.2f}")

            print(
                f"  E{epoch + 1:3d}/{self.epochs} | "
                f"{elapsed:.1f}s | LR: {lr:.1e} | "
                f"Loss: {train_m['loss_total']:.4f}/{val_m['loss_total']:.4f} | "
                f"IoU: {val_m.get('bbox_iou', 0):.3f} | "
                f"ObjRec: {val_m.get('obj_recall', 0) * 100:.1f}%"
                f"{gate_info}{marker}"
            )

            if self.no_improve_count >= patience:
                print(f"\n  Early stopping -- no improvement for "
                      f"{patience} epochs.")
                break

        print(f"\n  Phase {self.phase} complete. "
              f"Best epoch: {self.best_epoch + 1}, "
              f"Best val loss: {self.best_val_loss:.4f}")

        return self.history

    def _save_checkpoint(self, path, epoch, metrics):
        torch.save({
            'epoch': epoch,
            'phase': self.phase,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': {
                'num_classes': self.model.num_classes,
                'input_size': self.model.input_size,
                'classifier_hidden_dim': self.model.classifier_hidden_dim,
                'model_version': 'v2',
            },
        }, path)


# ============================================================================
# 8. VISUALIZATION
# ============================================================================

def plot_training_history(history, save_path='training_history.png'):
    """Plot comprehensive training curves."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    epochs = range(1, len(history['train_loss']) + 1)

    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax.plot(epochs, history['val_loss'], 'r-', label='Val')
    ax.set_title('Total Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history['train_bbox_loss'], 'b-', label='Train')
    ax.plot(epochs, history['val_bbox_loss'], 'r-', label='Val')
    ax.set_title('BBox Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(epochs, history['train_bbox_iou'], 'b-', label='Train')
    ax.plot(epochs, history['val_bbox_iou'], 'r-', label='Val')
    ax.set_title('BBox IoU (positive cells)')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, [x * 100 for x in history['train_obj_recall']], 'b-', label='Train')
    ax.plot(epochs, [x * 100 for x in history['val_obj_recall']], 'r-', label='Val')
    ax.set_title('Objectness Recall (%)')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, [x * 100 for x in history['train_intrusion_f1']], 'b-', label='Train')
    ax.plot(epochs, [x * 100 for x in history['val_intrusion_f1']], 'r-', label='Val')
    ax.set_title('Binary Intrusion F1 (%)')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(epochs, history['lr'], 'g-')
    ax.set_title('Learning Rate')
    ax.set_xlabel('Epoch')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.suptitle('MicroGhost Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved training history to {save_path}")


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
    plt.title('Intrusion Detection -- Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix to {save_path}")


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    print("Training Module -- V2 Self Test")
    print("-" * 40)

    from model import MicroGhostV2

    model = MicroGhostV2()

    # Test V2 loss
    loss_fn = MicroGhostV2Loss(use_contrast=True, use_gate_reg=True)

    small_grid_w = INPUT_WIDTH // 8
    small_grid_h = INPUT_HEIGHT // 8
    large_grid_w = INPUT_WIDTH // 16
    large_grid_h = INPUT_HEIGHT // 16
    B = 4

    dummy_input = torch.randn(B, 4, INPUT_HEIGHT, INPUT_WIDTH)
    model.train()
    preds = model(dummy_input)

    dummy_targets = {
        'bbox_small': torch.randn(B, NUM_ANCHORS * 4, small_grid_h, small_grid_w),
        'obj_small': torch.rand(B, NUM_ANCHORS, small_grid_h, small_grid_w),
        'bbox_large': torch.randn(B, NUM_ANCHORS * 4, large_grid_h, large_grid_w),
        'obj_large': torch.rand(B, NUM_ANCHORS, large_grid_h, large_grid_w),
        'label': torch.randint(0, NUM_CLASSES, (B,)),
    }

    losses = loss_fn(preds, dummy_targets)
    print("  V2 Loss computation test:")
    for k, v in losses.items():
        val = v.item() if torch.is_tensor(v) else v
        print(f"    {k}: {val:.4f}")

    # Test metrics tracker
    tracker = MetricsTracker()
    tracker.update(preds, dummy_targets, losses)
    metrics = tracker.compute()
    print(f"\n  Gate balance: RGB={metrics.get('gate_rgb_mean', 0):.3f}, "
          f"THM={metrics.get('gate_thm_mean', 0):.3f}")

    print("[OK] V2 Training module test passed!")
