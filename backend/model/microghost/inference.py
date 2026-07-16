"""
MicroGhost-Thermal: Inference & Deployment Module (V2)
========================================================
Handles edge inference, multi-target detection, NMS, alert generation,
and model export (ONNX, TFLite/INT8) for ESP32-S3 deployment.

V2 Additions:
- Supports loading MicroGhostV2 with BiFusion neck and EnergyGate
- Dual threshold logic (inference vs eval)
- Gate weight extraction during inference
- Graceful backward compatibility for V1 models
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import cv2

from model import MicroGhostThermal, MicroGhostV2, estimate_peak_sram
from preprocessing import ThermalPreprocessor
from config import (
    INPUT_SIZE, INPUT_WIDTH, INPUT_HEIGHT, CLASS_MAP, NUM_CLASSES, NUM_ANCHORS,
    DEFAULT_ANCHOR_SIZES, CONFIDENCE_THRESHOLD, EVAL_CONFIDENCE_THRESHOLD, 
    NMS_IOU_THRESHOLD, MAX_DETECTIONS, DEVICE, ESP32_S3, ALERT_CONFIG, 
    V2_CLASSIFIER_HIDDEN_DIM, CLASSIFIER_HIDDEN_DIM,
    MIN_BOX_WIDTH_NORM, MIN_BOX_HEIGHT_NORM, MIN_BOX_AREA_NORM,
    LOG_CLAMP_MIN, LOG_CLAMP_MAX
)

# Inverse class map for alert generation
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}


def _normalize_legacy_state_dict(state_dict):
    """Map checkpoints saved with the older GhostModule layout to this model."""
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        new_key = new_key.replace(".primary_conv.0.", ".primary_conv.")
        new_key = new_key.replace(".primary_conv.1.", ".primary_bn.")
        normalized[new_key] = value
    return normalized


def _infer_checkpoint_num_anchors(state_dict):
    """Infer anchor count from the detection head tensors."""
    obj_weight = state_dict.get("head_small.obj_head.weight")
    if obj_weight is not None:
        return int(obj_weight.shape[0])

    bbox_weight = state_dict.get("head_small.bbox_head.weight")
    if bbox_weight is not None and bbox_weight.shape[0] % 4 == 0:
        return int(bbox_weight.shape[0] // 4)

    return None


def _checkpoint_is_v2(config, state_dict):
    return (
        config.get("model_version", "v1") == "v2"
        or any(k.startswith(("energy_gate.", "bifusion_neck.", "thm_stem.")) for k in state_dict)
    )


def _match_classifier_to_checkpoint(model, state_dict):
    """Resize the final classifier layer when loading legacy classifier heads."""
    weight = state_dict.get("classifier.classifier.3.weight")
    bias = state_dict.get("classifier.classifier.3.bias")
    if weight is None or bias is None:
        return

    final_layer = model.classifier.classifier[3]
    checkpoint_outputs = int(weight.shape[0])
    if checkpoint_outputs != final_layer.out_features:
        model.classifier.classifier[3] = nn.Linear(final_layer.in_features, checkpoint_outputs)


def _coerce_anchor_list(values, fallback, expected_len):
    if values is None:
        values = fallback
    if torch.is_tensor(values):
        values = values.detach().cpu().tolist()
    values = [float(v) for v in values]
    if len(values) == expected_len:
        return values
    if expected_len == 2 and len(values) >= 2:
        return [values[0], values[-1]]
    if len(fallback) == expected_len:
        return [float(v) for v in fallback]
    return [float(v) for v in fallback[:expected_len]]


# ============================================================================
# 1. MODEL LOADING
# ============================================================================

def load_inference_model(model_path, device=DEVICE, override_num_anchors=None):
    """Load PyTorch model for inference, supporting both V1 and V2 architectures."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    config = checkpoint.get('config', {})
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    state_dict = _normalize_legacy_state_dict(state_dict)
    
    # Determine model version
    is_v2 = _checkpoint_is_v2(config, state_dict)

    checkpoint_anchors = _infer_checkpoint_num_anchors(state_dict)
    anchors_to_use = config.get('num_anchors') or checkpoint_anchors or NUM_ANCHORS
    if override_num_anchors is not None:
        if checkpoint_anchors is not None and checkpoint_anchors != override_num_anchors:
            print(
                f"Warning: requested {override_num_anchors} anchors, "
                f"but checkpoint uses {checkpoint_anchors}; using checkpoint value."
            )
            anchors_to_use = checkpoint_anchors
        else:
            anchors_to_use = override_num_anchors

    if is_v2:
        model = MicroGhostV2(
            num_classes=config.get('num_classes', NUM_CLASSES),
            input_size=config.get('input_size', INPUT_SIZE),
            classifier_hidden_dim=config.get('classifier_hidden_dim', V2_CLASSIFIER_HIDDEN_DIM),
            num_anchors=anchors_to_use,
            training_mode=False
        )
    else:
        model = MicroGhostThermal(
            num_classes=config.get('num_classes', NUM_CLASSES),
            input_size=config.get('input_size', INPUT_SIZE),
            classifier_hidden_dim=config.get('classifier_hidden_dim', CLASSIFIER_HIDDEN_DIM),
            num_anchors=anchors_to_use
        )

    _match_classifier_to_checkpoint(model, state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    anchor_sizes = _coerce_anchor_list(
        config.get('anchor_sizes'), DEFAULT_ANCHOR_SIZES, anchors_to_use
    )
    anchor_ratios = _coerce_anchor_list(
        config.get('anchor_ratios'), [1.0] * anchors_to_use, anchors_to_use
    )

    print(f"Loaded {'MicroGhost-V2' if is_v2 else 'MicroGhost-V1'} successfully.")
    print(f"  Inference anchors: {anchors_to_use} sizes={anchor_sizes}")
    if missing or unexpected:
        print(
            "Compatibility load note: "
            f"{len(missing)} missing/new keys and {len(unexpected)} unused keys."
        )
    return model, {
        'num_anchors': anchors_to_use,
        'anchor_sizes': anchor_sizes,
        'anchor_ratios': anchor_ratios,
        'model_version': 'v2' if is_v2 else 'v1',
    }


# ============================================================================
# 2. INFERENCE ENGINE (MULTI-TARGET + NMS)
# ============================================================================

def calculate_iou_numpy(box1, box2):
    """Calculate IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter_area / (box1_area + box2_area - inter_area)


def nms(detections, iou_threshold=NMS_IOU_THRESHOLD):
    """Non-Maximum Suppression to filter overlapping boxes."""
    if not detections:
        return []

    # Sort by confidence descending
    detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
    keep = []

    for det in detections:
        discard = False
        for kept_det in keep:
            iou = calculate_iou_numpy(det['bbox'], kept_det['bbox'])
            if iou > iou_threshold:
                discard = True
                break
        if not discard:
            keep.append(det)

    return keep[:MAX_DETECTIONS]


def filter_spurious_detections(detections):
    """Remove tiny boxes and edge artifacts."""
    kept = []
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh

        if bw < MIN_BOX_WIDTH_NORM or bh < MIN_BOX_HEIGHT_NORM:
            continue
        if area < MIN_BOX_AREA_NORM:
            continue
        touches_side = x1 <= 0.002 or x2 >= 0.998
        touches_bottom = y2 >= 0.998
        if (touches_side or touches_bottom) and (bw < 0.06 or bh < 0.16):
            continue

        kept.append(det)
    return kept


def decode_predictions(obj_small, bbox_small, obj_large, bbox_large,
                       anchor_sizes=None, is_eval=False, iou_pred=1.0,
                       conf_threshold=None):
    num_anchors = int(obj_small.shape[0])
    anchor_sizes = _coerce_anchor_list(anchor_sizes, DEFAULT_ANCHOR_SIZES, num_anchors)

    if getattr(__import__('config'), 'DEBUG_MODE', False):
        conf_threshold = 0.05
    else:
        conf_threshold = (
            conf_threshold
            if conf_threshold is not None
            else EVAL_CONFIDENCE_THRESHOLD if is_eval else CONFIDENCE_THRESHOLD
        )

    small_grid_w, small_grid_h = INPUT_WIDTH // 8, INPUT_HEIGHT // 8
    large_grid_w, large_grid_h = INPUT_WIDTH // 16, INPUT_HEIGHT // 16

    def _decode_box(pred_box, grid_x, grid_y, grid_w, grid_h, anchor_size):
        cx = (grid_x + pred_box[0]) / grid_w
        cy = (grid_y + pred_box[1]) / grid_h
        w = anchor_size * np.exp(np.clip(pred_box[2], LOG_CLAMP_MIN, LOG_CLAMP_MAX))
        h = anchor_size * np.exp(np.clip(pred_box[3], LOG_CLAMP_MIN, LOG_CLAMP_MAX))
        return [
            max(0.0, min(1.0, cx - w / 2)),
            max(0.0, min(1.0, cy - h / 2)),
            max(0.0, min(1.0, cx + w / 2)),
            max(0.0, min(1.0, cy + h / 2)),
        ]

    def _extract(obj_map, bbox_map, grid_w, grid_h):
        candidates = []
        obj_probs = torch.sigmoid(obj_map).cpu().numpy()
        bbox_data = bbox_map.cpu().numpy()
        for a in range(obj_probs.shape[0]):
            y_idx, x_idx = np.where(obj_probs[a] > conf_threshold)
            for gy, gx in zip(y_idx, x_idx):
                off = a * 4
                bbox = _decode_box(
                    bbox_data[off:off + 4, gy, gx],
                    gx, gy, grid_w, grid_h, anchor_sizes[a],
                )
                candidates.append({
                    'conf': float(obj_probs[a, gy, gx]) * iou_pred,
                    'bbox': bbox,
                })
        return candidates

    candidates = []
    candidates.extend(_extract(obj_small, bbox_small, small_grid_w, small_grid_h))
    candidates.extend(_extract(obj_large, bbox_large, large_grid_w, large_grid_h))
    candidates = filter_spurious_detections(candidates)
    return nms(candidates)


class ThermalInferenceEngine:
    def __init__(self, model_path=None, model=None, device=DEVICE, override_num_anchors=None):
        self.device = device
        self.preprocessor = ThermalPreprocessor()
        
        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()
            self.anchor_sizes = DEFAULT_ANCHOR_SIZES 
        elif model_path is not None:
            self.model, metadata = load_inference_model(model_path, device, override_num_anchors)
            self.anchor_sizes = metadata['anchor_sizes']
        else:
            raise ValueError("Must provide either model_path or model instance.")

    def detect(self, image_rgb, image_thermal, is_eval=False, conf_threshold=None):
        """
        Run full detection pipeline on a dual-modality input.

        Args:
            image_rgb: numpy array (H, W, 3)
            image_thermal: numpy array (H, W)
            is_eval: If true, uses the lower EVAL_CONFIDENCE_THRESHOLD to maximize recall.

        Returns:
            list of dicts: {'conf': float, 'bbox': [x1,y1,x2,y2], 'class': str, 'gate_w': dict}
        """
        h_orig, w_orig = image_thermal.shape[:2]

        # 1. Preprocess
        img_tensor, _ = self.preprocessor.process(
            image_rgb, image_thermal, [], [], img_size=(h_orig, w_orig), augment=False
        )
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # 2. Forward pass
        with torch.no_grad():
            preds = self.model(img_tensor)

        # 3. Classifier predictions
        label_outputs = preds['label'].shape[1]
        if label_outputs == NUM_CLASSES + 1:
            cls_logits = preds['label'][:, :NUM_CLASSES]
            iou_pred = float(torch.sigmoid(preds['label'][:, NUM_CLASSES])[0].item())
        else:
            cls_logits = preds['label']
            iou_pred = 1.0
        cls_probs = torch.softmax(cls_logits, dim=1)[0].cpu().numpy()
        pred_class_idx = int(np.argmax(cls_probs))
        rgb_present = bool(np.any(image_rgb))
        thermal_present = bool(np.any(image_thermal))

        # 4. Extract boxes from both scales with IoU-aware confidence
        final_detections = decode_predictions(
            preds['obj_small'][0], preds['bbox_small'][0],
            preds['obj_large'][0], preds['bbox_large'][0],
            anchor_sizes=self.anchor_sizes,
            is_eval=is_eval,
            iou_pred=iou_pred,
            conf_threshold=conf_threshold,
        )
        
        # Thermal-only forest frames can be under-called by the global classifier.
        fallback_class_idx = CLASS_MAP.get('person_visible')
        if pred_class_idx == CLASS_MAP['background']:
            fallback_ok = (
                thermal_present and not rgb_present and final_detections
                and fallback_class_idx is not None
                and float(cls_probs[fallback_class_idx]) >= 0.15
            )
            if fallback_ok:
                pred_class_idx = fallback_class_idx
            else:
                return []

        detected_class_name = INV_CLASS_MAP.get(pred_class_idx, "unknown")
        class_conf = float(cls_probs[pred_class_idx])
        
        # Extract mean gate weights for diagnostic (V2 only)
        gate_w = None
        if 'w_rgb' in preds:
            gate_w = {
                'rgb': float(preds['w_rgb'][0].mean().item()),
                'thm': float(preds['w_thm'][0].mean().item())
            }

        for det in final_detections:
            det['bbox'] = [float(v) for v in det['bbox']]
            det['class'] = detected_class_name
            det['combined_conf'] = (det['conf'] + class_conf) / 2.0
            det['gate_w'] = gate_w

            # Thermal images here are 8-bit intensity maps, not calibrated Celsius.
            x1, y1, x2, y2 = det['bbox']
            x1_p, y1_p = int(x1 * w_orig), int(y1 * h_orig)
            x2_p, y2_p = int(x2 * w_orig), int(y2 * h_orig)
            x1_p, y1_p = max(0, x1_p), max(0, y1_p)
            x2_p, y2_p = min(w_orig, x2_p), min(h_orig, y2_p)

            if x2_p > x1_p and y2_p > y1_p:
                crop = image_thermal[y1_p:y2_p, x1_p:x2_p]
                max_val = float(np.max(crop))
                det['thermal_peak'] = round(max_val / 255.0, 3)
                det['thermal_score'] = det['thermal_peak']
            else:
                det['thermal_peak'] = 0.0
                det['thermal_score'] = 0.0

        return final_detections

    def filter_by_laplacian(self, lap_image, detections, lap_thresh=80.0,
                            high_conf_bypass=None, thermal_present=True,
                            rgb_present=True):
        """Reject detections whose visual crop is too smooth to look like a real target."""
        if lap_thresh is None:
            return detections

        h_lap, w_lap = lap_image.shape[:2]
        confirmed = []

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            box_w = x2 - x1
            box_h = y2 - y1
            x1_p = max(0, min(w_lap, int(x1 * w_lap)))
            y1_p = max(0, min(h_lap, int(y1 * h_lap)))
            x2_p = max(0, min(w_lap, int(x2 * w_lap)))
            y2_p = max(0, min(h_lap, int(y2 * h_lap)))

            det = dict(det)
            small_thermal_target = thermal_present and not rgb_present and box_w < 0.09 and box_h < 0.14
            weak_small_target = det.get('thermal_peak', 0.0) < 0.93 or det.get('combined_conf', 0.0) < 0.35
            if small_thermal_target and weak_small_target:
                det['lap_filter'] = 'rejected_small_thermal_blob'
                continue

            if x2_p <= x1_p or y2_p <= y1_p:
                det['lap_var'] = 0.0
                continue

            crop = lap_image[y1_p:y2_p, x1_p:x2_p]
            if crop.ndim == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            else:
                gray = crop

            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            det['lap_var'] = round(lap_var, 2)

            if thermal_present and not rgb_present:
                hot_thresh = max(float(np.percentile(gray, 85)), 180.0)
                hot_mask = gray >= hot_thresh
                if np.any(hot_mask):
                    ys, xs = np.where(hot_mask)
                    hot_ratio = float(hot_mask.mean())
                    hot_vspan = float((ys.max() - ys.min() + 1) / max(gray.shape[0], 1))
                    det['thermal_hot_ratio'] = round(hot_ratio, 3)
                    det['thermal_hot_vspan'] = round(hot_vspan, 3)
                    weak_small_hot_target = det.get('thermal_peak', 0.0) < 0.93 or det.get('combined_conf', 0.0) < 0.35
                    if hot_ratio < 0.06 and hot_vspan < 0.30 and weak_small_hot_target:
                        det['lap_filter'] = 'rejected_small_hot_blob'
                        continue

            if crop.ndim == 3:
                mean_intensity = float(gray.mean())
                top_band = y1 <= 0.12 and y2 <= 0.30
                if top_band and mean_intensity < 45.0 and lap_var < 120.0:
                    det['lap_filter'] = 'rejected_top_dark_smooth'
                    continue

            min_thermal_peak = 0.95 if thermal_present and rgb_present else 0.70
            if thermal_present and det.get('thermal_peak', 0.0) < min_thermal_peak and det.get('combined_conf', 0.0) < 0.70:
                det['lap_filter'] = 'rejected_low_thermal_low_conf'
                continue

            if high_conf_bypass is not None and det.get('combined_conf', 0.0) >= high_conf_bypass:
                det['lap_filter'] = 'bypassed_high_conf'
                confirmed.append(det)
            elif lap_var >= lap_thresh:
                det['lap_filter'] = 'kept'
                confirmed.append(det)

        return confirmed

    def merge_related_detections(self, detections, x_overlap_thresh=0.35, vertical_gap_thresh=0.08):
        """Merge boxes that look like stacked/overlapping parts of the same person."""
        if len(detections) < 2:
            return detections

        def should_merge(a, b):
            ax1, ay1, ax2, ay2 = a['bbox']
            bx1, by1, bx2, by2 = b['bbox']
            aw, bw = ax2 - ax1, bx2 - bx1
            ah, bh = ay2 - ay1, by2 - by1
            if aw <= 0 or bw <= 0 or ah <= 0 or bh <= 0:
                return False

            x_overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
            x_overlap_ratio = x_overlap / max(min(aw, bw), 1e-6)
            cx_gap = abs(((ax1 + ax2) / 2.0) - ((bx1 + bx2) / 2.0))
            same_column = x_overlap_ratio >= x_overlap_thresh or cx_gap <= min(aw, bw) * 0.65

            y_overlap = max(0.0, min(ay2, by2) - max(ay1, by1))
            y_gap = max(0.0, max(ay1, by1) - min(ay2, by2))
            vertically_related = y_overlap > 0.0 or y_gap <= vertical_gap_thresh

            return same_column and vertically_related

        parent = list(range(len(detections)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                if should_merge(detections[i], detections[j]):
                    union(i, j)

        groups = {}
        for i, det in enumerate(detections):
            groups.setdefault(find(i), []).append(det)

        merged = []
        for group in groups.values():
            if len(group) == 1:
                merged.append(group[0])
                continue

            best = max(group, key=lambda d: d.get('combined_conf', d.get('conf', 0.0)))
            out = dict(best)
            out['bbox'] = [
                min(d['bbox'][0] for d in group),
                min(d['bbox'][1] for d in group),
                max(d['bbox'][2] for d in group),
                max(d['bbox'][3] for d in group),
            ]
            out['conf'] = max(d.get('conf', 0.0) for d in group)
            out['combined_conf'] = max(d.get('combined_conf', 0.0) for d in group)
            out['thermal_peak'] = max(d.get('thermal_peak', 0.0) for d in group)
            out['thermal_score'] = max(d.get('thermal_score', 0.0) for d in group)
            out['lap_var'] = max(d.get('lap_var', 0.0) for d in group)
            out['merged_parts'] = len(group)
            merged.append(out)

        return sorted(merged, key=lambda d: d.get('combined_conf', d.get('conf', 0.0)), reverse=True)

    def suppress_duplicate_detections(self, detections, iou_thresh=0.22, ios_thresh=0.55):
        """Remove duplicate boxes that describe the same target at different scales."""
        if len(detections) < 2:
            return detections

        def area(box):
            return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

        def intersection(a, b):
            return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))

        def center_inside(inner, outer):
            cx = (inner[0] + inner[2]) / 2.0
            cy = (inner[1] + inner[3]) / 2.0
            return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]

        def same_column_overlap(a, b):
            aw, ah = a[2] - a[0], a[3] - a[1]
            bw, bh = b[2] - b[0], b[3] - b[1]
            if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
                return False
            x_overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) / max(min(aw, bw), 1e-6)
            y_overlap = max(0.0, min(a[3], b[3]) - max(a[1], b[1])) / max(min(ah, bh), 1e-6)
            return x_overlap >= 0.60 and y_overlap >= 0.20

        ranked = sorted(
            detections,
            key=lambda d: (d.get('combined_conf', d.get('conf', 0.0)), area(d['bbox'])),
            reverse=True,
        )
        kept = []

        for det in ranked:
            box = det['bbox']
            box_area = area(box)
            duplicate = False

            for kept_idx, kept_det in enumerate(kept):
                kept_box = kept_det['bbox']
                kept_area = area(kept_box)
                if box_area <= 0 or kept_area <= 0:
                    continue

                inter = intersection(box, kept_box)
                ios = inter / max(min(box_area, kept_area), 1e-6)
                iou = calculate_iou_numpy(box, kept_box)
                area_ratio = min(box_area, kept_area) / max(box_area, kept_area)
                det_score = det.get('combined_conf', det.get('conf', 0.0))
                kept_score = kept_det.get('combined_conf', kept_det.get('conf', 0.0))

                if iou >= iou_thresh or ios >= ios_thresh:
                    if box_area > kept_area and det_score >= kept_score - 0.10:
                        kept[kept_idx] = det
                    duplicate = True
                    break
                if area_ratio <= 0.80 and (center_inside(box, kept_box) or center_inside(kept_box, box)):
                    if box_area > kept_area and center_inside(kept_box, box) and det_score >= kept_score - 0.10:
                        kept[kept_idx] = det
                    duplicate = True
                    break
                if same_column_overlap(box, kept_box):
                    if box_area > kept_area and det_score >= kept_score - 0.10:
                        kept[kept_idx] = det
                    duplicate = True
                    break

            if not duplicate:
                kept.append(det)

        return kept[:MAX_DETECTIONS]

    def detect_confirmed(self, image_rgb, image_thermal, lap_thresh=80.0,
                         lap_image=None,
                         high_conf_bypass=None, is_eval=False,
                         conf_threshold=None, merge_boxes=False,
                         suppress_duplicates=True):
        """Run detection, then reject flat crops such as hot car bonnets."""
        detections = self.detect(
            image_rgb,
            image_thermal,
            is_eval=is_eval,
            conf_threshold=conf_threshold,
        )
        detections = self.filter_by_laplacian(
            lap_image if lap_image is not None else image_rgb,
            detections,
            lap_thresh=lap_thresh,
            high_conf_bypass=high_conf_bypass,
            thermal_present=bool(np.any(image_thermal)),
            rgb_present=bool(np.any(image_rgb)),
        )
        paired_input = bool(np.any(image_rgb)) and bool(np.any(image_thermal))
        if merge_boxes:
            detections = self.merge_related_detections(detections)
        if suppress_duplicates and not paired_input:
            detections = self.suppress_duplicate_detections(detections)
        return detections

# ============================================================================
# 3. ALERT GENERATOR & EXPORT
# ============================================================================

class AlertGenerator:
    """Generates payloads for transmission from edge node."""
    def __init__(self, config=ALERT_CONFIG):
        self.config = config

    def generate_json_alert(self, detections, node_id="NODE-001"):
        if not detections: return None
        alert = {"node": node_id, "status": "INTRUSION_DETECTED"}
        if self.config['include_timestamp']:
            alert["timestamp"] = int(time.time())
        if self.config['include_intrusion_count']:
            alert["count"] = len(detections)
        targets = []
        for d in detections:
            t = {}
            if self.config['include_confidence']: t["conf"] = round(d['combined_conf'], 3)
            t["class"] = d.get('class', 'unknown')
            if self.config['include_bbox']: t["bbox"] = [round(x, 3) for x in d['bbox']]
            if self.config.get('include_thermal_score', True): t["thermal_score"] = d.get('thermal_score', 0.0)
            if 'gate_w' in d and d['gate_w'] is not None: t['gate_w'] = d['gate_w']
            targets.append(t)
        alert["targets"] = targets
        return json.dumps(alert)


def export_to_onnx(model, save_path):
    """Export PyTorch model to ONNX."""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    model.eval()
    model.to('cpu')
    dummy_input = torch.randn(1, 4, INPUT_HEIGHT, INPUT_WIDTH)
    
    # V2 has w_rgb, w_thm outputs. Collect actual output keys from a forward pass
    with torch.no_grad():
        out = model(dummy_input)
    output_names = list(out.keys())
    
    torch.onnx.export(
        model, dummy_input, save_path,
        export_params=True, opset_version=11, do_constant_folding=True,
        input_names=['input'], output_names=output_names,
        dynamic_axes={'input': {0: 'batch_size'}}
    )
    print(f"Exported ONNX model to {save_path}")
    return save_path



def benchmark_model(model):
    """Run performance benchmarks on the model."""
    print("\n" + "="*50)
    print("RUNNING PERFORMANCE BENCHMARK (V2)")
    print("="*50)
    model.eval()
    model.to('cpu')
    sram_info = estimate_peak_sram(model)
    print("Memory Estimates:")
    print(f"  Flash (INT8 weights):  ~{sram_info['peak_activation_int8_kb'] * 2:.1f} KB") 
    print(f"  Peak SRAM Arena:       {sram_info['total_arena_int8_kb']:.1f} KB")
    print(f"  Fits ESP32-S3:         {'YES' if sram_info['fits_esp32_s3'] else 'NO'}")

    dummy = torch.randn(1, 4, INPUT_HEIGHT, INPUT_WIDTH)
    with torch.no_grad():
        for _ in range(10): model(dummy)
    iters = 100
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters): model(dummy)
    t1 = time.time()
    ms_per_frame = ((t1 - t0) / iters) * 1000
    print(f"  Average Time:          {ms_per_frame:.2f} ms")
    print(f"  Throughput:            {iters / (t1 - t0):.1f} FPS")
    print("="*50)

if __name__ == '__main__':
    print("Inference Module (V2) Test:")
    gen = AlertGenerator()
    dummy_dets = [
        {'conf': 0.85, 'combined_conf': 0.9, 'bbox': [0.1, 0.2, 0.3, 0.4], 'thermal_score': 0.92, 
         'gate_w': {'rgb': 0.8, 'thm': 0.2}},
    ]
    print(gen.generate_json_alert(dummy_dets))
