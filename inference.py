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
    
    # Determine model version
    is_v2 = config.get('model_version', 'v1') == 'v2'

    anchors_to_use = override_num_anchors or config.get('num_anchors', NUM_ANCHORS)

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

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Loaded {'MicroGhost-V2' if is_v2 else 'MicroGhost-V1'} successfully.")
    return model, anchors_to_use


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

        kept.append(det)
    return kept


def decode_predictions(obj_small, bbox_small, obj_large, bbox_large,
                       anchor_sizes=None, conf_threshold=None, is_eval=False):
    """
    Decode SSD-style head outputs into detection candidates (normalized xyxy).
    """
    anchor_sizes = anchor_sizes or DEFAULT_ANCHOR_SIZES
    
    # Use appropriate threshold
    if conf_threshold is None:
        conf_threshold = EVAL_CONFIDENCE_THRESHOLD if is_eval else CONFIDENCE_THRESHOLD

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
                    'conf': float(obj_probs[a, gy, gx]),
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
            self.model, anchors_used = load_inference_model(model_path, device, override_num_anchors)
            if anchors_used == 2:
                self.anchor_sizes = [0.108, 0.180] # Fallback V1 sizes
            else:
                self.anchor_sizes = DEFAULT_ANCHOR_SIZES
        else:
            raise ValueError("Must provide either model_path or model instance.")

    def detect(self, image_rgb, image_thermal, is_eval=False):
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

        # 3. Extract boxes from both scales
        final_detections = decode_predictions(
            preds['obj_small'][0], preds['bbox_small'][0],
            preds['obj_large'][0], preds['bbox_large'][0],
            anchor_sizes=self.anchor_sizes,
            is_eval=is_eval
        )

        # 4. Classifier labels detections
        cls_probs = torch.softmax(preds['label'], dim=1)[0].cpu().numpy()
        pred_class_idx = int(np.argmax(cls_probs))
        
        # If classifier says background, return empty (unless in strict eval where we might bypass)
        if pred_class_idx == CLASS_MAP['background']:
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
            det['class'] = detected_class_name
            det['combined_conf'] = (det['conf'] + class_conf) / 2.0
            det['gate_w'] = gate_w

            # Pseudo-radiometric temp extraction
            x1, y1, x2, y2 = det['bbox']
            x1_p, y1_p = int(x1 * w_orig), int(y1 * h_orig)
            x2_p, y2_p = int(x2 * w_orig), int(y2 * h_orig)
            x1_p, y1_p = max(0, x1_p), max(0, y1_p)
            x2_p, y2_p = min(w_orig, x2_p), min(h_orig, y2_p)

            if x2_p > x1_p and y2_p > y1_p:
                crop = image_thermal[y1_p:y2_p, x1_p:x2_p]
                max_val = np.max(crop)
                det['temp_c'] = round(20.0 + (max_val / 255.0) * 20.0, 1)
            else:
                det['temp_c'] = 0.0

        return final_detections

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
            if self.config['include_temperature']: t["temp_c"] = d.get('temp_c', 0.0)
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
        {'conf': 0.85, 'combined_conf': 0.9, 'bbox': [0.1, 0.2, 0.3, 0.4], 'temp_c': 35.2, 
         'gate_w': {'rgb': 0.8, 'thm': 0.2}},
    ]
    print(gen.generate_json_alert(dummy_dets))
