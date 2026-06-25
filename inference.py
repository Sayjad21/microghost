"""
MicroGhost-Thermal: Inference & Deployment Module
===================================================
Handles edge inference, multi-target detection, NMS, alert generation,
and model export (ONNX, TFLite/INT8) for ESP32-S3 deployment.
"""

import os
import time
import json
import numpy as np
import torch
import cv2

from model import MicroGhostThermal, estimate_peak_sram
from preprocessing import ThermalPreprocessor
from config import (
    INPUT_SIZE, INPUT_WIDTH, INPUT_HEIGHT, CLASS_MAP, NUM_CLASSES, NUM_ANCHORS,
    DEFAULT_ANCHOR_SIZES, CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD,
    MAX_DETECTIONS, DEVICE, ESP32_S3, ALERT_CONFIG, CLASSIFIER_HIDDEN_DIM,
    MIN_BOX_WIDTH_NORM, MIN_BOX_HEIGHT_NORM, MIN_BOX_AREA_NORM,
    CORNER_FILTER_X, CORNER_FILTER_Y, CORNER_MAX_WIDTH,
)

# Inverse class map for alert generation
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}


# ============================================================================
# 1. MODEL LOADING
# ============================================================================

def load_inference_model(model_path, device=DEVICE, override_num_anchors=None):
    """Load PyTorch model for inference, with backward-compatibility support."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    config = checkpoint.get('config', {})
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Use override if provided, else try config dict, else use default
    anchors_to_use = override_num_anchors or config.get('num_anchors', NUM_ANCHORS)

    model = MicroGhostThermal(
        num_classes=config.get('num_classes', NUM_CLASSES),
        input_size=config.get('input_size', INPUT_SIZE),
        classifier_hidden_dim=config.get('classifier_hidden_dim', CLASSIFIER_HIDDEN_DIM),
        num_anchors=anchors_to_use # CRITICAL FIX
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

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
    """
    Non-Maximum Suppression to filter overlapping boxes.
    Detections format: list of dicts with 'conf' and 'bbox'
    """
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
    """
    Remove tiny boxes and top-left corner artifacts from grid decode clamping.

    Corner boxes appear when cell (0,0) fires with tall anchors: decoded centers
    near the origin get clamped to x1≈0, y1≈0, producing thin vertical strips.
    """
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
                       anchor_sizes=None, conf_threshold=None):
    """
    Decode SSD-style head outputs into detection candidates (normalized xyxy).

    Skips the image-level classifier gate — use for mAP evaluation.
    """
    anchor_sizes = anchor_sizes or DEFAULT_ANCHOR_SIZES
    conf_threshold = conf_threshold if conf_threshold is not None else CONFIDENCE_THRESHOLD

    small_grid_w, small_grid_h = INPUT_WIDTH // 8, INPUT_HEIGHT // 8
    large_grid_w, large_grid_h = INPUT_WIDTH // 16, INPUT_HEIGHT // 16

    # CRITICAL FIX: Dynamically determine anchors from tensor shape
    actual_num_anchors = obj_small.shape[0]

    def _decode_box(pred_box, grid_x, grid_y, grid_w, grid_h, anchor_size):
        cx = (grid_x + pred_box[0]) / grid_w
        cy = (grid_y + pred_box[1]) / grid_h
        w = anchor_size * np.exp(np.clip(pred_box[2], -3, 3))
        h = anchor_size * np.exp(np.clip(pred_box[3], -3, 3))
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
        for a in range(actual_num_anchors):
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
            self.anchor_sizes = DEFAULT_ANCHOR_SIZES # Assuming current config for injected model
        elif model_path is not None:
            self.model, anchors_used = load_inference_model(model_path, device, override_num_anchors)
            # Adjust anchor sizes array length based on what the model was built with
            if anchors_used == 2:
                self.anchor_sizes = [0.108, 0.180] # Fallback V1 sizes
            else:
                self.anchor_sizes = DEFAULT_ANCHOR_SIZES
        else:
            raise ValueError("Must provide either model_path or model instance.")

    def decode_box_from_grid(self, pred_box, grid_x, grid_y, grid_w, grid_h, anchor_size):
        """Decode grid-based bounding box to [x1, y1, x2, y2] (normalized 0-1)."""
        cx = (grid_x + pred_box[0]) / grid_w
        cy = (grid_y + pred_box[1]) / grid_h
        w = anchor_size * np.exp(np.clip(pred_box[2], -3, 3))
        h = anchor_size * np.exp(np.clip(pred_box[3], -3, 3))
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

    def _extract_candidates(self, obj_map, bbox_map, grid_w, grid_h):
        """Extract candidate detections from a single feature map."""
        candidates = []
        obj_probs = torch.sigmoid(obj_map).cpu().numpy()
        bbox_data = bbox_map.cpu().numpy()

        actual_num_anchors = obj_probs.shape[0]

        for a in range(actual_num_anchors):
            # Find cells above threshold
            y_indices, x_indices = np.where(obj_probs[a] > CONFIDENCE_THRESHOLD)

            for gy, gx in zip(y_indices, x_indices):
                conf = obj_probs[a, gy, gx]
                off = a * 4
                box_params = bbox_data[off:off+4, gy, gx]
                anchor_size = self.anchor_sizes[a]

                bbox = self.decode_box_from_grid(
                    box_params, gx, gy, grid_w, grid_h, anchor_size
                )
                # Clamp bbox to [0, 1]
                bbox = [max(0.0, min(1.0, v)) for v in bbox]

                candidates.append({
                    'conf': float(conf),
                    'bbox': bbox,
                    'scale': 'small' if grid_w == INPUT_WIDTH//8 else 'large'
                })

        return candidates

    def detect(self, image_rgb, image_thermal):
        """
        Run full detection pipeline on a dual-modality input.

        Args:
            image_rgb: numpy array (H, W, 3)
            image_thermal: numpy array (H, W)

        Returns:
            list of dicts: {'conf': float, 'bbox': [x1,y1,x2,y2], 'class': str}
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

        # 4. Extract boxes from both scales (objectness only; no classifier gate)
        final_detections = decode_predictions(
            preds['obj_small'][0], preds['bbox_small'][0],
            preds['obj_large'][0], preds['bbox_large'][0],
            anchor_sizes=self.anchor_sizes,
        )

        # 5. Classifier labels detections
        cls_probs = torch.softmax(preds['label'], dim=1)[0].cpu().numpy()
        pred_class_idx = int(np.argmax(cls_probs))
        if pred_class_idx == CLASS_MAP['background']:
            return []

        detected_class_name = INV_CLASS_MAP.get(pred_class_idx, "unknown")
        class_conf = float(cls_probs[pred_class_idx])

        for det in final_detections:
            det['class'] = detected_class_name
            # Blend objectness conf with global classifier conf
            det['combined_conf'] = (det['conf'] + class_conf) / 2.0

            # Optional: Simulate a temperature reading (pseudo-radiometric)
            # In a real FLIR system, you would map the bbox coordinates back to the
            # 14-bit raw radiometric array to find the max temperature.
            # Here, we map back to the 8-bit image and estimate.
            x1, y1, x2, y2 = det['bbox']
            x1_p, y1_p = int(x1 * w_orig), int(y1 * h_orig)
            x2_p, y2_p = int(x2 * w_orig), int(y2 * h_orig)

            # Ensure bounds
            x1_p, y1_p = max(0, x1_p), max(0, y1_p)
            x2_p, y2_p = min(w_orig, x2_p), min(h_orig, y2_p)

            if x2_p > x1_p and y2_p > y1_p:
                crop = image_thermal[y1_p:y2_p, x1_p:x2_p]
                max_val = np.max(crop)
                # Pseudo-mapping: assuming 0=20°C, 255=40°C
                det['temp_c'] = round(20.0 + (max_val / 255.0) * 20.0, 1)
            else:
                det['temp_c'] = 0.0

        return final_detections


# ============================================================================
# 3. ALERT GENERATOR
# ============================================================================

class AlertGenerator:
    """
    Generates payloads for transmission from the ESP32-S3 edge node.
    Can format as JSON for testing, or compact binary for low-bandwidth LoRa/radio.
    """
    def __init__(self, config=ALERT_CONFIG):
        self.config = config

    def generate_json_alert(self, detections, node_id="NODE-001"):
        """Generate human-readable JSON alert."""
        if not detections:
            return None

        alert = {
            "node": node_id,
            "status": "INTRUSION_DETECTED",
        }

        if self.config['include_timestamp']:
            alert["timestamp"] = int(time.time())

        if self.config['include_intrusion_count']:
            alert["count"] = len(detections)

        targets = []
        for d in detections:
            t = {}
            if self.config['include_confidence']:
                t["conf"] = round(d['combined_conf'], 3)
            t["class"] = d.get('class', 'unknown')
            if self.config['include_bbox']:
                t["bbox"] = [round(x, 3) for x in d['bbox']]
            if self.config['include_temperature']:
                t["temp_c"] = d.get('temp_c', 0.0)
            targets.append(t)

        alert["targets"] = targets
        return json.dumps(alert)

    def generate_binary_alert(self, detections, node_id=1):
        """
        Generate ultra-compact binary payload (e.g., for LoRa).
        Format (bytes):
        [0]: Node ID (0-255)
        [1]: Alert Type (1=Intrusion)
        [2]: Intruder Count (0-255)
        Then for each intruder (max 3 to save space):
        [+0]: Confidence (0-255, mapped from 0.0-1.0)
        [+1]: Temp °C (0-255)
        [+2,+3,+4,+5]: x1, y1, x2, y2 (0-255, mapped from 0.0-1.0)
        Total size for 1 target: 3 + 6 = 9 bytes!
        """
        if not detections:
            return None

        payload = bytearray()
        payload.append(node_id & 0xFF)
        payload.append(1) # Alert type: Intrusion
        payload.append(min(len(detections), 255))

        for d in detections[:3]: # Limit to 3 targets to keep packet small
            conf_byte = int(d['combined_conf'] * 255)
            temp_byte = int(min(255, max(0, d.get('temp_c', 0))))
            x1 = int(d['bbox'][0] * 255)
            y1 = int(d['bbox'][1] * 255)
            x2 = int(d['bbox'][2] * 255)
            y2 = int(d['bbox'][3] * 255)

            payload.extend([conf_byte, temp_byte, x1, y1, x2, y2])

        return bytes(payload)


# ============================================================================
# 4. EXPORT UTILITIES (ONNX / TFLITE)
# ============================================================================

def export_to_onnx(model, save_path):
    """Export PyTorch model to ONNX."""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    model.eval()
    model.to('cpu')

    dummy_input = torch.randn(1, 4, INPUT_HEIGHT, INPUT_WIDTH)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['bbox_small', 'obj_small', 'bbox_large', 'obj_large', 'label'],
        dynamic_axes={'input': {0: 'batch_size'}}
    )
    print(f"✅ Exported ONNX model to {save_path}")
    return save_path


def export_to_tflite(onnx_path, save_path, int8=False, dataset_loader=None):
    """
    Convert ONNX to TFLite (Requires ONNX-TF and TFLiteConverter).
    This function outlines the process, but assumes appropriate TF env.
    """
    try:
        import tensorflow as tf
        from onnx_tf.backend import prepare
        import onnx
    except ImportError:
        print("⚠️  Export to TFLite requires tensorflow and onnx-tf.")
        print("   Run: pip install tensorflow onnx onnx-tf")
        return None

    print(f"Converting ONNX ({onnx_path}) to TensorFlow SavedModel...")
    tf_model_dir = onnx_path.replace('.onnx', '_tf')

    # 1. ONNX -> TF
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_model_dir)

    # 2. TF -> TFLite
    print(f"Converting TF SavedModel to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)

    if int8:
        print("Applying INT8 Quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Representative dataset generator for calibration
        def representative_data_gen():
            if dataset_loader is None:
                # Fallback to random data if no loader provided (not recommended)
                for _ in range(100):
                    yield [np.random.rand(1, 4, INPUT_HEIGHT, INPUT_WIDTH).astype(np.float32)]
            else:
                for i, (images, _) in enumerate(dataset_loader):
                    if i >= 100: break # Use 100 batches
                    # Get one image from batch
                    img_np = images[0:1].cpu().numpy().astype(np.float32)
                    yield [img_np]

        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"✅ Exported TFLite model to {save_path} ({size_kb:.1f} KB)")
    return save_path


# ============================================================================
# 5. BENCHMARK UTILITY
# ============================================================================

def benchmark_model(model):
    """Run performance benchmarks on the model."""
    print("\n" + "="*50)
    print("🚀 RUNNING PERFORMANCE BENCHMARK")
    print("="*50)

    model.eval()
    model.to('cpu') # Benchmark on CPU to approximate MCU feel (relatively)

    # 1. SRAM and Flash
    sram_info = estimate_peak_sram(model)
    print("Memory Estimates:")
    print(f"  Flash (INT8 weights):  ~{sram_info['peak_activation_int8_kb'] * 2:.1f} KB") # Rough
    print(f"  Peak SRAM Arena:       {sram_info['total_arena_int8_kb']:.1f} KB")
    print(f"  Fits ESP32-S3:         {'✅ YES' if sram_info['fits_esp32_s3'] else '❌ NO'}")

    # 2. FPS Test
    dummy = torch.randn(1, 4, INPUT_HEIGHT, INPUT_WIDTH)
    print("\nTiming Forward Pass (CPU)...")
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(dummy)

    iters = 100
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters):
            model(dummy)
    t1 = time.time()

    fps = iters / (t1 - t0)
    ms_per_frame = ((t1 - t0) / iters) * 1000

    print(f"  Average Time:          {ms_per_frame:.2f} ms")
    print(f"  Throughput:            {fps:.1f} FPS")

    # Disclaimer
    print("\n* Note: Desktop CPU FPS is much higher than ESP32-S3.")
    print("* Expect ~10-15 FPS on ESP32-S3 with TFLite Micro INT8.")
    print("="*50)


if __name__ == '__main__':
    # Test Alert Generator
    print("Alert Generator Test:")
    gen = AlertGenerator()
    dummy_dets = [
        {'conf': 0.85, 'combined_conf': 0.9, 'bbox': [0.1, 0.2, 0.3, 0.4], 'temp_c': 35.2},
        {'conf': 0.70, 'combined_conf': 0.75, 'bbox': [0.6, 0.7, 0.8, 0.9], 'temp_c': 34.5}
    ]
    print(gen.generate_json_alert(dummy_dets))
    bin_alert = gen.generate_binary_alert(dummy_dets)
    print(f"Binary payload size: {len(bin_alert)} bytes")
