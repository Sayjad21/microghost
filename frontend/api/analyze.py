import base64
import os
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

try:
    import psutil
except ImportError:  # Vercel installs this from requirements.txt; local fallback keeps dev resilient.
    psutil = None


API_DIR = Path(__file__).resolve().parent
MODEL_PATH = API_DIR / "models" / "microghost_thermal.onnx"

INPUT_WIDTH = 160
INPUT_HEIGHT = 128
NUM_CLASSES = 4
CLASS_MAP = {
    "background": 0,
    "person_visible": 1,
    "person_camouflaged": 2,
    "vehicle_boat": 3,
}
INV_CLASS_MAP = {value: key for key, value in CLASS_MAP.items()}

DEFAULT_ANCHOR_SIZES = [0.0717509437717046, 0.13346257948562704, 0.4044514489729058]
CONFIDENCE_THRESHOLD = 0.20
DEFAULT_LAPLACIAN_THRESHOLD = 80.0
THERMAL_ONLY_CONFIDENCE_THRESHOLD = 0.20
RGB_ONLY_CONFIDENCE_THRESHOLD = 0.16
RGB_ONLY_MIN_OBJECTNESS = 0.18
RGB_ONLY_MAX_DETECTIONS = 4
NMS_IOU_THRESHOLD = 0.35
MAX_DETECTIONS = 10
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_IMAGE_SIDE = 1600
MIN_BOX_WIDTH_NORM = 0.03
MIN_BOX_HEIGHT_NORM = 0.05
MIN_BOX_AREA_NORM = 0.002
LOG_CLAMP_MIN = -4.5
LOG_CLAMP_MAX = 4.5


app = FastAPI()
_engine = None
_engine_lock = threading.Lock()
_process = psutil.Process(os.getpid()) if psutil is not None else None


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def softmax(values):
    values = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / np.sum(exp, axis=1, keepdims=True)


def normalize_thermal(image):
    img_min = image.min()
    img_max = image.max()
    if img_max > img_min:
        return ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    return image


def build_rgb_thermal_proxy(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    return cv2.GaussianBlur(inverted, (3, 3), 0)


def calculate_iou(box1, box2):
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


def nms(detections):
    if not detections:
        return []

    detections = sorted(detections, key=lambda item: item["conf"], reverse=True)
    keep = []

    for det in detections:
        if all(calculate_iou(det["bbox"], kept["bbox"]) <= NMS_IOU_THRESHOLD for kept in keep):
            keep.append(det)

    return keep[:MAX_DETECTIONS]


def filter_spurious_detections(detections):
    kept = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        width = x2 - x1
        height = y2 - y1
        area = width * height

        if width < MIN_BOX_WIDTH_NORM or height < MIN_BOX_HEIGHT_NORM:
            continue
        if area < MIN_BOX_AREA_NORM:
            continue
        touches_side = x1 <= 0.002 or x2 >= 0.998
        touches_bottom = y2 >= 0.998
        if (touches_side or touches_bottom) and (width < 0.06 or height < 0.16):
            continue

        kept.append(det)
    return kept


def decode_box(pred_box, grid_x, grid_y, grid_w, grid_h, anchor_size):
    center_x = (grid_x + pred_box[0]) / grid_w
    center_y = (grid_y + pred_box[1]) / grid_h
    width = anchor_size * np.exp(np.clip(pred_box[2], LOG_CLAMP_MIN, LOG_CLAMP_MAX))
    height = anchor_size * np.exp(np.clip(pred_box[3], LOG_CLAMP_MIN, LOG_CLAMP_MAX))

    return [
        max(0.0, min(1.0, center_x - width / 2)),
        max(0.0, min(1.0, center_y - height / 2)),
        max(0.0, min(1.0, center_x + width / 2)),
        max(0.0, min(1.0, center_y + height / 2)),
    ]


def decode_predictions(obj_small, bbox_small, obj_large, bbox_large, iou_pred, conf_threshold):
    threshold = CONFIDENCE_THRESHOLD if conf_threshold is None else conf_threshold
    small_grid_w = INPUT_WIDTH // 8
    small_grid_h = INPUT_HEIGHT // 8
    large_grid_w = INPUT_WIDTH // 16
    large_grid_h = INPUT_HEIGHT // 16

    def extract(obj_map, bbox_map, grid_w, grid_h):
        candidates = []
        obj_probs = sigmoid(obj_map)

        for anchor_idx in range(obj_probs.shape[0]):
            y_idx, x_idx = np.where(obj_probs[anchor_idx] > threshold)
            for gy, gx in zip(y_idx, x_idx):
                offset = anchor_idx * 4
                bbox = decode_box(
                    bbox_map[offset:offset + 4, gy, gx],
                    gx,
                    gy,
                    grid_w,
                    grid_h,
                    DEFAULT_ANCHOR_SIZES[anchor_idx],
                )
                candidates.append(
                    {
                        "conf": float(obj_probs[anchor_idx, gy, gx] * iou_pred),
                        "bbox": bbox,
                    }
                )

        return candidates

    candidates = []
    candidates.extend(extract(obj_small, bbox_small, small_grid_w, small_grid_h))
    candidates.extend(extract(obj_large, bbox_large, large_grid_w, large_grid_h))
    return nms(filter_spurious_detections(candidates))


def filter_by_laplacian(lap_image, detections, lap_thresh, thermal_present=True, rgb_present=True):
    if lap_thresh is None:
        return detections

    h_lap, w_lap = lap_image.shape[:2]
    confirmed = []

    for source_det in detections:
        x1, y1, x2, y2 = source_det["bbox"]
        box_w = x2 - x1
        box_h = y2 - y1
        left = max(0, min(w_lap, int(x1 * w_lap)))
        top = max(0, min(h_lap, int(y1 * h_lap)))
        right = max(0, min(w_lap, int(x2 * w_lap)))
        bottom = max(0, min(h_lap, int(y2 * h_lap)))

        det = dict(source_det)
        small_thermal_target = thermal_present and not rgb_present and box_w < 0.09 and box_h < 0.14
        weak_small_target = det.get("thermal_peak", 0.0) < 0.93 or det.get("combined_conf", 0.0) < 0.35
        if small_thermal_target and weak_small_target:
            det["lap_filter"] = "rejected_small_thermal_blob"
            continue

        if right <= left or bottom <= top:
            det["lap_var"] = 0.0
            continue

        crop = lap_image[top:bottom, left:right]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        det["lap_var"] = round(lap_var, 2)

        if thermal_present and not rgb_present:
            hot_thresh = max(float(np.percentile(gray, 85)), 180.0)
            hot_mask = gray >= hot_thresh
            if np.any(hot_mask):
                ys, xs = np.where(hot_mask)
                hot_ratio = float(hot_mask.mean())
                hot_vspan = float((ys.max() - ys.min() + 1) / max(gray.shape[0], 1))
                det["thermal_hot_ratio"] = round(hot_ratio, 3)
                det["thermal_hot_vspan"] = round(hot_vspan, 3)
                weak_small_hot_target = det.get("thermal_peak", 0.0) < 0.93 or det.get("combined_conf", 0.0) < 0.35
                if hot_ratio < 0.06 and hot_vspan < 0.30 and weak_small_hot_target:
                    det["lap_filter"] = "rejected_small_hot_blob"
                    continue

        if crop.ndim == 3:
            mean_intensity = float(gray.mean())
            top_band = y1 <= 0.12 and y2 <= 0.30
            if top_band and mean_intensity < 45.0 and lap_var < 120.0:
                det["lap_filter"] = "rejected_top_dark_smooth"
                continue

        min_thermal_peak = 0.95 if thermal_present and rgb_present else 0.70
        if thermal_present and det.get("thermal_peak", 0.0) < min_thermal_peak and det.get("combined_conf", 0.0) < 0.70:
            det["lap_filter"] = "rejected_low_thermal_low_conf"
            continue

        thermal_only_hot_body = (
            thermal_present
            and not rgb_present
            and det.get("combined_conf", 0.0) >= 0.72
            and det.get("thermal_peak", 0.0) >= 0.95
            and det.get("thermal_hot_ratio", 0.0) >= 0.10
            and det.get("thermal_hot_vspan", 0.0) >= 0.55
            and box_w >= 0.045
            and box_h >= 0.16
        )

        if lap_var >= lap_thresh or thermal_only_hot_body:
            det["lap_filter"] = "kept"
            if thermal_only_hot_body and lap_var < lap_thresh:
                det["lap_filter"] = "kept_hot_thermal_body"
            confirmed.append(det)

    return confirmed


def merge_related_detections(detections):
    if len(detections) < 2:
        return detections

    def should_merge(a, b):
        ax1, ay1, ax2, ay2 = a["bbox"]
        bx1, by1, bx2, by2 = b["bbox"]
        aw, bw = ax2 - ax1, bx2 - bx1
        ah, bh = ay2 - ay1, by2 - by1
        if aw <= 0 or bw <= 0 or ah <= 0 or bh <= 0:
            return False

        x_overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        x_overlap_ratio = x_overlap / max(min(aw, bw), 1e-6)
        cx_gap = abs(((ax1 + ax2) / 2.0) - ((bx1 + bx2) / 2.0))
        same_column = x_overlap_ratio >= 0.65 or cx_gap <= min(aw, bw) * 0.35

        y_overlap = max(0.0, min(ay2, by2) - max(ay1, by1))
        y_gap = max(0.0, max(ay1, by1) - min(ay2, by2))
        vertically_related = y_overlap > 0.0 or y_gap <= 0.08

        return same_column and vertically_related

    parent = list(range(len(detections)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            if should_merge(detections[i], detections[j]):
                union(i, j)

    groups = {}
    for index, det in enumerate(detections):
        groups.setdefault(find(index), []).append(det)

    merged = []
    for group in groups.values():
        if len(group) == 1:
            merged.append(group[0])
            continue

        best = max(group, key=lambda item: item.get("combined_conf", item.get("conf", 0.0)))
        out = dict(best)
        out["bbox"] = [
            min(item["bbox"][0] for item in group),
            min(item["bbox"][1] for item in group),
            max(item["bbox"][2] for item in group),
            max(item["bbox"][3] for item in group),
        ]
        out["conf"] = max(item.get("conf", 0.0) for item in group)
        out["combined_conf"] = max(item.get("combined_conf", 0.0) for item in group)
        out["thermal_peak"] = max(item.get("thermal_peak", 0.0) for item in group)
        out["thermal_score"] = max(item.get("thermal_score", 0.0) for item in group)
        out["lap_var"] = max(item.get("lap_var", 0.0) for item in group)
        out["merged_parts"] = len(group)
        merged.append(out)

    return sorted(merged, key=lambda item: item.get("combined_conf", item.get("conf", 0.0)), reverse=True)


def filter_thermal_only_artifacts(detections):
    kept = []
    for det in detections:
        x1, y1, x2, y2 = [float(value) for value in det["bbox"]]
        width = x2 - x1
        height = y2 - y1
        conf = float(det.get("combined_conf", det.get("conf", 0.0)))
        thermal_peak = float(det.get("thermal_peak", 0.0))

        touches_top_artifact_band = y1 <= 0.08 and y2 <= 0.30
        wide_low_thermal_low_conf_artifact = width >= 0.12 and conf < 0.70 and thermal_peak < 0.70
        too_flat = height <= 0 or width <= 0

        if touches_top_artifact_band or wide_low_thermal_low_conf_artifact or too_flat:
            continue

        kept.append(det)

    return kept


def filter_rgb_only_detections(detections):
    kept = []
    for det in detections:
        x1, y1, x2, y2 = [float(value) for value in det["bbox"]]
        width = x2 - x1
        height = y2 - y1
        aspect = height / max(width, 1e-6)
        objectness = float(det.get("conf", 0.0))
        confidence = float(det.get("combined_conf", objectness))

        too_weak = objectness < RGB_ONLY_MIN_OBJECTNESS and confidence < 0.62
        not_person_shaped = aspect < 1.15 or aspect > 4.2
        too_small = width < 0.035 or height < 0.055
        too_flat = height <= 0 or width <= 0

        if too_weak or not_person_shaped or too_small or too_flat:
            continue

        kept.append(det)

    return sorted(
        kept,
        key=lambda item: (item.get("conf", 0.0), item.get("combined_conf", 0.0)),
        reverse=True,
    )[:RGB_ONLY_MAX_DETECTIONS]


class OnnxMicroGhostEngine:
    def __init__(self, model_path):
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def preprocess(self, image_rgb, image_thermal):
        image_thermal = normalize_thermal(image_thermal)
        image_rgb = cv2.resize(image_rgb, (INPUT_WIDTH, INPUT_HEIGHT))
        image_thermal = cv2.resize(image_thermal, (INPUT_WIDTH, INPUT_HEIGHT))

        tensor_rgb = image_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor_thermal = image_thermal[None, :, :].astype(np.float32) / 255.0
        return np.concatenate([tensor_rgb, tensor_thermal], axis=0)[None, :, :, :]

    def detect_confirmed(self, image_rgb, image_thermal, lap_image, lap_thresh, conf_threshold, merge_boxes=False):
        input_tensor = self.preprocess(image_rgb, image_thermal)
        raw_outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        outputs = dict(zip(self.output_names, raw_outputs))

        label = outputs["label"]
        if label.shape[1] == NUM_CLASSES + 1:
            cls_logits = label[:, :NUM_CLASSES]
            iou_pred = float(sigmoid(label[:, NUM_CLASSES])[0])
        else:
            cls_logits = label
            iou_pred = 1.0

        cls_probs = softmax(cls_logits)[0]
        pred_class_idx = int(np.argmax(cls_probs))
        detections = decode_predictions(
            outputs["obj_small"][0],
            outputs["bbox_small"][0],
            outputs["obj_large"][0],
            outputs["bbox_large"][0],
            iou_pred=iou_pred,
            conf_threshold=conf_threshold,
        )
        rgb_present = bool(np.any(image_rgb))
        thermal_present = bool(np.any(image_thermal))
        fallback_class_idx = CLASS_MAP.get("person_visible")
        if pred_class_idx == CLASS_MAP["background"]:
            fallback_ok = (
                thermal_present and not rgb_present and detections
                and fallback_class_idx is not None
                and float(cls_probs[fallback_class_idx]) >= 0.15
            )
            if fallback_ok:
                pred_class_idx = fallback_class_idx
            else:
                return []

        h_orig, w_orig = image_thermal.shape[:2]
        detected_class_name = INV_CLASS_MAP.get(pred_class_idx, "unknown")
        class_conf = float(cls_probs[pred_class_idx])
        gate_w = None
        if "w_rgb" in outputs and "w_thm" in outputs:
            gate_w = {
                "rgb": float(outputs["w_rgb"][0].mean()),
                "thm": float(outputs["w_thm"][0].mean()),
            }

        for det in detections:
            det["bbox"] = [float(value) for value in det["bbox"]]
            det["class"] = detected_class_name
            det["combined_conf"] = (det["conf"] + class_conf) / 2.0
            det["gate_w"] = gate_w

            x1, y1, x2, y2 = det["bbox"]
            left = max(0, min(w_orig, int(x1 * w_orig)))
            top = max(0, min(h_orig, int(y1 * h_orig)))
            right = max(0, min(w_orig, int(x2 * w_orig)))
            bottom = max(0, min(h_orig, int(y2 * h_orig)))

            if right > left and bottom > top:
                crop = image_thermal[top:bottom, left:right]
                max_val = float(np.max(crop))
                det["thermal_peak"] = round(max_val / 255.0, 3)
                det["thermal_score"] = det["thermal_peak"]
            else:
                det["thermal_peak"] = 0.0
                det["thermal_score"] = 0.0

        detections = filter_by_laplacian(
            lap_image,
            detections,
            lap_thresh,
            thermal_present=bool(np.any(image_thermal)),
            rgb_present=bool(np.any(image_rgb)),
        )
        if merge_boxes:
            detections = merge_related_detections(detections)
        return detections


def get_engine():
    global _engine

    with _engine_lock:
        if _engine is None:
            if not MODEL_PATH.exists():
                raise RuntimeError(f"Model file missing: {MODEL_PATH}")
            _engine = OnnxMicroGhostEngine(MODEL_PATH)
        return _engine


def resize_for_runtime(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= MAX_IMAGE_SIDE:
        return image

    scale = MAX_IMAGE_SIDE / float(longest_side)
    next_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, next_size, interpolation=cv2.INTER_AREA)


async def read_image(upload: Optional[UploadFile], mode: int):
    if upload is None:
        return None

    data = await upload.read()
    if not data:
        return None
    if len(data) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"{upload.filename} is larger than {limit_mb} MB.")

    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, mode)
    if image is None:
        raise HTTPException(status_code=400, detail=f"Could not decode {upload.filename}.")
    return resize_for_runtime(image)


def get_runtime_snapshot():
    if psutil is None or _process is None:
        return {
            "cpu_percent": None,
            "cpu_frequency_mhz": None,
            "process_memory_gb": None,
            "system_memory_percent": None,
            "system_memory_used_gb": None,
            "system_memory_total_gb": None,
            "thread_count": threading.active_count(),
            "cpu_count": os.cpu_count(),
        }

    memory = psutil.virtual_memory()
    rss_bytes = _process.memory_info().rss
    cpu_freq = psutil.cpu_freq()
    return {
        "cpu_percent": round(psutil.cpu_percent(interval=None), 1),
        "cpu_frequency_mhz": round(cpu_freq.current, 1) if cpu_freq else None,
        "process_memory_gb": round(rss_bytes / (1024 ** 3), 3),
        "system_memory_percent": round(memory.percent, 1),
        "system_memory_used_gb": round(memory.used / (1024 ** 3), 2),
        "system_memory_total_gb": round(memory.total / (1024 ** 3), 2),
        "thread_count": _process.num_threads(),
        "cpu_count": psutil.cpu_count(logical=True),
    }


def encode_jpeg_data_url(image_bgr: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode result image.")
    encoded = base64.b64encode(buffer).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def draw_detections(image_bgr: np.ndarray, detections) -> np.ndarray:
    output = image_bgr.copy()
    h, w = output.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        left = max(0, min(w - 1, int(x1 * w)))
        top = max(0, min(h - 1, int(y1 * h)))
        right = max(0, min(w - 1, int(x2 * w)))
        bottom = max(0, min(h - 1, int(y2 * h)))

        cv2.rectangle(output, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"{det.get('combined_conf', det.get('conf', 0.0)):.2f}"
        cv2.putText(
            output,
            label,
            (left, max(top - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return output


def normalize_detection(det):
    return {
        "class": det.get("class", "unknown"),
        "confidence": float(det.get("combined_conf", det.get("conf", 0.0))),
        "objectness": float(det.get("conf", 0.0)),
        "thermal_score": float(det.get("thermal_score", det.get("thermal_peak", 0.0))),
        "laplacian_variance": float(det.get("lap_var", 0.0)),
        "bbox": [float(value) for value in det["bbox"]],
        "merged_parts": int(det.get("merged_parts", 1)),
    }


@app.get("/")
def health():
    return {
        "ok": True,
        "service": "microghost-vercel-backend",
        "runtime": "onnxruntime",
        "model_exists": MODEL_PATH.exists(),
    }


@app.post("/")
@app.post("/api/analyze")
async def analyze(
    rgb_image: Optional[UploadFile] = File(default=None),
    thermal_image: Optional[UploadFile] = File(default=None),
    conf_thresh: Optional[float] = Form(default=None),
    lap_thresh: float = Form(default=DEFAULT_LAPLACIAN_THRESHOLD),
    merge_boxes: bool = Form(default=False),
):
    start = time.perf_counter()
    process_start = time.process_time()
    _ = get_runtime_snapshot()

    rgb_bgr = await read_image(rgb_image, cv2.IMREAD_COLOR)
    thermal_gray = await read_image(thermal_image, cv2.IMREAD_GRAYSCALE)

    if rgb_bgr is None and thermal_gray is None:
        raise HTTPException(status_code=400, detail="Upload an RGB image, a thermal image, or both.")

    if rgb_bgr is not None and thermal_gray is not None:
        mode = "paired"
        rgb_for_model = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        thermal_for_model = thermal_gray
        primary_bgr = rgb_bgr
        lap_image = rgb_for_model
        effective_conf = conf_thresh
        effective_lap = lap_thresh
    elif thermal_gray is not None:
        mode = "thermal_only"
        h, w = thermal_gray.shape[:2]
        rgb_for_model = np.zeros((h, w, 3), dtype=np.uint8)
        thermal_for_model = thermal_gray
        primary_bgr = cv2.cvtColor(thermal_gray, cv2.COLOR_GRAY2BGR)
        lap_image = thermal_gray
        effective_conf = THERMAL_ONLY_CONFIDENCE_THRESHOLD if conf_thresh is None else conf_thresh
        effective_lap = lap_thresh
    else:
        mode = "rgb_only"
        rgb_for_model = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        thermal_for_model = build_rgb_thermal_proxy(rgb_bgr)
        primary_bgr = rgb_bgr
        lap_image = rgb_for_model
        effective_conf = RGB_ONLY_CONFIDENCE_THRESHOLD if conf_thresh is None else conf_thresh
        effective_lap = lap_thresh

    try:
        engine = get_engine()
        detections = engine.detect_confirmed(
            rgb_for_model,
            thermal_for_model,
            lap_image=lap_image,
            lap_thresh=effective_lap,
            conf_threshold=effective_conf,
            merge_boxes=merge_boxes,
        )
        if mode == "thermal_only":
            detections = filter_thermal_only_artifacts(detections)
        elif mode == "rgb_only":
            detections = filter_rgb_only_detections(detections)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    annotated_primary = draw_detections(primary_bgr, detections)
    images = {"annotated_primary": encode_jpeg_data_url(annotated_primary)}

    if thermal_gray is not None:
        thermal_vis = cv2.applyColorMap(thermal_gray, cv2.COLORMAP_JET)
        images["annotated_thermal"] = encode_jpeg_data_url(draw_detections(thermal_vis, detections))

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    process_cpu_ms = (time.process_time() - process_start) * 1000
    cpu_count = os.cpu_count() or 1
    cpu_load_estimate = min(100.0, (process_cpu_ms / max(elapsed_ms, 1)) * 100.0 / cpu_count)
    runtime_after = get_runtime_snapshot()

    return {
        "ok": True,
        "mode": mode,
        "count": len(detections),
        "elapsed_ms": elapsed_ms,
        "thresholds": {
            "confidence": effective_conf,
            "laplacian": lap_thresh,
            "effective_laplacian": effective_lap,
            "lap_bypass_confidence": None,
            "merge_boxes": merge_boxes,
        },
        "performance": {
            "wall_time_ms": elapsed_ms,
            "process_cpu_time_ms": int(process_cpu_ms),
            "estimated_cpu_load_percent": round(cpu_load_estimate, 1),
            "cpu_percent": runtime_after["cpu_percent"],
            "cpu_frequency_mhz": runtime_after["cpu_frequency_mhz"],
            "process_memory_gb": runtime_after["process_memory_gb"],
            "system_memory_percent": runtime_after["system_memory_percent"],
            "system_memory_used_gb": runtime_after["system_memory_used_gb"],
            "system_memory_total_gb": runtime_after["system_memory_total_gb"],
            "thread_count": runtime_after["thread_count"],
            "cpu_count": runtime_after["cpu_count"],
            "model_runtime": "onnxruntime-cpu",
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
            "max_image_side_px": MAX_IMAGE_SIDE,
        },
        "detections": [normalize_detection(det) for det in detections],
        "images": images,
    }
