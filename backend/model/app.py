import base64
import sys
import threading
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(*args, **kwargs):
            def decorator(func):
                return func
            return decorator


APP_DIR = Path(__file__).resolve().parent
CORE_DIR = APP_DIR / "microghost"
DEFAULT_MODEL_PATH = APP_DIR / "checkpoints" / "best_microghost_thermal_v3.pth"

sys.path.insert(0, str(CORE_DIR))

from inference import ThermalInferenceEngine  # noqa: E402


_engine = None
_engine_lock = threading.Lock()


def get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            if not DEFAULT_MODEL_PATH.exists():
                raise RuntimeError(f"Model checkpoint not found: {DEFAULT_MODEL_PATH}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _engine = ThermalInferenceEngine(model_path=str(DEFAULT_MODEL_PATH), device=device)
    return _engine


def data_url(image_bgr, ext=".jpg"):
    ok, encoded = cv2.imencode(ext, image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode result image.")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{payload}"


def draw_detections(base_bgr, detections):
    out = base_bgr.copy()
    h, w = out.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = [int(float(v) * d) for v, d in zip(det["bbox"], [w, h, w, h])]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), (31, 214, 128), 2)
        label = f"{det.get('class', 'target')} {det.get('combined_conf', 0):.2f}"
        cv2.putText(
            out,
            label,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (31, 214, 128),
            2,
            cv2.LINE_AA,
        )
    return out


def normalize_detection(det):
    return {
        "class": det.get("class", "unknown"),
        "confidence": round(float(det.get("combined_conf", det.get("conf", 0.0))), 4),
        "objectness": round(float(det.get("conf", 0.0)), 4),
        "temperature_c": float(det.get("temp_c", 0.0)),
        "laplacian_variance": float(det.get("lap_var", 0.0)),
        "bbox": [round(float(v), 6) for v in det.get("bbox", [])],
        "merged_parts": int(det.get("merged_parts", 1)),
    }


def prepare_inputs(rgb_image, thermal_image, conf_thresh):
    rgb_bgr = None
    thermal_gray = None

    if rgb_image is not None:
        rgb_array = rgb_image.astype(np.uint8)
        rgb_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    if thermal_image is not None:
        thermal_array = thermal_image.astype(np.uint8)
        if thermal_array.ndim == 3:
            thermal_gray = cv2.cvtColor(thermal_array, cv2.COLOR_RGB2GRAY)
        else:
            thermal_gray = thermal_array

    if rgb_bgr is None and thermal_gray is None:
        raise gr.Error("Upload an RGB image, a thermal image, or both.")

    if rgb_bgr is not None and thermal_gray is not None:
        return {
            "mode": "paired",
            "model_rgb": cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB),
            "model_thermal": thermal_gray,
            "lap_image": cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB),
            "primary_bgr": rgb_bgr,
            "effective_conf": None if conf_thresh <= 0 else conf_thresh,
        }

    if thermal_gray is not None:
        h, w = thermal_gray.shape[:2]
        return {
            "mode": "thermal_only",
            "model_rgb": np.zeros((h, w, 3), dtype=np.uint8),
            "model_thermal": thermal_gray,
            "lap_image": thermal_gray,
            "primary_bgr": cv2.cvtColor(thermal_gray, cv2.COLOR_GRAY2BGR),
            "effective_conf": 0.20 if conf_thresh <= 0 else conf_thresh,
        }

    h, w = rgb_bgr.shape[:2]
    return {
        "mode": "rgb_only",
        "model_rgb": cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB),
        "model_thermal": np.zeros((h, w), dtype=np.uint8),
        "lap_image": cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB),
        "primary_bgr": rgb_bgr,
        "effective_conf": 0.20 if conf_thresh <= 0 else conf_thresh,
    }


@spaces.GPU(duration=60)
def gradio_analyze(rgb_image, thermal_image, conf_thresh, lap_thresh):
    started = time.perf_counter()
    inputs = prepare_inputs(rgb_image, thermal_image, float(conf_thresh),)

    engine = get_engine()
    detections = engine.detect_confirmed(
        inputs["model_rgb"],
        inputs["model_thermal"],
        lap_image=inputs["lap_image"],
        lap_thresh=float(lap_thresh),
        conf_threshold=inputs["effective_conf"],
    )

    thermal_bgr = cv2.applyColorMap(inputs["model_thermal"], cv2.COLORMAP_JET)
    annotated_primary = draw_detections(inputs["primary_bgr"], detections)
    annotated_thermal = draw_detections(thermal_bgr, detections)

    return {
        "ok": True,
        "mode": inputs["mode"],
        "count": len(detections),
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "thresholds": {
            "confidence": inputs["effective_conf"],
            "laplacian": float(lap_thresh),
            "lap_bypass_confidence": None,
        },
        "detections": [normalize_detection(det) for det in detections],
        "images": {
            "annotated_primary": data_url(annotated_primary),
            "annotated_thermal": data_url(annotated_thermal),
        },
    }


with gr.Blocks(title="MicroGhost Thermal Inference") as demo:
    gr.Markdown("# MicroGhost Thermal Inference")
    gr.Markdown("Upload RGB, thermal, or both. Use this Space directly, or call it from the Vercel app.")
    with gr.Row():
        rgb_input = gr.Image(label="RGB image", type="numpy", image_mode="RGB")
        thermal_input = gr.Image(label="Thermal image", type="numpy", image_mode="L")
    with gr.Accordion("Advanced tuning", open=False):
        conf_input = gr.Slider(0, 0.9, value=0, step=0.01, label="Confidence override (0 = automatic)")
        lap_input = gr.Slider(0, 220, value=80, step=5, label="Laplacian threshold")
    analyze_button = gr.Button("Analyze", variant="primary")
    json_output = gr.JSON(label="Result")

    analyze_button.click(
        gradio_analyze,
        inputs=[rgb_input, thermal_input, conf_input, lap_input],
        outputs=[json_output],
        api_name="gradio_analyze",
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
