import base64
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse


APP_DIR = Path(__file__).resolve().parent
CORE_DIR = APP_DIR / "microghost"
DEFAULT_MODEL_PATH = APP_DIR / "checkpoints" / "best_microghost_thermal_v3.pth"

sys.path.insert(0, str(CORE_DIR))

from inference import ThermalInferenceEngine  # noqa: E402


api = FastAPI(
    title="MicroGhost Thermal Inference",
    version="1.0.0",
    description="Free Gradio Space wrapper with a FastAPI /analyze endpoint.",
)

cors_origins = os.environ.get("CORS_ORIGINS", "*")
api.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins.split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_engine = None
_engine_lock = threading.Lock()


def get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            model_path = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
            if not model_path.exists():
                raise RuntimeError(f"Model checkpoint not found: {model_path}")
            _engine = ThermalInferenceEngine(model_path=str(model_path))
    return _engine


async def read_upload_image(upload: Optional[UploadFile], mode: int):
    if upload is None:
        return None

    data = await upload.read()
    if not data:
        return None

    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, mode)
    if image is None:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {upload.filename}")
    return image


def data_url(image_bgr, ext=".jpg"):
    ok, encoded = cv2.imencode(ext, image_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode result image.")
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


def run_analysis(rgb_bgr, thermal_gray, conf_thresh=None, lap_thresh=80.0, lap_bypass_conf=None):
    started = time.perf_counter()

    if rgb_bgr is None and thermal_gray is None:
        raise ValueError("Upload an RGB image, a thermal image, or both.")

    if rgb_bgr is not None and thermal_gray is not None:
        mode = "paired"
        model_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        model_thermal = thermal_gray
        lap_image = model_rgb
        primary_bgr = rgb_bgr
        effective_conf = conf_thresh
    elif thermal_gray is not None:
        mode = "thermal_only"
        h, w = thermal_gray.shape[:2]
        model_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        model_thermal = thermal_gray
        lap_image = thermal_gray
        primary_bgr = cv2.cvtColor(thermal_gray, cv2.COLOR_GRAY2BGR)
        effective_conf = 0.20 if conf_thresh is None else conf_thresh
    else:
        mode = "rgb_only"
        h, w = rgb_bgr.shape[:2]
        model_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        model_thermal = np.zeros((h, w), dtype=np.uint8)
        lap_image = model_rgb
        primary_bgr = rgb_bgr
        effective_conf = 0.20 if conf_thresh is None else conf_thresh

    engine = get_engine()
    detections = engine.detect_confirmed(
        model_rgb,
        model_thermal,
        lap_image=lap_image,
        lap_thresh=lap_thresh,
        high_conf_bypass=lap_bypass_conf,
        conf_threshold=effective_conf,
    )

    thermal_bgr = cv2.applyColorMap(model_thermal, cv2.COLORMAP_JET)
    annotated_primary = draw_detections(primary_bgr, detections)
    annotated_thermal = draw_detections(thermal_bgr, detections)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {
        "ok": True,
        "mode": mode,
        "count": len(detections),
        "elapsed_ms": elapsed_ms,
        "thresholds": {
            "confidence": effective_conf,
            "laplacian": lap_thresh,
            "lap_bypass_confidence": lap_bypass_conf,
        },
        "detections": [normalize_detection(det) for det in detections],
        "annotated_primary_bgr": annotated_primary,
        "annotated_thermal_bgr": annotated_thermal,
    }


@api.get("/")
def root():
    return RedirectResponse(url="/ui")


@api.get("/health")
def health():
    model_path = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    return {
        "ok": True,
        "sdk": "gradio",
        "model_exists": model_path.exists(),
        "model_path": str(model_path),
    }


@api.post("/analyze")
async def analyze(
    rgb_image: Optional[UploadFile] = File(default=None),
    thermal_image: Optional[UploadFile] = File(default=None),
    conf_thresh: Optional[float] = Form(default=None),
    lap_thresh: float = Form(default=80.0),
    lap_bypass_conf: Optional[float] = Form(default=None),
):
    rgb_bgr = await read_upload_image(rgb_image, cv2.IMREAD_COLOR)
    thermal_gray = await read_upload_image(thermal_image, cv2.IMREAD_GRAYSCALE)

    try:
        result = run_analysis(
            rgb_bgr,
            thermal_gray,
            conf_thresh=conf_thresh,
            lap_thresh=lap_thresh,
            lap_bypass_conf=lap_bypass_conf,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "ok": True,
        "mode": result["mode"],
        "count": result["count"],
        "elapsed_ms": result["elapsed_ms"],
        "thresholds": result["thresholds"],
        "detections": result["detections"],
        "images": {
            "annotated_primary": data_url(result["annotated_primary_bgr"]),
            "annotated_thermal": data_url(result["annotated_thermal_bgr"]),
        },
    }


def gradio_analyze(rgb_image, thermal_image, conf_thresh, lap_thresh):
    rgb_bgr = None
    thermal_gray = None

    if rgb_image is not None:
        rgb_bgr = cv2.cvtColor(rgb_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    if thermal_image is not None:
        if thermal_image.ndim == 3:
            thermal_gray = cv2.cvtColor(thermal_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            thermal_gray = thermal_image.astype(np.uint8)

    result = run_analysis(
        rgb_bgr,
        thermal_gray,
        conf_thresh=None if conf_thresh <= 0 else conf_thresh,
        lap_thresh=lap_thresh,
    )

    primary_rgb = cv2.cvtColor(result["annotated_primary_bgr"], cv2.COLOR_BGR2RGB)
    thermal_rgb = cv2.cvtColor(result["annotated_thermal_bgr"], cv2.COLOR_BGR2RGB)
    summary = {
        "mode": result["mode"],
        "count": result["count"],
        "elapsed_ms": result["elapsed_ms"],
        "thresholds": result["thresholds"],
        "detections": result["detections"],
    }
    return primary_rgb, thermal_rgb, summary


with gr.Blocks(title="MicroGhost Thermal Inference") as demo:
    gr.Markdown("# MicroGhost Thermal Inference")
    gr.Markdown("Upload RGB, thermal, or both. The Vercel frontend uses the same `/analyze` endpoint.")
    with gr.Row():
        rgb_input = gr.Image(label="RGB image", type="numpy", image_mode="RGB")
        thermal_input = gr.Image(label="Thermal image", type="numpy", image_mode="L")
    with gr.Accordion("Advanced tuning", open=False):
        conf_input = gr.Slider(0, 0.9, value=0, step=0.01, label="Confidence override (0 = automatic)")
        lap_input = gr.Slider(0, 220, value=80, step=5, label="Laplacian threshold")
    analyze_button = gr.Button("Analyze", variant="primary")
    with gr.Row():
        primary_output = gr.Image(label="Primary annotated output")
        thermal_output = gr.Image(label="Thermal annotated output")
    json_output = gr.JSON(label="Detections")

    analyze_button.click(
        gradio_analyze,
        inputs=[rgb_input, thermal_input, conf_input, lap_input],
        outputs=[primary_output, thermal_output, json_output],
        api_name="gradio_analyze",
    )


app = gr.mount_gradio_app(api, demo, path="/ui")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
