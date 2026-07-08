import argparse

import cv2
import numpy as np

from inference import ThermalInferenceEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Run MicroGhost inference on one RGB/thermal image pair.")
    parser.add_argument("--image-rgb", default=None, help="Path to RGB image.")
    parser.add_argument("--image-thermal", required=True, help="Path to thermal image.")
    parser.add_argument(
        "--model-path",
        default="checkpoints/best_microghost_thermal_v3.pth",
        help="Path to a trained checkpoint.",
    )
    parser.add_argument("--num-anchors", type=int, default=None, help="Optional anchor override.")
    parser.add_argument(
        "--conf-thresh",
        "--conf-threshold",
        dest="conf_thresh",
        type=float,
        default=None,
        help="Override detection confidence threshold; thermal-only defaults to 0.20.",
    )
    parser.add_argument(
        "--lap-thresh",
        "--lap-thres",
        dest="lap_thresh",
        type=float,
        default=80.0,
        help="Reject detections with RGB crop Laplacian variance below this value.",
    )
    parser.add_argument(
        "--lap-bypass-conf",
        type=float,
        default=None,
        help="Optional: skip the Laplacian gate for detections at or above this confidence.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    engine = ThermalInferenceEngine(
        model_path=args.model_path,
        override_num_anchors=args.num_anchors,
    )

    img_thermal = cv2.imread(args.image_thermal, cv2.IMREAD_GRAYSCALE)

    if img_thermal is None:
        raise FileNotFoundError(f"Could not load thermal image: {args.image_thermal}")

    if args.image_rgb:
        img_bgr = cv2.imread(args.image_rgb, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load RGB image: {args.image_rgb}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        lap_image = img_rgb
        conf_thresh = args.conf_thresh
    else:
        print("No RGB image provided. Running thermal-only inference with a blank RGB branch.")
        h_thm, w_thm = img_thermal.shape[:2]
        img_rgb = np.zeros((h_thm, w_thm, 3), dtype=np.uint8)
        lap_image = img_thermal
        conf_thresh = 0.20 if args.conf_thresh is None else args.conf_thresh

    detections = engine.detect_confirmed(
        img_rgb,
        img_thermal,
        lap_image=lap_image,
        lap_thresh=args.lap_thresh,
        high_conf_bypass=args.lap_bypass_conf,
        conf_threshold=conf_thresh,
    )

    if not detections:
        print("No intrusion detected.")
        return

    for i, det in enumerate(detections):
        print(
            f"[{i + 1}] {det['class']} | conf={det['combined_conf']:.2f} "
            f"| temp={det.get('temp_c', 0)} degC "
            f"| lap_var={det.get('lap_var', 0.0):.1f} | bbox={det['bbox']}"
        )


if __name__ == "__main__":
    main()
