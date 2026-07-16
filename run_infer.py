import argparse
import os

import cv2
import numpy as np

from inference import ThermalInferenceEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Run MicroGhost inference on an RGB image, thermal image, or pair.")
    parser.add_argument("--image-rgb", default=None, help="Path to RGB image.")
    parser.add_argument("--image-thermal", default=None, help="Path to thermal image.")
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
    parser.add_argument(
        "--merge-boxes",
        action="store_true",
        help="Optionally merge nearby stacked boxes; off by default to avoid grouping adjacent people.",
    )
    parser.add_argument("--output", default=None, help="Optional annotated image output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.image_rgb and not args.image_thermal:
        raise ValueError("Provide --image-rgb, --image-thermal, or both.")

    engine = ThermalInferenceEngine(
        model_path=args.model_path,
        override_num_anchors=args.num_anchors,
    )

    if args.image_rgb:
        img_bgr = cv2.imread(args.image_rgb, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load RGB image: {args.image_rgb}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        lap_image = img_rgb
    else:
        print("No RGB image provided. Running thermal-only inference with a blank RGB branch.")
        img_rgb = None

    if args.image_thermal:
        img_thermal = cv2.imread(args.image_thermal, cv2.IMREAD_GRAYSCALE)
        if img_thermal is None:
            raise FileNotFoundError(f"Could not load thermal image: {args.image_thermal}")
    else:
        print("No thermal image provided. Running RGB-only inference with a blank thermal branch.")
        h_rgb, w_rgb = img_rgb.shape[:2]
        img_thermal = np.zeros((h_rgb, w_rgb), dtype=np.uint8)

    if img_rgb is None:
        h_thm, w_thm = img_thermal.shape[:2]
        img_rgb = np.zeros((h_thm, w_thm, 3), dtype=np.uint8)
        img_bgr = cv2.cvtColor(img_thermal, cv2.COLOR_GRAY2BGR)
        lap_image = img_thermal

    single_modality = not (args.image_rgb and args.image_thermal)
    conf_thresh = args.conf_thresh
    if conf_thresh is None and single_modality:
        conf_thresh = 0.20
    auto_merge = (
        args.merge_boxes
        or (single_modality and args.image_thermal and not args.image_rgb and conf_thresh <= 0.15)
    )
    if auto_merge and not args.merge_boxes:
        print("Low-threshold thermal-only inference: consolidating stacked duplicate boxes.")

    detections = engine.detect_confirmed(
        img_rgb,
        img_thermal,
        lap_image=lap_image,
        lap_thresh=args.lap_thresh,
        high_conf_bypass=args.lap_bypass_conf,
        conf_threshold=conf_thresh,
        merge_boxes=auto_merge,
    )

    if not detections:
        print("No intrusion detected.")
        return

    for i, det in enumerate(detections):
        print(
            f"[{i + 1}] {det['class']} | conf={det['combined_conf']:.2f} "
            f"| thermal={det.get('thermal_score', 0.0):.2f} "
            f"| lap_var={det.get('lap_var', 0.0):.1f} | bbox={det['bbox']}"
        )

    if args.output:
        vis = img_bgr.copy()
        h, w = vis.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = [int(v * d) for v, d in zip(det['bbox'], [w, h, w, h])]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['combined_conf']:.2f}"
            cv2.putText(vis, label, (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        cv2.imwrite(args.output, vis)
        print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
