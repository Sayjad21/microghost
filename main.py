"""
MicroGhost-Thermal: Main Entry Point
======================================
CLI orchestration for the ESP32-S3 thermal intrusion detection system.
"""

import os
import argparse
import torch
import numpy as np

from config import (
    ACTIVE_DATASET, DATASET_ROOT, MODEL_SAVE_DIR, EXPORT_DIR, LOG_DIR,
    BEST_MODEL_PATH, ONNX_PATH, TFLITE_FP16_PATH, get_dataset_path, NUM_ANCHORS
)
from data_loading import create_dataloaders
from preprocessing import ThermalPreprocessor, analyze_dataset_anchors, GridEncoder
from model import MicroGhostThermal, print_model_analysis
from training import Trainer, plot_training_history
from inference import ThermalInferenceEngine, benchmark_model, export_to_onnx, export_to_tflite
from evaluation import run_detection_evaluation

def parse_args():
    parser = argparse.ArgumentParser(description="MicroGhost-Thermal Intrusion Detection")
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # Train
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--dataset', type=str, default=ACTIVE_DATASET,
                              choices=['llvip', 'kaist', 'flirv2'], help='Dataset to use')
    train_parser.add_argument('--data-root', type=str, default=None,
                              help='Override dataset directory (default: data/<dataset>/)')
    train_parser.add_argument('--epochs', type=int, default=None, help='Override config epochs')
    train_parser.add_argument('--batch-size', type=int, default=None, help='Override config batch size')
    train_parser.add_argument('--num-workers', type=int, default=None, help='Override config num workers (CPU scaling)')
    train_parser.add_argument('--no-kmeans', action='store_true', help='Skip K-Means anchor optimization')

    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model-path', type=str, default=BEST_MODEL_PATH)
    eval_parser.add_argument('--dataset', type=str, default=ACTIVE_DATASET,
                             choices=['llvip', 'kaist', 'flirv2'])
    eval_parser.add_argument('--data-root', type=str, default=None)
    eval_parser.add_argument('--conf-threshold', type=float, default=None,
                             help='Objectness confidence threshold (default: config)')
    eval_parser.add_argument('--limit', type=int, default=None,
                             help='Evaluate only first N val images (quick test)')

    # Diagnose
    diagnose_parser = subparsers.add_parser('diagnose', help='Run deep error profiling and save images')
    diagnose_parser.add_argument('--dataset', type=str, default=ACTIVE_DATASET,
                                 choices=['llvip', 'kaist', 'flirv2'])
    diagnose_parser.add_argument('--data-root', type=str, default=None)

    # Infer
    infer_parser = subparsers.add_parser('infer', help='Run inference on an image')
    infer_parser.add_argument('--image-rgb', type=str, required=True, help='Path to RGB image')
    infer_parser.add_argument('--image-thermal', type=str, required=True, help='Path to thermal image')
    infer_parser.add_argument('--model-path', type=str, default=BEST_MODEL_PATH)
    # ~line 58 — infer subparser
    infer_parser.add_argument('--num-anchors', type=int, default=None,help='Override num_anchors (use 2 for v1 checkpoint)')

    # Export
    export_parser = subparsers.add_parser('export', help='Export model to ONNX/TFLite')
    export_parser.add_argument('--model-path', type=str, default=BEST_MODEL_PATH)
    export_parser.add_argument('--format', type=str, choices=['onnx', 'tflite'], default='tflite')

    # Benchmark
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark model performance/size')
    bench_parser.add_argument('--model-path', type=str, default=None, help='Load specific model (default: uninitialized)')

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    if args.mode == 'train':
        dataset_root = args.data_root or get_dataset_path(args.dataset)
        print(f"Initializing training pipeline with dataset: {args.dataset}")
        print(f"Dataset path: {dataset_root}")

        # 1. Base preprocessor
        encoder = GridEncoder()
        preprocessor = ThermalPreprocessor(encoder=encoder)

        # 2. Dataloaders
        train_loader, val_loader = create_dataloaders(
            dataset_name=args.dataset,
            preprocessor=preprocessor,
            dataset_root=dataset_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        # 3. K-Means Anchor Optimization
        if not args.no_kmeans:
            # We access the raw base dataset for K-Means
            opt_ratios, opt_sizes = analyze_dataset_anchors(train_loader.dataset.base_dataset, num_anchors=NUM_ANCHORS)
            encoder.update_anchors(opt_ratios, opt_sizes)
            # Re-initialize dataloaders with updated encoder targets
            train_loader, val_loader = create_dataloaders(
                dataset_name=args.dataset,
                dataset_root=dataset_root,
                preprocessor=preprocessor,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
        else:
            opt_sizes = None

        # 4. Model & Trainer
        model = MicroGhostThermal()

        # --- TARGET HARDWARE PROFILING (ESP32-S3) ---
        print("\n" + "="*60)
        print("  TARGET HARDWARE PROFILING (ESP32-S3)")
        print("="*60)
        from model import print_model_analysis
        from inference import benchmark_model
        
        print_model_analysis(model)
        benchmark_model(model)
        print("="*60 + "\n")
        # --------------------------------------------

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            anchor_sizes=opt_sizes
        )

        # 5. Train
        history = trainer.fit(save_path=BEST_MODEL_PATH)
        plot_training_history(history)

    elif args.mode == 'benchmark':
        if args.model_path and os.path.exists(args.model_path):
            from inference import load_inference_model
            model = load_inference_model(args.model_path)
            print("Loaded trained model for benchmark.")
        else:
            model = MicroGhostThermal()
            print("Using newly initialized model for benchmark.")

        print_model_analysis(model)
        benchmark_model(model)

    elif args.mode == 'export':
        from inference import load_inference_model
        model = load_inference_model(args.model_path)

        if args.format == 'onnx' or args.format == 'tflite':
            export_to_onnx(model, ONNX_PATH)

        if args.format == 'tflite':
            # Note: For real FP16 quantization, you don't necessarily need calibration
            # but it is usually exported directly via TFLite converter standard FP16 flag.
            export_to_tflite(ONNX_PATH, TFLITE_FP16_PATH, int8=False) # int8=False for fp16 target handling in future implementation

    elif args.mode == 'infer':
        import cv2
        engine = ThermalInferenceEngine(model_path=args.model_path, override_num_anchors=args.num_anchors)
        img_bgr = cv2.imread(args.image_rgb, cv2.IMREAD_COLOR)       # ← keep BGR for saving
        img_thermal = cv2.imread(args.image_thermal, cv2.IMREAD_GRAYSCALE)

        if img_bgr is None or img_thermal is None:
            print("Error: Could not load one or both images.")
            return

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        detections = engine.detect_confirmed(img_rgb, img_thermal, lap_thresh=70.0) 
        
#       what lap_thresh=80.0 does is:

#       For each thermal detection, crop that same bounding box region from the RGB image
#       Convert it to grayscale and run the Laplacian filter
#       Compute the variance of the result
#       variance < 80 → the crop is too smooth → likely a flat hot surface → reject
#       If variance ≥ 80 → the crop has real structure → likely a person → keep

#       A car bonnet is a smooth painted metal surface — almost no texture in RGB, so its Laplacian variance might be 10–30. A person has hair, face, clothing folds, limb edges — easily 80–200+.
#       The jump from 50 → 80 that fixed your case means the scooter/motorbike crop was sitting somewhere between 50 and 80 — enough texture to fool the looser threshold, but not enough to pass 80.
#       The tradeoff to be aware of: if someone is wearing a plain single-colour outfit and standing far away (small box, low resolution crop), their variance could also dip below 80. If that becomes an issue during testing, you can combine it with the confidence score — e.g. only apply the Laplacian gate to boxes with combined_conf < 0.85, trusting very high confidence detections regardless.
        
        print("\n--- Detection Results ---")
        if not detections:
            print("No intrusion detected.")
        else:
            for i, det in enumerate(detections):
                print(f"[{i+1}] Intrusion! Conf: {det['combined_conf']:.2f} | "
                      f"Temp: {det.get('temp_c', 0)}°C | BBox: {det['bbox']}")

        # ── Draw & save ──────────────────────────────────────────────────────
        h, w = img_bgr.shape[:2]
        vis = img_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v * d) for v, d in zip(det['bbox'], [w, h, w, h])]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det.get('class', '?')} {det['combined_conf']:.2f} {det.get('temp_c', 0)}degC"
            cv2.putText(vis, label, (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        os.makedirs('runs', exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.image_rgb))[0]
        out_path = os.path.join('runs', f'{stem}_det.jpg')
        cv2.imwrite(out_path, vis)
        
        #defining a color map for thermal visualization
        colormap = cv2.COLORMAP_JET
        # ── Save thermal with boxes too ───────────────────────────
        #thermal_vis = cv2.cvtColor(img_thermal, cv2.COLOR_GRAY2BGR)
        thermal_vis = cv2.applyColorMap(img_thermal, colormap)
        for det in detections:
            x1, y1, x2, y2 = [int(v * d) for v, d in zip(det['bbox'], [w, h, w, h])]
            cv2.rectangle(thermal_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        out_thermal = os.path.join('runs', f'{stem}_thermal_det.jpg')
        cv2.imwrite(out_thermal, thermal_vis)
        print(f"Saved → {out_thermal}")
        # ──────────────────────────────────────────────────────────
        print(f"\nSaved → {out_path}")
        os.startfile(out_path)
        # ─────────────────────────────────────────────────────────────────────

    elif args.mode == 'evaluate':
        dataset_root = args.data_root or get_dataset_path(args.dataset)
        run_detection_evaluation(
            model_path=args.model_path,
            dataset_name=args.dataset,
            dataset_root=dataset_root,
            conf_threshold=args.conf_threshold,
            limit=args.limit,
        )

    elif args.mode == 'diagnose':
        from diagnose import run_visual_diagnostics
        dataset_root = args.data_root or get_dataset_path(args.dataset)
        run_visual_diagnostics(dataset_name=args.dataset, dataset_root=dataset_root)

    else:
        print("Please specify a mode. Use --help for options.")


if __name__ == '__main__':
    main()
