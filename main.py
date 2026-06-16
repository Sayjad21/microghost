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
    BEST_MODEL_PATH, ONNX_PATH, TFLITE_FP16_PATH, get_dataset_path
)
from data_loading import create_dataloaders
from preprocessing import ThermalPreprocessor, analyze_dataset_anchors, GridEncoder
from model import MicroGhostThermal, print_model_analysis
from training import Trainer, plot_training_history
from inference import ThermalInferenceEngine, benchmark_model, export_to_onnx, export_to_tflite

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
    train_parser.add_argument('--no-kmeans', action='store_true', help='Skip K-Means anchor optimization')

    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model-path', type=str, default=BEST_MODEL_PATH)
    eval_parser.add_argument('--dataset', type=str, default=ACTIVE_DATASET,
                             choices=['llvip', 'kaist', 'flirv2'])
    eval_parser.add_argument('--data-root', type=str, default=None)

    # Infer
    infer_parser = subparsers.add_parser('infer', help='Run inference on an image')
    infer_parser.add_argument('--image-rgb', type=str, required=True, help='Path to RGB image')
    infer_parser.add_argument('--image-thermal', type=str, required=True, help='Path to thermal image')
    infer_parser.add_argument('--model-path', type=str, default=BEST_MODEL_PATH)

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
            batch_size=args.batch_size
        )

        # 3. K-Means Anchor Optimization
        if not args.no_kmeans:
            # We access the raw base dataset for K-Means
            opt_ratios, opt_sizes = analyze_dataset_anchors(train_loader.dataset.base_dataset)
            encoder.update_anchors(opt_ratios, opt_sizes)
            # Re-initialize dataloaders with updated encoder targets
            train_loader, val_loader = create_dataloaders(
                dataset_name=args.dataset,
                dataset_root=dataset_root,
                preprocessor=preprocessor,
                batch_size=args.batch_size
            )
        else:
            opt_sizes = None

        # 4. Model & Trainer
        model = MicroGhostThermal()
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
        engine = ThermalInferenceEngine(model_path=args.model_path)
        img_rgb = cv2.imread(args.image_rgb, cv2.IMREAD_COLOR)
        img_thermal = cv2.imread(args.image_thermal, cv2.IMREAD_GRAYSCALE)
        
        if img_rgb is None or img_thermal is None:
            print("Error: Could not load one or both images.")
            return

        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        detections = engine.detect(img_rgb, img_thermal)
        print("\n--- Detection Results ---")
        if not detections:
            print("No intrusion detected.")
        else:
            for i, det in enumerate(detections):
                print(f"[{i+1}] Intrusion! Conf: {det['combined_conf']:.2f} | "
                      f"Temp: {det.get('temp_c', 0)}°C | BBox: {det['bbox']}")

    elif args.mode == 'evaluate':
        print("Evaluation pipeline not fully implemented in CLI yet. Use training script outputs.")

    else:
        print("Please specify a mode. Use --help for options.")


if __name__ == '__main__':
    main()
