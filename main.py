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
    BEST_MODEL_PATH, ONNX_PATH, TFLITE_FP16_PATH, get_dataset_path, NUM_ANCHORS,
    TRAIN_INPUT_SIZE, FINETUNE_LEARNING_RATE, MIXED_CAMO_SAMPLE_WEIGHT,
    resolve_input_size,
)
from data_loading import create_dataloaders, create_mixed_dataloaders
from preprocessing import ThermalPreprocessor, analyze_dataset_anchors, GridEncoder, make_preprocessor
from model import MicroGhostThermal, print_model_analysis
from training import Trainer, plot_training_history, freeze_backbone, load_pretrained_weights
from inference import ThermalInferenceEngine, benchmark_model, export_to_onnx, export_to_tflite
from evaluation import run_detection_evaluation

_DATASET_CHOICES = ['llvip', 'kaist', 'flirv2', 'camo_m3fd', 'mixed']


def parse_args():
    parser = argparse.ArgumentParser(description="MicroGhost-Thermal Intrusion Detection")
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # Train
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--dataset', type=str, default=ACTIVE_DATASET,
                              choices=_DATASET_CHOICES, help='Dataset to use')
    train_parser.add_argument('--data-root', type=str, default=None,
                              help='Override dataset directory (default: data/<dataset>/)')
    train_parser.add_argument('--camo-root', type=str, default=None,
                              help='CAMO-M3FD path for mixed training')
    train_parser.add_argument('--epochs', type=int, default=None, help='Override config epochs')
    train_parser.add_argument('--batch-size', type=int, default=None, help='Override config batch size')
    train_parser.add_argument('--no-kmeans', action='store_true', help='Skip K-Means anchor optimization')
    train_parser.add_argument('--save-path', type=str, default=None,
                              help='Checkpoint path (default: config BEST_MODEL_PATH)')
    train_parser.add_argument('--init-weights', type=str, default=None,
                              help='Load weights from checkpoint before training (fine-tune)')
    train_parser.add_argument('--lr', type=float, default=None,
                              help='Learning rate (default: 1e-3 train, 1e-4 with --init-weights)')
    train_parser.add_argument('--input-size', type=int, nargs=2, metavar=('H', 'W'),
                              default=None,
                              help='Input resolution H W (default: 160x128; mixed fine-tune: 256 320)')
    train_parser.add_argument('--freeze-backbone', action='store_true',
                              help='Freeze stem/backbone/FPN; train heads only')
    train_parser.add_argument('--camo-weight', type=float, default=None,
                              help='CAMO fraction in mixed training (default: 0.30)')

    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model-path', type=str, default=BEST_MODEL_PATH)
    eval_parser.add_argument('--dataset', type=str, default=ACTIVE_DATASET,
                             choices=_DATASET_CHOICES[:-1])  # no 'mixed' eval
    eval_parser.add_argument('--data-root', type=str, default=None)
    eval_parser.add_argument('--conf-threshold', type=float, default=None,
                             help='Objectness confidence threshold (default: config)')
    eval_parser.add_argument('--limit', type=int, default=None,
                             help='Evaluate only first N val images (quick test)')

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


def _resolve_train_input_size(args):
    if args.input_size:
        return tuple(args.input_size)
    if args.dataset == 'mixed' or args.init_weights:
        return TRAIN_INPUT_SIZE
    return None


def main():
    args = parse_args()
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    if args.mode == 'train':
        input_size = _resolve_train_input_size(args)
        train_h, train_w = resolve_input_size(input_size)
        lr = args.lr
        if lr is None:
            lr = FINETUNE_LEARNING_RATE if args.init_weights else None

        print(f"Training input size: {train_h}×{train_w}")
        if lr:
            print(f"Learning rate: {lr}")

        encoder = GridEncoder(input_size=input_size)
        preprocessor = make_preprocessor(input_size=input_size, encoder=encoder)

        if args.dataset == 'mixed':
            llvip_root = args.data_root or get_dataset_path('llvip')
            camo_root = args.camo_root or get_dataset_path('camo_m3fd')
            print(f"Mixed training: LLVIP={llvip_root}, CAMO={camo_root}")
            train_loader, val_loader = create_mixed_dataloaders(
                preprocessor=preprocessor,
                llvip_root=llvip_root,
                camo_root=camo_root,
                batch_size=args.batch_size,
                camo_weight=args.camo_weight,
            )
            opt_sizes = None
        else:
            dataset_root = args.data_root or get_dataset_path(args.dataset)
            print(f"Dataset: {args.dataset} @ {dataset_root}")
            train_loader, val_loader = create_dataloaders(
                dataset_name=args.dataset,
                preprocessor=preprocessor,
                dataset_root=dataset_root,
                batch_size=args.batch_size,
            )

            if not args.no_kmeans:
                base = getattr(train_loader.dataset, 'base_dataset', train_loader.dataset)
                if hasattr(base, 'datasets'):
                    base = base.datasets[0].base_dataset
                opt_ratios, opt_sizes = analyze_dataset_anchors(base, num_anchors=NUM_ANCHORS)
                encoder.update_anchors(opt_ratios, opt_sizes)
                train_loader, val_loader = create_dataloaders(
                    dataset_name=args.dataset,
                    dataset_root=dataset_root,
                    preprocessor=preprocessor,
                    batch_size=args.batch_size,
                )
            else:
                opt_sizes = None

        model = MicroGhostThermal(input_size=(train_h, train_w))
        if args.init_weights:
            load_pretrained_weights(model, args.init_weights)
            print(f"Fine-tuning from: {args.init_weights}")
        if args.freeze_backbone:
            freeze_backbone(model, freeze=True)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            anchor_sizes=opt_sizes,
            lr=lr,
            input_size=(train_h, train_w),
        )

        save_path = args.save_path or BEST_MODEL_PATH
        history = trainer.fit(save_path=save_path)
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
            export_to_tflite(ONNX_PATH, TFLITE_FP16_PATH, int8=False)

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
        dataset_root = args.data_root or get_dataset_path(args.dataset)
        run_detection_evaluation(
            model_path=args.model_path,
            dataset_name=args.dataset,
            dataset_root=dataset_root,
            conf_threshold=args.conf_threshold,
            limit=args.limit,
        )

    else:
        print("Please specify a mode. Use --help for options.")


if __name__ == '__main__':
    main()
