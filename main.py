"""
MicroGhost-Thermal: Main Entry Point (V2)
===========================================
CLI orchestration for the ESP32-S3 thermal intrusion detection system.
Supports 4-phase V2 training, dynamic evaluation, and edge inference.
"""

import os
import argparse
import torch
import numpy as np

from config import (
    ACTIVE_DATASET, DATASET_ROOT, MODEL_SAVE_DIR, EXPORT_DIR, LOG_DIR,
    BEST_MODEL_PATH, ONNX_PATH, TFLITE_FP16_PATH, get_dataset_path, NUM_ANCHORS,
    TRAINING_PHASES
)
from data_loading import create_dataloaders, create_phase_dataloaders
from preprocessing import ThermalPreprocessor, analyze_dataset_anchors, GridEncoder
from model import MicroGhostThermal, MicroGhostV2, print_model_analysis
from training import Trainer, PhaseTrainer, plot_training_history
from inference import ThermalInferenceEngine, benchmark_model, export_to_onnx
from evaluation import run_detection_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="MicroGhost-Thermal Intrusion Detection (V2)")
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # Train
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--v1', action='store_true', help='Train legacy V1 model (single phase)')
    train_parser.add_argument('--dataset', type=str, default=ACTIVE_DATASET,
                              choices=['llvip', 'kaist', 'flirv2', 'camod3fd'], help='Dataset for V1 training')
    train_parser.add_argument('--data-root', type=str, default=None,
                              help='Override dataset directory')
    train_parser.add_argument('--epochs', type=int, default=None, help='Override config epochs')
    train_parser.add_argument('--batch-size', type=int, default=None, help='Override config batch size')
    train_parser.add_argument('--num-workers', type=int, default=0, help='Override config num workers (CPU scaling)')
    train_parser.add_argument('--no-kmeans', action='store_true', help='Skip K-Means anchor optimization')
    train_parser.add_argument('--resume-v1', action='store_true', help='Fine-tune from V1 checkpoint')
    train_parser.add_argument('--phase', type=int, default=None, choices=[1, 2, 3, 4],
                              help='Run a specific V2 training phase (1-4). If omitted, runs all phases.')
    train_parser.add_argument('--debug', action='store_true', help='Run 1 toy epoch per phase for debugging')

    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model-path', type=str, default=BEST_MODEL_PATH)
    eval_parser.add_argument('--dataset', type=str, default=ACTIVE_DATASET,
                             choices=['llvip', 'kaist', 'flirv2', 'camod3fd'])
    eval_parser.add_argument('--data-root', type=str, default=None)
    eval_parser.add_argument('--conf-threshold', type=float, default=None,
                             help='Objectness confidence threshold (default: config)')
    eval_parser.add_argument('--limit', type=int, default=None,
                             help='Evaluate only first N val images (quick test)')

    # Diagnose
    diagnose_parser = subparsers.add_parser('diagnose', help='Run deep error profiling and save images')
    diagnose_parser.add_argument('--dataset', type=str, default=ACTIVE_DATASET,
                                 choices=['llvip', 'kaist', 'flirv2', 'camod3fd'])
    diagnose_parser.add_argument('--data-root', type=str, default=None)

    # Infer
    infer_parser = subparsers.add_parser('infer', help='Run inference on an image')
    infer_parser.add_argument('--image-rgb', type=str, required=True, help='Path to RGB image')
    infer_parser.add_argument('--image-thermal', type=str, required=True, help='Path to thermal image')
    infer_parser.add_argument('--model-path', type=str, default=BEST_MODEL_PATH)
    infer_parser.add_argument('--num-anchors', type=int, default=None, help='Override num_anchors')

    # Export
    export_parser = subparsers.add_parser('export', help='Export model to ONNX/TFLite')
    export_parser.add_argument('--model-path', type=str, default=BEST_MODEL_PATH)
    export_parser.add_argument('--format', type=str, choices=['onnx'], default='onnx')

    # Benchmark
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark model performance/size')
    bench_parser.add_argument('--model-path', type=str, default=None, help='Load specific model')

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    if args.mode == 'train':
        if args.debug:
            import config
            config.DEBUG_MODE = True
            args.no_kmeans = True
            print("=> DEBUG MODE ENABLED: Running exactly 1 toy epoch per phase (1 batch only) and skipping K-Means.")
            
        print(f"Initializing training pipeline... (V2 mode: {not args.v1})")
        
        encoder = GridEncoder()
        preprocessor = ThermalPreprocessor(encoder=encoder)
        opt_sizes = None

        if args.v1:
            # =================================================================
            # V1 TRAINING PIPELINE (Legacy Single Phase)
            # =================================================================
            dataset_root = args.data_root or get_dataset_path(args.dataset)
            print(f"Dataset path: {dataset_root}")

            train_loader, val_loader = create_dataloaders(
                dataset_name=args.dataset, preprocessor=preprocessor,
                dataset_root=dataset_root, batch_size=args.batch_size,
                num_workers=args.num_workers
            )

            if not args.no_kmeans:
                opt_ratios, opt_sizes = analyze_dataset_anchors(
                    train_loader.dataset.base_dataset, num_anchors=NUM_ANCHORS)
                encoder.update_anchors(opt_ratios, opt_sizes)
                train_loader, val_loader = create_dataloaders(
                    dataset_name=args.dataset, dataset_root=dataset_root,
                    preprocessor=preprocessor, batch_size=args.batch_size,
                    num_workers=args.num_workers
                )

            model = MicroGhostThermal()
            if args.resume_v1:
                from config import BEST_MODEL_V1_PATH
                if os.path.exists(BEST_MODEL_V1_PATH):
                    ckpt = torch.load(BEST_MODEL_V1_PATH, map_location='cpu')
                    model.load_state_dict(ckpt['model_state_dict'], strict=False)
                    print(f"  Fine-tuning from V1: {BEST_MODEL_V1_PATH}")

            print_model_analysis(model)
            benchmark_model(model)

            trainer = Trainer(
                model=model, train_loader=train_loader, val_loader=val_loader,
                epochs=args.epochs, anchor_sizes=opt_sizes
            )
            if getattr(__import__('config'), 'DEBUG_MODE', False):
                print(f"\n[DEBUG] Skipping actual V1 training as requested.")
                trainer._save_checkpoint(BEST_MODEL_PATH, 0, {'loss_total': 999.0})
            else:
                history = trainer.fit(save_path=BEST_MODEL_PATH)
                plot_training_history(history, save_path='training_history_v1.png')

        else:
            # =================================================================
            # V2 TRAINING PIPELINE (Multi-Phase)

            if not args.no_kmeans:
                print("\n[V2] Running K-Means clustering on ALL datasets for universal anchors...")
                from data_loading import LLVIPDataset, ForestPersonsDataset, ForestPersonsIRDataset, CAMOD3FDDataset
                from torch.utils.data import ConcatDataset
                
                raw_train_datasets = []
                for ds_name in ['llvip', 'forestpersons', 'forestpersonsir', 'camod3fd']:
                    ds_path = get_dataset_path(ds_name)
                    if os.path.exists(ds_path):
                        try:
                            if ds_name == 'llvip': raw_train_datasets.append(LLVIPDataset(ds_path, split='train', verbose=False))
                            elif ds_name == 'forestpersons': raw_train_datasets.append(ForestPersonsDataset(ds_path, split='train', verbose=False))
                            elif ds_name == 'forestpersonsir': raw_train_datasets.append(ForestPersonsIRDataset(ds_path, split='train', verbose=False))
                            elif ds_name == 'camod3fd': raw_train_datasets.append(CAMOD3FDDataset(ds_path, split='train', verbose=False))
                        except Exception as e:
                            print(f"  Warning: failed to load {ds_name} for K-Means: {e}")
                            
                if raw_train_datasets:
                    combined_raw = ConcatDataset(raw_train_datasets)
                    opt_ratios, opt_sizes = analyze_dataset_anchors(combined_raw, num_anchors=3)
                    preprocessor.encoder.update_anchors(opt_ratios, opt_sizes)
                else:
                    print("  No datasets found. Skipping K-Means.")
            
            model = MicroGhostV2(training_mode=True)
            print_model_analysis(model)
            benchmark_model(model)

            phases_to_run = [args.phase] if args.phase else [1, 2, 3, 4]

            # If starting from a later phase, attempt to load previous best model
            if phases_to_run[0] > 1 and os.path.exists(BEST_MODEL_PATH):
                print(f"Loading previous weights from {BEST_MODEL_PATH} before starting Phase {phases_to_run[0]}")
                ckpt = torch.load(BEST_MODEL_PATH, map_location='cpu')
                model.load_state_dict(ckpt['model_state_dict'])
            
            for phase in phases_to_run:
                train_loader, val_loader = create_phase_dataloaders(
                    phase=phase, preprocessor=preprocessor,
                    batch_size=args.batch_size, num_workers=args.num_workers
                )
                
                trainer = PhaseTrainer(
                    model=model, train_loader=train_loader, val_loader=val_loader,
                    phase=phase, anchor_sizes=opt_sizes
                )
                
                # Use fewer epochs if --epochs is specified (for quick testing)
                if args.epochs:
                    trainer.epochs = args.epochs
                    
                if getattr(__import__('config'), 'DEBUG_MODE', False):
                    print(f"\n[DEBUG] Skipping actual training for Phase {phase} as requested.")
                    trainer._save_checkpoint(BEST_MODEL_PATH, 0, {'loss_total': 999.0})
                    break  # Stop after skipping the first phase setup in debug mode
                else:
                    history = trainer.fit(save_path=BEST_MODEL_PATH)
                    plot_training_history(history, save_path=f'training_history_v2_phase{phase}.png')

            # =================================================================
            # POST-TRAINING PLUG-AND-PLAY SUITE
            # =================================================================
            print("\n" + "="*60)
            print("  STARTING AUTOMATED POST-TRAINING SUITE")
            print("="*60)
            
            # 1. Evaluate
            print("\n[1/3] Running Full Validation Evaluation...")
            dataset_root = args.data_root or get_dataset_path(args.dataset)
            run_detection_evaluation(
                model_path=BEST_MODEL_PATH,
                dataset_name=args.dataset,
                dataset_root=dataset_root
            )

            # 2. Diagnose
            print("\n[2/3] Generating Visual Diagnostics...")
            try:
                from diagnose import run_visual_diagnostics
                for dset in ['llvip', 'camod3fd', 'forestpersons']:
                    print(f"  -> Diagnosing {dset}...")
                    run_visual_diagnostics(
                        dataset_name=dset, 
                        dataset_root=get_dataset_path(dset)
                    )
            except Exception as e:
                print(f"Diagnostics skipped or failed: {e}")

            # 3. Export
            print("\n[3/3] Exporting to ONNX...")
            from inference import load_inference_model
            final_model, _ = load_inference_model(BEST_MODEL_PATH)
            export_to_onnx(final_model, ONNX_PATH)
            print("\nPlug-and-Play Suite Complete! All artifacts saved.")


    elif args.mode == 'benchmark':
        if args.model_path and os.path.exists(args.model_path):
            from inference import load_inference_model
            model, _ = load_inference_model(args.model_path)
            print("Loaded trained model for benchmark.")
        else:
            model = MicroGhostV2(training_mode=False)
            print("Using newly initialized V2 model for benchmark.")

        print_model_analysis(model)
        benchmark_model(model)

    elif args.mode == 'export':
        from inference import load_inference_model
        model, _ = load_inference_model(args.model_path)

        if args.format == 'onnx':
            export_to_onnx(model, ONNX_PATH)

    elif args.mode == 'infer':
        engine = ThermalInferenceEngine(model_path=args.model_path, override_num_anchors=args.num_anchors)
        img_bgr = cv2.imread(args.image_rgb, cv2.IMREAD_COLOR)
        img_thermal = cv2.imread(args.image_thermal, cv2.IMREAD_GRAYSCALE)

        if img_bgr is None or img_thermal is None:
            print("Error: Could not load one or both images.")
            return

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        detections = engine.detect_confirmed(img_rgb, img_thermal, lap_thresh=80.0) 
        
        print("\n--- Detection Results ---")
        if not detections:
            print("No intrusion detected.")
        else:
            for i, det in enumerate(detections):
                gw = det.get('gate_w', {})
                gw_str = f"[Gate R:{gw.get('rgb',0):.2f}/T:{gw.get('thm',0):.2f}] " if gw else ""
                print(f"[{i+1}] Intrusion! Conf: {det['combined_conf']:.2f} | "
                      f"Temp: {det.get('temp_c', 0)}°C | {gw_str}BBox: {det['bbox']}")

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
        
        colormap = cv2.COLORMAP_JET
        thermal_vis = cv2.applyColorMap(img_thermal, colormap)
        for det in detections:
            x1, y1, x2, y2 = [int(v * d) for v, d in zip(det['bbox'], [w, h, w, h])]
            cv2.rectangle(thermal_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        out_thermal = os.path.join('runs', f'{stem}_thermal_det.jpg')
        cv2.imwrite(out_thermal, thermal_vis)
        
        print(f"Saved → {out_thermal}")
        print(f"Saved → {out_path}")
        
        try:
            os.startfile(out_path)
        except AttributeError:
            pass # os.startfile is Windows only

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
