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
import cv2

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

DATASET_CHOICES = ['llvip', 'forestpersons', 'forestpersonsir', 'camod3fd', 'kaist', 'flirv2']


def parse_args():
    parser = argparse.ArgumentParser(description="MicroGhost-Thermal Intrusion Detection (V2)")
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # Train
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--dataset', type=str, default=ACTIVE_DATASET,
                              choices=DATASET_CHOICES, help='Dataset for evaluation after training')
    train_parser.add_argument('--data-root', type=str, default=None,
                              help='Override dataset directory')
    train_parser.add_argument('--epochs', type=int, default=None, help='Override config epochs')
    train_parser.add_argument('--batch-size', type=int, default=None, help='Override config batch size')
    train_parser.add_argument('--num-workers', type=int, default=0, help='Override config num workers (CPU scaling)')
    train_parser.add_argument('--no-kmeans', action='store_true', help='Skip K-Means anchor optimization')
    train_parser.add_argument('--phase', type=int, default=None, choices=[1, 2, 3],
                              help='Run a specific V2 training phase (1-3). If omitted, runs all phases.')
    train_parser.add_argument('--debug', action='store_true', help='Run 1 toy epoch per phase for debugging')

    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model-path', type=str, default=BEST_MODEL_PATH)
    eval_parser.add_argument('--dataset', type=str, default=ACTIVE_DATASET,
                             choices=DATASET_CHOICES)
    eval_parser.add_argument('--data-root', type=str, default=None)
    eval_parser.add_argument('--conf-threshold', type=float, default=None,
                             help='Objectness confidence threshold (default: config)')
    eval_parser.add_argument('--limit', type=int, default=None,
                             help='Evaluate only first N val images (quick test)')

    # Diagnose
    diagnose_parser = subparsers.add_parser('diagnose', help='Run deep error profiling and save images')
    diagnose_parser.add_argument('--dataset', type=str, default=ACTIVE_DATASET,
                                 choices=DATASET_CHOICES)
    diagnose_parser.add_argument('--data-root', type=str, default=None)

    # Infer
    infer_parser = subparsers.add_parser('infer', help='Run inference on an image')
    infer_parser.add_argument('--image-rgb', type=str, default=None, help='Path to RGB image')
    infer_parser.add_argument('--image-thermal', type=str, default=None, help='Path to thermal image')
    infer_parser.add_argument('--model-path', type=str, default=BEST_MODEL_PATH)
    infer_parser.add_argument('--num-anchors', type=int, default=None, help='Override num_anchors')
    infer_parser.add_argument(
        '--conf-thresh', '--conf-threshold',
        dest='conf_thresh',
        type=float,
        default=None,
        help='Override detection confidence threshold; thermal-only defaults to 0.20'
    )
    infer_parser.add_argument(
        '--lap-thresh', '--lap-thres',
        dest='lap_thresh',
        type=float,
        default=80.0,
        help='Reject detections with RGB crop Laplacian variance below this value'
    )
    infer_parser.add_argument(
        '--lap-bypass-conf',
        type=float,
        default=None,
        help='Optional: skip the Laplacian gate for detections at or above this confidence'
    )
    infer_parser.add_argument(
        '--merge-boxes',
        action='store_true',
        help='Optionally merge nearby stacked boxes; off by default to avoid grouping adjacent people'
    )

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
            
        print(f"Initializing training pipeline... (V2 mode: True)")
        
        encoder = GridEncoder()
        preprocessor = ThermalPreprocessor(encoder=encoder)
        opt_ratios = None
        opt_sizes = None

        # =================================================================
        # V2 TRAINING PIPELINE (Multi-Phase)

        if not args.no_kmeans:
            print("\n[V2] Running K-Means clustering on ALL datasets for universal anchors...")
            from data_loading import LLVIPDataset, ForestPersonsDataset, ForestPersonsIRDataset, CAMOD3FDDataset
            from torch.utils.data import ConcatDataset
            
            raw_train_datasets = []
            for ds_name in ['llvip', 'forestpersons', 'forestpersonsir']:
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

        phases_to_run = [args.phase] if args.phase else [1, 2, 3]

        # If starting from a later phase, attempt to load previous best model
        if phases_to_run[0] > 1 and os.path.exists(BEST_MODEL_PATH):
            print(f"Loading previous weights from {BEST_MODEL_PATH} before starting Phase {phases_to_run[0]}")
            ckpt = torch.load(BEST_MODEL_PATH, map_location='cpu', weights_only=False)
            state_dict = ckpt.get('model_state_dict', ckpt)
            model_state = model.state_dict()
            compatible = {
                k: v for k, v in state_dict.items()
                if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)
            }
            skipped = sorted(set(state_dict.keys()) - set(compatible.keys()))
            model.load_state_dict(compatible, strict=False)
            if skipped:
                print(f"  Compatibility load skipped {len(skipped)} mismatched/new keys.")
            ckpt_cfg = ckpt.get('config', {}) if isinstance(ckpt, dict) else {}
            opt_sizes = ckpt_cfg.get('anchor_sizes', opt_sizes)
            opt_ratios = ckpt_cfg.get('anchor_ratios', opt_ratios)
            if opt_sizes is not None and opt_ratios is not None:
                preprocessor.encoder.update_anchors(opt_ratios, opt_sizes)
        
        for phase in phases_to_run:
            train_loader, val_loader = create_phase_dataloaders(
                phase=phase, preprocessor=preprocessor,
                batch_size=args.batch_size, num_workers=args.num_workers
            )
            
            trainer = PhaseTrainer(
                model=model, train_loader=train_loader, val_loader=val_loader,
                phase=phase, anchor_sizes=opt_sizes, anchor_ratios=opt_ratios
            )
            
            # Use fewer epochs if --epochs is specified (for quick testing)
            if args.epochs:
                trainer.epochs = args.epochs
                
            if getattr(__import__('config'), 'DEBUG_MODE', False):
                print(f"\n[DEBUG] Skipping actual training for Phase {phase} as requested.")
                # Save dummy checkpoint so evaluation doesn't fail
                trainer._save_checkpoint(BEST_MODEL_PATH, 0, {'loss_total': 999.0})
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
            dataset_root=dataset_root,
            limit=10 if getattr(__import__('config'), 'DEBUG_MODE', False) else None
        )

        # 2. Diagnose
        print("\n[2/3] Generating Visual Diagnostics...")
        try:
            from diagnose import run_visual_diagnostics
            for dset in ['llvip', 'forestpersons', 'forestpersonsir']:
                print(f"  -> Diagnosing {dset}...")
                run_visual_diagnostics(
                    dataset_name=dset, 
                    dataset_root=get_dataset_path(dset),
                    output_dir=f"diagnostic_results/{dset}",
                    limit=10 if getattr(__import__('config'), 'DEBUG_MODE', False) else None
                )
        except Exception as e:
            print(f"Diagnostics skipped or failed: {e}")

        # 3. Export
        print("\n[3/3] Exporting to ONNX...")
        from inference import load_inference_model
        final_model, _ = load_inference_model(BEST_MODEL_PATH)
        try:
            export_to_onnx(final_model, ONNX_PATH)
        except Exception as e:
            print(f"  [Warning] ONNX export failed (you may need to run '!pip install onnxscript'): {e}")
            
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
        if not args.image_rgb and not args.image_thermal:
            print("Error: provide --image-rgb, --image-thermal, or both.")
            return

        engine = ThermalInferenceEngine(model_path=args.model_path, override_num_anchors=args.num_anchors)

        if args.image_rgb:
            img_bgr = cv2.imread(args.image_rgb, cv2.IMREAD_COLOR)
            if img_bgr is None:
                print("Error: Could not load RGB image.")
                return
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            lap_image = img_rgb
            output_source = args.image_rgb
        else:
            print("No RGB image provided. Running thermal-only inference with a blank RGB branch.")
            img_rgb = None

        if args.image_thermal:
            img_thermal = cv2.imread(args.image_thermal, cv2.IMREAD_GRAYSCALE)
            if img_thermal is None:
                print("Error: Could not load thermal image.")
                return
        else:
            print("No thermal image provided. Running RGB-only inference with a blank thermal branch.")
            h_rgb, w_rgb = img_rgb.shape[:2]
            img_thermal = np.zeros((h_rgb, w_rgb), dtype=np.uint8)

        if img_rgb is None:
            h_thm, w_thm = img_thermal.shape[:2]
            img_rgb = np.zeros((h_thm, w_thm, 3), dtype=np.uint8)
            img_bgr = cv2.cvtColor(img_thermal, cv2.COLOR_GRAY2BGR)
            lap_image = img_thermal
            output_source = args.image_thermal

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
        
        print("\n--- Detection Results ---")
        if not detections:
            print("No intrusion detected.")
        else:
            for i, det in enumerate(detections):
                gw = det.get('gate_w', {})
                lap_str = f"LapVar: {det.get('lap_var', 0.0):.1f} | "
                gw_str = lap_str + (f"[Gate R:{gw.get('rgb',0):.2f}/T:{gw.get('thm',0):.2f}] " if gw else "")
                print(f"[{i+1}] Intrusion! Conf: {det['combined_conf']:.2f} | "
                      f"Thermal: {det.get('thermal_score', 0.0):.2f} | {gw_str}BBox: {det['bbox']}")

        h, w = img_bgr.shape[:2]
        vis = img_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v * d) for v, d in zip(det['bbox'], [w, h, w, h])]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['combined_conf']:.2f}"
            cv2.putText(vis, label, (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        os.makedirs('runs', exist_ok=True)
        stem = os.path.splitext(os.path.basename(output_source))[0]
        out_path = os.path.join('runs', f'{stem}_det.jpg')
        cv2.imwrite(out_path, vis)
        
        colormap = cv2.COLORMAP_JET
        thermal_vis = cv2.applyColorMap(img_thermal, colormap)
        for det in detections:
            x1, y1, x2, y2 = [int(v * d) for v, d in zip(det['bbox'], [w, h, w, h])]
            cv2.rectangle(thermal_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        out_thermal = os.path.join('runs', f'{stem}_thermal_det.jpg')
        cv2.imwrite(out_thermal, thermal_vis)
        
        print(f"Saved -> {out_thermal}")
        print(f"Saved -> {out_path}")
        
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
