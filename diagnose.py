import os
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for Kaggle compatibility
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import BEST_MODEL_PATH, INPUT_WIDTH, INPUT_HEIGHT
from data_loading import get_val_base_dataset
from inference import ThermalInferenceEngine, calculate_iou_numpy

def run_visual_diagnostics(dataset_name='llvip', dataset_root=None, output_dir='diagnostic_results', iou_thresh=0.5, limit=None):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Initialize Engine and Raw Validation Split
    print("Loading inference engine...")
    engine = ThermalInferenceEngine(model_path=BEST_MODEL_PATH)
    val_dataset = get_val_base_dataset(dataset_name, dataset_root=dataset_root)
    
    # Error pattern counters
    stats = {'pure_false_positives': 0, 'missed_targets': 0, 'mislocalized': 0, 'correct': 0}
    
    num_samples = len(val_dataset)
    if limit is not None:
        num_samples = min(num_samples, limit)
        
    print(f"Investigating {num_samples} validation frames...")
    for idx in tqdm(range(num_samples), desc="Diagnosing"):
        # Fetch raw image entry
        (img_rgb, img_thermal), annotations, (h_orig, w_orig) = val_dataset[idx]
        
        # Run model inference (returns processed list of dicts with 'bbox' and 'combined_conf')
        predictions = engine.detect(img_rgb, img_thermal)
        
        # Decode Ground Truths
        gt_pixels = []
        for gt in annotations:
            # VOC format [xmin, ymin, xmax, ymax]
            gt_pixels.append([gt['xmin'], gt['ymin'], gt['xmax'], gt['ymax']])
            
        # Decode Predictions (Inference engine returns normalized coords 0-1)
        pred_confs = []
        pred_gate_w = []
        for p in predictions:
            x1, y1, x2, y2 = p['bbox']
            # Scale back to original image dimensions for accurate IoU
            pred_pixels.append([x1 * w_orig, y1 * h_orig, x2 * w_orig, y2 * h_orig])
            pred_confs.append(p['combined_conf'])
            pred_gate_w.append(p.get('gate_w', None))

        # 2. Categorize Errors via IoU Matching
        matched_gts = set()
        matched_preds = set()
        
        if len(pred_pixels) > 0 and len(gt_pixels) > 0:
            for p_idx, pred in enumerate(pred_pixels):
                best_iou = 0
                best_gt_idx = -1
                for g_idx, gt in enumerate(gt_pixels):
                    iou = calculate_iou_numpy(np.array(pred), np.array(gt))
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx
                
                if best_iou >= iou_thresh:
                    stats['correct'] += 1
                    matched_gts.add(best_gt_idx)
                    matched_preds.add(p_idx)
                elif best_iou > 0.1:
                    stats['mislocalized'] += 1
                    matched_preds.add(p_idx)
                    
        stats['pure_false_positives'] += (len(pred_pixels) - len(matched_preds))
        stats['missed_targets'] += (len(gt_pixels) - len(matched_gts))

        # 3. Save side-by-side snapshots for error patterns (e.g., mislocalizations or misses)
        has_error = (len(pred_pixels) != len(gt_pixels)) or (len(matched_gts) < len(gt_pixels))
        if has_error and idx < 50:  # Cap at 50 images to save disk space
            gt_boxes_for_plot = []
            for gt in annotations:
                gt_boxes_for_plot.append([gt['xmin'], gt['ymin'], gt['xmax'], gt['ymax']])
            save_diagnostic_plot(img_rgb, img_thermal, gt_boxes_for_plot, pred_pixels, pred_confs, pred_gate_w, idx, output_dir)

    print("\n" + "="*40 + "\n  DETAILED ERROR PATTERN SUMMARY\n" + "="*40)
    for k, v in stats.items():
        print(f"  {k.replace('_', ' ').title():<25}: {v}")
    print("="*40)
    print(f"✅ Diagnostic snapshots saved to: ./{output_dir}/")


def save_diagnostic_plot(rgb_img, thermal_img, gt_boxes, pred_boxes, confs, gate_ws, index, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ((ax1, ax2), (ax3, ax4)) = axes
    
    # --- Top Row: Ground Truth ---
    ax1.imshow(rgb_img)
    ax1.set_title("Ground Truth (RGB)")
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='cyan', linewidth=2)
        ax1.add_patch(rect)
        
    ax2.imshow(thermal_img, cmap='inferno')
    ax2.set_title("Ground Truth (Thermal)")
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='cyan', linewidth=2)
        ax2.add_patch(rect)

    # --- Bottom Row: Model Inference ---
    ax3.imshow(rgb_img)
    ax3.set_title("Model Inference (RGB)")
    for box, conf, gw in zip(pred_boxes, confs, gate_ws):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='magenta', linewidth=2)
        ax3.add_patch(rect)
        label = f"{conf:.2f}"
        if gw: label += f" [R:{gw['rgb']:.2f}]"
        ax3.text(x1, y1 - 4, label, color='magenta', fontsize=9, weight='bold')

    ax4.imshow(thermal_img, cmap='inferno')
    ax4.set_title("Model Inference (Thermal)")
    for box, conf, gw in zip(pred_boxes, confs, gate_ws):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='magenta', linewidth=2)
        ax4.add_patch(rect)
        label = f"{conf:.2f}"
        if gw: label += f" [T:{gw['thm']:.2f}]"
        ax4.text(x1, y1 - 4, label, color='magenta', fontsize=9, weight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"error_frame_{index:04d}.png"), dpi=150)
    plt.close()

if __name__ == '__main__':
    run_visual_diagnostics()
