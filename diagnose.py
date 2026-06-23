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

def run_visual_diagnostics(dataset_name='llvip', dataset_root=None, output_dir='diagnostic_results', iou_thresh=0.5):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Initialize Engine and Raw Validation Split
    print("Loading inference engine...")
    engine = ThermalInferenceEngine(model_path=BEST_MODEL_PATH)
    val_dataset = get_val_base_dataset(dataset_name, dataset_root=dataset_root)
    
    # Error pattern counters
    stats = {'pure_false_positives': 0, 'missed_targets': 0, 'mislocalized': 0, 'correct': 0}
    
    print(f"Investigating {len(val_dataset)} validation frames...")
    for idx in tqdm(range(len(val_dataset)), desc="Diagnosing"):
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
        pred_pixels = []
        pred_confs = []
        for p in predictions:
            x1, y1, x2, y2 = p['bbox']
            # Scale back to original image dimensions for accurate IoU
            pred_pixels.append([x1 * w_orig, y1 * h_orig, x2 * w_orig, y2 * h_orig])
            pred_confs.append(p['combined_conf'])

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
            save_diagnostic_plot(img_thermal, gt_pixels, pred_pixels, pred_confs, idx, output_dir)

    print("\n" + "="*40 + "\n  DETAILED ERROR PATTERN SUMMARY\n" + "="*40)
    for k, v in stats.items():
        print(f"  {k.replace('_', ' ').title():<25}: {v}")
    print("="*40)
    print(f"✅ Diagnostic snapshots saved to: ./{output_dir}/")


def save_diagnostic_plot(thermal_img, gt_boxes, pred_boxes, confs, index, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left: Ground Truth
    ax1.imshow(thermal_img, cmap='inferno')
    ax1.set_title("Ground Truth Annotations")
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='cyan', linewidth=2)
        ax1.add_patch(rect)
        
    # Right: Model Inference
    ax2.imshow(thermal_img, cmap='inferno')
    ax2.set_title("Model Inference Predictions")
    for box, conf in zip(pred_boxes, confs):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='magenta', linewidth=2)
        ax2.add_patch(rect)
        ax2.text(x1, y1 - 4, f"{conf:.2f}", color='magenta', fontsize=9, weight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"error_frame_{index:04d}.png"), dpi=150)
    plt.close()

if __name__ == '__main__':
    run_visual_diagnostics()
