import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import INPUT_WIDTH, INPUT_HEIGHT
from data_loading import get_val_base_dataset
from inference import ThermalInferenceEngine, calculate_iou_numpy

# --- PATH CONFIGURATION ---
V1_MODEL_PATH = 'checkpoints/best_microghost_thermal_v1.pth'
V2_MODEL_PATH = 'checkpoints/best_microghost_thermal_v2.pth'

def calculate_metrics(pred_pixels, gt_pixels, iou_thresh=0.5):
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
                matched_gts.add(best_gt_idx)
                matched_preds.add(p_idx)

    tp = len(matched_preds)
    fp = len(pred_pixels) - tp
    fn = len(gt_pixels) - len(matched_gts)
    return tp, fp, fn

def run_model_comparison(dataset_name='llvip', output_dir='comparison_results'):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading V1 (Base) Engine... [Forcing 2 Anchors]")
    engine_v1 = ThermalInferenceEngine(model_path=V1_MODEL_PATH, override_num_anchors=2)
    
    if not os.path.exists(V2_MODEL_PATH):
        print(f"ERROR: {V2_MODEL_PATH} not found. Please train the V2 model first.")
        return

    print("Loading V2 (Updated) Engine... [Dynamic/3 Anchors]")
    engine_v2 = ThermalInferenceEngine(model_path=V2_MODEL_PATH)
    
    val_dataset = get_val_base_dataset(dataset_name)
    
    # Stat Tracking
    stats = {
        'v1': {'tp': 0, 'fp': 0, 'fn': 0, 'total_preds': 0},
        'v2': {'tp': 0, 'fp': 0, 'fn': 0, 'total_preds': 0},
        'total_gt': 0
    }
    
    print(f"Comparing models across {len(val_dataset)} validation frames...")
    
    for idx in tqdm(range(len(val_dataset)), desc="Evaluating"):
        sample = val_dataset[idx]
        img_rgb = sample[0][0]
        img_thermal = sample[0][1]
        annotations = sample[1]
        h_orig, w_orig = sample[2]
        
        # Ground Truth
        gt_pixels = [[gt['xmin'], gt['ymin'], gt['xmax'], gt['ymax']] for gt in annotations]
        stats['total_gt'] += len(gt_pixels)
        
        # V1 Inference
        preds_v1 = engine_v1.detect(img_rgb, img_thermal)
        v1_pixels = [[p['bbox'][0]*w_orig, p['bbox'][1]*h_orig, p['bbox'][2]*w_orig, p['bbox'][3]*h_orig] for p in preds_v1]
        v1_confs = [p['combined_conf'] for p in preds_v1]
        
        # V2 Inference
        preds_v2 = engine_v2.detect(img_rgb, img_thermal)
        v2_pixels = [[p['bbox'][0]*w_orig, p['bbox'][1]*h_orig, p['bbox'][2]*w_orig, p['bbox'][3]*h_orig] for p in preds_v2]
        v2_confs = [p['combined_conf'] for p in preds_v2]
        
        # Update Stats
        tp1, fp1, fn1 = calculate_metrics(v1_pixels, gt_pixels)
        stats['v1']['tp'] += tp1; stats['v1']['fp'] += fp1; stats['v1']['fn'] += fn1; stats['v1']['total_preds'] += len(v1_pixels)
        
        tp2, fp2, fn2 = calculate_metrics(v2_pixels, gt_pixels)
        stats['v2']['tp'] += tp2; stats['v2']['fp'] += fp2; stats['v2']['fn'] += fn2; stats['v2']['total_preds'] += len(v2_pixels)
        
        # Save Visual Comparison (Cap at 30 to save space)
        if (fp1 != fp2 or fn1 != fn2) and idx < 30: 
            save_comparison_plot(img_rgb, img_thermal, gt_pixels, v1_pixels, v1_confs, v2_pixels, v2_confs, idx, output_dir)

    # Calculate final metrics
    for v in ['v1', 'v2']:
        p = stats[v]['tp'] / max(1, stats[v]['tp'] + stats[v]['fp'])
        r = stats[v]['tp'] / max(1, stats[v]['tp'] + stats[v]['fn'])
        f1 = 2 * (p * r) / max(1e-9, p + r)
        stats[v]['precision'] = p; stats[v]['recall'] = r; stats[v]['f1'] = f1

    # Print Report
    print("\n" + "="*60)
    print(f"{'METRIC':<20} | {'V1 (BASE MODEL)':<15} | {'V2 (UPDATED MODEL)':<15}")
    print("="*60)
    print(f"{'Total GT Targets':<20} | {stats['total_gt']:<15} | {stats['total_gt']:<15}")
    print(f"{'Total Predictions':<20} | {stats['v1']['total_preds']:<15} | {stats['v2']['total_preds']:<15}")
    print(f"{'True Positives':<20} | {stats['v1']['tp']:<15} | {stats['v2']['tp']:<15}")
    print(f"{'False Positives':<20} | {stats['v1']['fp']:<15} | {stats['v2']['fp']:<15}")
    print(f"{'Missed Targets (FN)':<20} | {stats['v1']['fn']:<15} | {stats['v2']['fn']:<15}")
    print("-" * 60)
    print(f"{'Precision':<20} | {stats['v1']['precision']:.4f}          | {stats['v2']['precision']:.4f}")
    print(f"{'Recall':<20} | {stats['v1']['recall']:.4f}          | {stats['v2']['recall']:.4f}")
    print(f"{'F1 Score':<20} | {stats['v1']['f1']:.4f}          | {stats['v2']['f1']:.4f}")
    print("="*60)


def save_comparison_plot(rgb, thermal, gt, v1_box, v1_conf, v2_box, v2_conf, idx, out_dir):
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    titles = ["RGB View", "Thermal View"]
    colors = ['cyan', 'red', 'lime'] # GT, V1, V2
    rows = [
        ("Ground Truth", gt, None, colors[0]),
        ("V1 Base Model", v1_box, v1_conf, colors[1]),
        ("V2 Updated Model", v2_box, v2_conf, colors[2])
    ]
    
    for row_idx, (row_title, boxes, confs, color) in enumerate(rows):
        for col_idx in range(2):
            ax = axes[row_idx, col_idx]
            img = rgb if col_idx == 0 else thermal
            ax.imshow(img, cmap='inferno' if col_idx == 1 else None)
            ax.set_title(f"{row_title} - {titles[col_idx]}", fontsize=14)
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=2)
                ax.add_patch(rect)
                if confs is not None:
                    ax.text(x1, y1 - 4, f"{confs[i]:.2f}", color=color, fontsize=10, weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"compare_{idx:04d}.png"), dpi=150)
    plt.close()

if __name__ == '__main__':
    run_model_comparison()
