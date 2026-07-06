"""
MicroGhost-Thermal: Detection Evaluation Module
================================================
Computes detection mAP (mean Average Precision) on the validation split.

Metrics:
- mAP@0.50       — PASCAL VOC style (IoU >= 0.5)
- mAP@0.50:0.95  — COCO style (mean AP over IoU 0.50–0.95, step 0.05)
- Precision / Recall @ fixed confidence threshold
- Mean IoU on matched true-positive detections
"""

import time
import numpy as np
import torch
from torch.utils.data import Subset

from config import (
    INPUT_HEIGHT, INPUT_WIDTH, DEVICE, DEFAULT_ANCHOR_SIZES,
    CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD,
)
from data_loading import get_val_base_dataset
from preprocessing import ThermalPreprocessor
from inference import load_inference_model, decode_predictions, calculate_iou_numpy

# COCO-style IoU thresholds for mAP@0.50:0.95
COCO_IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)


def annotations_to_norm_boxes(annotations, h_orig, w_orig):
    """
    Convert pixel annotations to normalized [x1, y1, x2, y2] at model input size.
    Matches the resize-only validation preprocessing path.
    """
    h_new, w_new = INPUT_HEIGHT, INPUT_WIDTH
    scale_x = w_new / w_orig
    scale_y = h_new / h_orig
    boxes = []

    for ann in annotations:
        xmin = ann['xmin'] * scale_x
        ymin = ann['ymin'] * scale_y
        xmax = ann['xmax'] * scale_x
        ymax = ann['ymax'] * scale_y
        xmin = max(0.0, min(xmin, w_new))
        ymin = max(0.0, min(ymin, h_new))
        xmax = max(xmin, min(xmax, w_new))
        ymax = max(ymin, min(ymax, h_new))
        if xmax > xmin and ymax > ymin:
            boxes.append([
                xmin / w_new, ymin / h_new,
                xmax / w_new, ymax / h_new,
            ])
    return boxes


def compute_ap(recalls, precisions):
    """Area under the precision-recall curve (VOC/COCO all-point interpolation)."""
    if len(recalls) == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    indices = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]))


def _match_predictions_to_gt(pred_boxes, gt_boxes, iou_threshold):
    """
    Greedy IoU matching for one image.

    Returns:
        num_tp, num_fp, matched_ious (list of IoU for each TP)
    """
    if not pred_boxes:
        return 0, 0, []

    order = sorted(range(len(pred_boxes)),
                   key=lambda i: pred_boxes[i]['conf'], reverse=True)
    gt_matched = [False] * len(gt_boxes)
    tp = 0
    fp = 0
    matched_ious = []

    for idx in order:
        pred = pred_boxes[idx]['bbox']
        best_iou = 0.0
        best_j = -1

        for j, gt in enumerate(gt_boxes):
            if gt_matched[j]:
                continue
            iou = calculate_iou_numpy(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j >= 0 and best_iou >= iou_threshold:
            tp += 1
            gt_matched[best_j] = True
            matched_ious.append(best_iou)
        else:
            fp += 1

    return tp, fp, matched_ious


def compute_ap_for_dataset(all_predictions, all_ground_truths, iou_threshold=0.5):
    """
    Compute AP for a single IoU threshold across the full validation set.

    Args:
        all_predictions: dict image_id -> list of {conf, bbox}
        all_ground_truths: dict image_id -> list of [x1,y1,x2,y2]
        iou_threshold: float

    Returns:
        ap, final_precision, final_recall
    """
    flat_preds = []
    for image_id, preds in all_predictions.items():
        for pred in preds:
            flat_preds.append((pred['conf'], pred['bbox'], image_id))

    flat_preds.sort(key=lambda x: -x[0])
    num_gt = sum(len(v) for v in all_ground_truths.values())
    if num_gt == 0:
        return 0.0, 0.0, 0.0

    tp = np.zeros(len(flat_preds))
    fp = np.zeros(len(flat_preds))
    gt_matched = {
        img_id: [False] * len(boxes)
        for img_id, boxes in all_ground_truths.items()
    }

    for i, (_, bbox, image_id) in enumerate(flat_preds):
        gt_boxes = all_ground_truths.get(image_id, [])
        best_iou = 0.0
        best_j = -1

        for j, gt in enumerate(gt_boxes):
            if gt_matched[image_id][j]:
                continue
            iou = calculate_iou_numpy(bbox, gt)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j >= 0 and best_iou >= iou_threshold:
            tp[i] = 1
            gt_matched[image_id][best_j] = True
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / num_gt
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    ap = compute_ap(recalls, precisions)
    final_prec = float(precisions[-1]) if len(precisions) else 0.0
    final_rec = float(recalls[-1]) if len(recalls) else 0.0
    return ap, final_prec, final_rec


def _resolve_sample(dataset, idx):
    """Load one validation sample from a raw or Subset dataset."""
    if isinstance(dataset, Subset):
        sample = dataset.dataset[dataset.indices[idx]]
    else:
        sample = dataset[idx]
    return sample


@torch.no_grad()
def run_detection_evaluation(
    model_path,
    dataset_name,
    dataset_root=None,
    device=None,
    conf_threshold=None,
    limit=None,
    verbose=True,
):
    """
    Run full detection mAP evaluation on the validation split.

    Args:
        model_path: Path to trained checkpoint
        dataset_name: 'llvip', 'kaist', or 'flirv2'
        dataset_root: Optional dataset path override
        device: torch device
        conf_threshold: Objectness confidence threshold for predictions
        limit: Evaluate only the first N images (quick smoke test)
        verbose: Print progress

    Returns:
        dict with mAP metrics
    """
    device = device or DEVICE
    conf_threshold = conf_threshold if conf_threshold is not None else CONFIDENCE_THRESHOLD

    if verbose:
        print("=" * 70)
        print("  DETECTION EVALUATION — MicroGhost-Thermal")
        print("=" * 70)
        print(f"  Model:      {model_path}")
        print(f"  Dataset:    {dataset_name}")
        print(f"  Device:     {device}")
        print(f"  Conf thresh:{conf_threshold}")
        print(f"  NMS IoU:    {NMS_IOU_THRESHOLD}")
        print("=" * 70)

    model, _ = load_inference_model(model_path, device=device)
    preprocessor = ThermalPreprocessor()
    val_dataset = get_val_base_dataset(dataset_name, dataset_root=dataset_root)

    n_samples = len(val_dataset)
    if limit is not None:
        n_samples = min(n_samples, limit)

    all_predictions = {}
    all_ground_truths = {}
    per_image_tp = 0
    per_image_fp = 0
    per_image_fn = 0
    matched_ious = []

    t0 = time.time()
    model.eval()

    for idx in range(n_samples):
        (image_rgb, image_thermal), annotations, (h_orig, w_orig) = \
            _resolve_sample(val_dataset, idx)

        img_tensor, _ = preprocessor.process(
            image_rgb=image_rgb,
            image_thermal=image_thermal,
            bboxes_pascal=[],
            labels=[],
            img_size=(h_orig, w_orig),
            augment=False,
        )

        preds = model(img_tensor.unsqueeze(0).to(device))
        detections = decode_predictions(
            preds['obj_small'][0], preds['bbox_small'][0],
            preds['obj_large'][0], preds['bbox_large'][0],
            anchor_sizes=DEFAULT_ANCHOR_SIZES,
            conf_threshold=conf_threshold,
        )

        gt_boxes = annotations_to_norm_boxes(annotations, h_orig, w_orig)
        all_predictions[idx] = detections
        all_ground_truths[idx] = gt_boxes

        tp, fp, ious = _match_predictions_to_gt(detections, gt_boxes, 0.5)
        per_image_tp += tp
        per_image_fp += fp
        per_image_fn += max(0, len(gt_boxes) - tp)
        matched_ious.extend(ious)

        if verbose and ((idx + 1) % 200 == 0 or idx + 1 == n_samples):
            elapsed = time.time() - t0
            print(f"  Processed {idx + 1}/{n_samples} images ({elapsed:.0f}s)")

    # mAP metrics
    ap50, prec50, rec50 = compute_ap_for_dataset(
        all_predictions, all_ground_truths, iou_threshold=0.5
    )

    ap_per_iou = []
    for iou_t in COCO_IOU_THRESHOLDS:
        ap, _, _ = compute_ap_for_dataset(
            all_predictions, all_ground_truths, iou_threshold=float(iou_t)
        )
        ap_per_iou.append(ap)
    map_coco = float(np.mean(ap_per_iou))

    total_gt = sum(len(v) for v in all_ground_truths.values())
    total_preds = sum(len(v) for v in all_predictions.values())
    images_with_gt = sum(1 for v in all_ground_truths.values() if v)
    images_with_det = sum(1 for v in all_predictions.values() if v)

    results = {
        'mAP50': ap50,
        'mAP50_95': map_coco,
        'precision_at_50': prec50,
        'recall_at_50': rec50,
        'mean_matched_iou': float(np.mean(matched_ious)) if matched_ious else 0.0,
        'total_gt_boxes': total_gt,
        'total_predictions': total_preds,
        'images_evaluated': n_samples,
        'images_with_gt': images_with_gt,
        'images_with_detections': images_with_det,
        'per_image_tp_50': per_image_tp,
        'per_image_fp_50': per_image_fp,
        'per_image_fn_50': per_image_fn,
    }

    elapsed = time.time() - t0
    if verbose:
        print("\n" + "=" * 70)
        print("  EVALUATION RESULTS")
        print("=" * 70)
        print(f"  Images evaluated:     {n_samples}")
        print(f"  GT boxes:             {total_gt}")
        print(f"  Predictions (post-NMS): {total_preds}")
        print(f"  Images with GT:       {images_with_gt}")
        print(f"  Images with dets:     {images_with_det}")
        print("-" * 70)
        print(f"  mAP@0.50:             {ap50 * 100:.2f}%")
        print(f"  mAP@0.50:0.95:        {map_coco * 100:.2f}%")
        print(f"  Precision@0.50:       {prec50 * 100:.2f}%")
        print(f"  Recall@0.50:          {rec50 * 100:.2f}%")
        print(f"  Mean matched IoU:     {results['mean_matched_iou']:.3f}")
        print("-" * 70)
        print(f"  TP / FP / FN @0.50:   {per_image_tp} / {per_image_fp} / {per_image_fn}")
        print(f"  Time:                 {elapsed:.1f}s")
        print("=" * 70)

    return results


if __name__ == '__main__':
    from config import BEST_MODEL_PATH, ACTIVE_DATASET, get_dataset_path

    run_detection_evaluation(
        model_path=BEST_MODEL_PATH,
        dataset_name=ACTIVE_DATASET,
        dataset_root=get_dataset_path(ACTIVE_DATASET),
        limit=50,
    )
