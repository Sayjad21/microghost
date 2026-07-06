import os
import glob
from evaluation import run_detection_evaluation
from config import get_dataset_path

def main():
    print("=" * 60)
    print("  STARTING AUTOMATED METRICS BENCHMARKING")
    print("=" * 60)

    # 1. Discover all model checkpoints
    checkpoints_dir = 'checkpoints'
    model_paths = sorted(glob.glob(os.path.join(checkpoints_dir, 'best_microghost_thermal_*.pth')))
    
    if not model_paths:
        print(f"No models found in {checkpoints_dir}/ matching 'best_microghost_thermal_*.pth'")
        return

    print(f"Found {len(model_paths)} models to benchmark:")
    for m in model_paths:
        print(f"  - {os.path.basename(m)}")

    datasets_to_test = ['llvip', 'camod3fd']
    
    # We will store results to format them into a table later
    results_table = []
    
    for dataset_name in datasets_to_test:
        dataset_root = get_dataset_path(dataset_name)
        if not os.path.exists(dataset_root):
            print(f"\nWarning: Dataset '{dataset_name}' not found at {dataset_root}. Skipping.")
            continue
            
        print(f"\nEvaluating on Dataset: {dataset_name.upper()}")
        
        for model_path in model_paths:
            model_name = os.path.basename(model_path).replace('.pth', '')
            print(f"  Evaluating {model_name}...")
            
            try:
                # Run the evaluation
                # limit=None means it evaluates the full validation set
                metrics = run_detection_evaluation(
                    model_path=model_path,
                    dataset_name=dataset_name,
                    dataset_root=dataset_root,
                    verbose=False  # Suppress the massive per-image logs for this summary
                )
                
                # Extract metrics
                p = metrics.get('precision_at_50', 0.0)
                r = metrics.get('recall_at_50', 0.0)
                map50 = metrics.get('mAP50', 0.0)
                
                # Calculate F1 Score
                f1 = 0.0
                if (p + r) > 0:
                    f1 = 2 * p * r / (p + r)
                    
                results_table.append({
                    'Dataset': dataset_name.upper(),
                    'Model': model_name,
                    'Precision': p * 100,
                    'Recall': r * 100,
                    'F1-Score': f1 * 100,
                    'mAP@0.50': map50 * 100
                })
            except Exception as e:
                print(f"    Failed to evaluate {model_name} on {dataset_name}: {e}")
                
    # 2. Format and Save Markdown Table
    out_dir = 'diagnostic_results'
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'metrics_comparison.md')
    
    md_content = "# Cross-Model Metrics Benchmark\n\n"
    md_content += "| Dataset | Model | Precision | Recall | F1-Score | mAP@0.50 |\n"
    md_content += "|---------|-------|-----------|--------|----------|----------|\n"
    
    for row in results_table:
        md_content += f"| {row['Dataset']} | {row['Model']} | {row['Precision']:.2f}% | {row['Recall']:.2f}% | **{row['F1-Score']:.2f}%** | {row['mAP@0.50']:.2f}% |\n"
        
    with open(out_file, 'w') as f:
        f.write(md_content)
        
    print("\n" + "=" * 60)
    print(f"Benchmarking Complete! Saved detailed markdown table to:\n -> {out_file}")
    print("=" * 60)
    print(md_content)

if __name__ == '__main__':
    main()
