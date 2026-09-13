"""
experiments/run_traditional.py

Executes Model 0 (Traditional ML Baselines: Random Forest & XGBoost).
This script leverages the data pipeline to extract stationary sliding windows,
flattens them, and evaluates the performance ceiling of non-sequential models.
"""

import sys
import json
from pathlib import Path

# Add the project root to the Python path to allow imports from src/
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.data.dataset import prepare_dataloaders
from src.models.traditional import train_random_forest, train_xgboost

def main():
    print("================================================================================")
    print("RUNNING EXPERIMENT: TRADITIONAL ML BASELINES (MODEL 0)")
    print("================================================================================\n")

    data_file = base_dir / "data" / "processed" / "daily_processed_features.csv"
    
    if not data_file.exists():
        print(f"Error: Data file not found at {data_file}")
        sys.exit(1)

    # 1. Load Data via the unified pipeline
    print("1. Initializing Chronological Data Pipeline...")
    dataloaders, _, _ = prepare_dataloaders(str(data_file))

    # 2. Extract full numpy arrays for Scikit-Learn/XGBoost
    # Traditional models require the entire dataset array in memory, not batched tensors.
    X_train = dataloaders['train'].dataset.X.numpy()
    y_train = dataloaders['train'].dataset.y.numpy()
    
    X_val = dataloaders['val'].dataset.X.numpy()
    y_val = dataloaders['val'].dataset.y.numpy()
    
    X_test = dataloaders['test'].dataset.X.numpy()
    y_test = dataloaders['test'].dataset.y.numpy()

    # 3. Train and Evaluate
    print("\n2. Executing Baseline Models...")
    
    rf_model, rf_metrics, rf_preds = train_random_forest(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    xgb_model, xgb_metrics, xgb_preds = train_xgboost(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # 4. Save Results
    print("\n3. Saving Evaluation Metrics...")
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "traditional_baseline_results.json"
    
    final_results = {
        "RandomForest": rf_metrics,
        "XGBoost": xgb_metrics
    }
    
    with open(out_file, "w") as f:
        json.dump(final_results, f, indent=4)
        
    print(f"Metrics successfully saved to: {out_file}")
    print("Experiment Complete.")

if __name__ == "__main__":
    main()