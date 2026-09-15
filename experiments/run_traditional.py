"""
experiments/run_traditional.py

Executes Model 0 (Traditional ML Baselines: Random Forest & XGBoost).
Incorporates strict naive baselines (Driftless Random Walk, Zero-Return, Historical Mean)
to establish the true economic forecasting floor.
"""

import sys
import json
import numpy as np
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.data.dataset import prepare_dataloaders
from src.models.traditional import train_random_forest, train_xgboost
from src.evaluation.metrics import calculate_point_metrics

def generate_naive_baselines(y_train, y_val, y_test):
    """Generates naive econometric baselines for point forecasting."""
    n_test = len(y_test)
    
    # 1. Driftless Random Walk (Persistence: y_hat_t = y_{t-1})
    y_persistence = np.roll(y_test, shift=1)
    y_persistence[0] = y_val[-1] if len(y_val) > 0 else y_train[-1]
    
    # 2. Zero-Return Benchmark (y_hat_t = 0)
    y_zero = np.zeros_like(y_test)
    
    # 3. Historical-Mean (Expanding window)
    y_mean = np.zeros_like(y_test)
    historical_data = np.concatenate([y_train, y_val]).tolist()
    for i in range(n_test):
        y_mean[i] = np.mean(historical_data)
        historical_data.append(y_test[i]) # Expand window
        
    return y_persistence, y_zero, y_mean

def main():
    print("================================================================================")
    print("RUNNING EXPERIMENT: TRADITIONAL ML & NAIVE BASELINES (MODEL 0)")
    print("================================================================================\n")

    data_file = base_dir / "data" / "processed" / "daily_processed_features.csv"
    if not data_file.exists():
        print(f"Error: Data file not found at {data_file}")
        sys.exit(1)

    print("1. Initializing Chronological Data Pipeline...")
    dataloaders, raw_arrays, _ = prepare_dataloaders(str(data_file))

    # Flatten tensors for Scikit-Learn
    X_train = dataloaders['train'].dataset.X.numpy().reshape(len(dataloaders['train'].dataset), -1)
    y_train = dataloaders['train'].dataset.y.numpy()
    X_val = dataloaders['val'].dataset.X.numpy().reshape(len(dataloaders['val'].dataset), -1)
    y_val = dataloaders['val'].dataset.y.numpy()
    X_test = dataloaders['test'].dataset.X.numpy().reshape(len(dataloaders['test'].dataset), -1)
    y_test = dataloaders['test'].dataset.y.numpy()

    print("\n2. Executing Naive Baselines...")
    y_pers, y_zero, y_mean = generate_naive_baselines(y_train, y_val, y_test)
    
    metrics_pers = calculate_point_metrics(y_test, y_pers)
    metrics_zero = calculate_point_metrics(y_test, y_zero)
    metrics_mean = calculate_point_metrics(y_test, y_mean)

    print("\n3. Executing Traditional ML Baselines...")
    rf_model, rf_metrics, rf_preds = train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test)
    xgb_model, xgb_metrics, xgb_preds = train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)

    print("\n4. Saving Evaluation Metrics...")
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "traditional_baseline_results.json"
    
    final_results = {
        "Naive_Persistence": metrics_pers,
        "Naive_ZeroReturn": metrics_zero,
        "Naive_HistoricalMean": metrics_mean,
        "RandomForest": rf_metrics,
        "XGBoost": xgb_metrics
    }
    
    with open(out_file, "w") as f:
        json.dump(final_results, f, indent=4)
        
    print(f"Metrics successfully saved to: {out_file}")

if __name__ == "__main__":
    main()