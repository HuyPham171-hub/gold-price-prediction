"""
experiments/run_rq1_stationarity.py

Executes RQ1: The Trivial Persistence Mimicry Trap.
Compares a deep learning model (CNN-GRU) trained on stationary log-returns (I(0))
against one trained on non-stationary price levels (I(1)).
Incorporates Clark-West test and a trifecta of unit-root tests (ADF, PP, KPSS).
"""

import sys
import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.stats.stattools import durbin_watson

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.data.dataset import prepare_dataloaders
from src.models.hybrid import CNN_GRU
from src.training.trainer import ModelTrainer
from src.evaluation.metrics import calculate_point_metrics, clark_west_test, run_stationarity_tests

def evaluate_econometrics(y_true: np.ndarray, y_pred: np.ndarray, y_val_last: float, series_name: str) -> dict:
    """Calculates standard regression metrics alongside advanced econometric diagnostics."""
    point_metrics = calculate_point_metrics(y_true, y_pred)
    
    residuals = y_true - y_pred
    dw_stat = durbin_watson(residuals)
    stationarity_results = run_stationarity_tests(residuals)
    
    # Naive Persistence Benchmark (y_hat_t = y_{t-1})
    y_naive = np.roll(y_true, shift=1)
    y_naive[0] = y_val_last
    
    naive_metrics = calculate_point_metrics(y_true, y_naive)
    
    # Clark-West Test for nested models
    cw_stat, cw_pval = clark_west_test(y_true, y_naive, y_pred)
    
    print(f"\n--- Econometric Report: {series_name} ---")
    print(f"R2 Score    : {point_metrics['R2']:.4f}")
    print(f"RMSE (DL)   : {point_metrics['RMSE']:.4f} | RMSE (Naive): {naive_metrics['RMSE']:.4f}")
    print(f"DW Stat     : {dw_stat:.4f} (Target ~ 2.0)")
    print(f"ADF p-value : {stationarity_results['ADF_pval']:.4f} (Target < 0.05)")
    print(f"CW p-value  : {cw_pval:.4f} (Target < 0.05 to beat Naive)")
    
    return {
        "Point_Metrics": point_metrics,
        "Residual_Diagnostics": {
            "Durbin_Watson": float(dw_stat),
            "ADF_pvalue": float(stationarity_results['ADF_pval']),
            "PP_pvalue": float(stationarity_results['PP_pval']),
            "KPSS_pvalue": float(stationarity_results['KPSS_pval'])
        },
        "Naive_Benchmark": naive_metrics,
        "Clark_West_Test": {
            "Statistic": float(cw_stat),
            "p_value": float(cw_pval),
            "Beats_Naive": bool(cw_pval < 0.05 and point_metrics['RMSE'] < naive_metrics['RMSE'])
        }
    }

def run_pipeline(csv_path: str, target_col: str, model_name: str, device: torch.device):
    """Trains and extracts predictions for a specific target series."""
    dataloaders, raw_arrays, _ = prepare_dataloaders(csv_path, target_col=target_col)
    
    sample_x, _ = next(iter(dataloaders['train']))
    input_size = int(sample_x.shape[2])
    model = CNN_GRU(input_size=input_size).to(device)
    trainer = ModelTrainer(model=model, dataloaders=dataloaders, device=device)
    
    trainer.train(model_name=model_name, num_epochs=50, patience=10)
    
    model.eval()
    y_pred_list, y_true_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloaders['test']:
            mu, _ = model(X_batch.to(device))
            y_pred_list.append(mu.cpu().numpy())
            y_true_list.append(y_batch.numpy())
            
    y_pred = np.concatenate(y_pred_list, axis=0).squeeze()
    y_true = np.concatenate(y_true_list, axis=0).squeeze()
    
    y_val_last = raw_arrays['y_val'][-1]
    
    return evaluate_econometrics(y_true, y_pred, y_val_last, model_name)

def main():
    print("================================================================================")
    print("RUNNING EXPERIMENT RQ1: STATIONARITY & PERSISTENCE MIMICRY")
    print("================================================================================\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_file = base_dir / "data" / "processed" / "daily_processed_features.csv"
    raw_file = base_dir / "data" / "raw" / "daily_raw_features.csv"
    
    results = {}
    
    # -------------------------------------------------------------------------
    # 1. Model A: Stationary Returns
    # -------------------------------------------------------------------------
    print("\n>>> Executing Model A (Stationary Returns)...")
    results["Stationary_Returns"] = run_pipeline(
        str(processed_file), target_col="Gold_Price_Return", model_name="Model_A_Returns", device=device
    )
    
    # -------------------------------------------------------------------------
    # 2. Model B: Non-Stationary Prices
    # -------------------------------------------------------------------------
    print("\n>>> Preparing Dataset for Model B (Non-Stationary Prices)...")
    df_proc = pd.read_csv(processed_file, index_col='Date', parse_dates=True)
    df_raw = pd.read_csv(raw_file, index_col='Date', parse_dates=True)
    
    # Locate the gold price column in daily_raw_features.csv
    raw_price_col = None
    for candidate in ['Gold_Price', 'Gold_Price_Close', 'Gold_Close', 'Close', 'Price']:
        if candidate in df_raw.columns:
            raw_price_col = candidate
            break
            
    if raw_price_col is None:
        raise KeyError(f"Gold price column not found in {raw_file}. Available columns: {list(df_raw.columns)}")
        
    print(f"Selected raw price column: '{raw_price_col}'")
    
    # Inner join on Date to remove holiday/trading-day discrepancies
    df_proc['Gold_Price_Raw'] = df_raw[raw_price_col]
    df_proc = df_proc.dropna(subset=['Gold_Price_Raw'])
    
    # Save temporary file for Model B
    temp_b_path = base_dir / "data" / "processed" / "temp_daily_price_experiment.csv"
    df_proc.to_csv(temp_b_path)
    
    try:
        print("\n>>> Executing Model B (Non-Stationary Prices)...")
        results["NonStationary_Prices"] = run_pipeline(
            str(temp_b_path), target_col="Gold_Price_Raw", model_name="Model_B_Prices", device=device
        )
    finally:
        # Delete temporary file after execution to keep repository clean
        if temp_b_path.exists():
            temp_b_path.unlink()
    
    # Save econometric results
    out_file = base_dir / "results" / "rq1_stationarity_econometrics.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nRQ1 Econometric Analysis complete. Results saved to {out_file}")

if __name__ == "__main__":
    main()