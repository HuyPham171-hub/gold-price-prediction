"""
experiments/run_rq1_stationarity.py

Executes the experiment for RQ1: Stationarity & Spurious Regression.
Compares:
1. Target = Stationary Log-Returns (Gold_Price_Return) from processed data
2. Target = Non-Stationary Price Levels (Gold_Price) aligned from raw data

Calculates Train/Test metrics and the Durbin-Watson statistic to demonstrate 
the failure of models trained on non-stationary price levels.
"""

import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Add project root to python path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.data.dataset import prepare_dataloaders, create_sliding_windows, TimeSeriesWindowDataset
from src.models.hybrid import CNN_GRU
from src.training.trainer import ModelTrainer
from src.evaluation.metrics import calculate_point_metrics
from torch.utils.data import DataLoader

def calculate_durbin_watson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Durbin-Watson statistic for residuals.
    DW ~ 2.0 indicates no autocorrelation (healthy model on stationary series).
    DW -> 0 indicates severe positive autocorrelation (spurious regression).
    """
    residuals = y_true.ravel() - y_pred.ravel()
    diff_residuals = np.diff(residuals)
    sum_sq_diff = np.sum(diff_residuals ** 2)
    sum_sq_res = np.sum(residuals ** 2)
    if sum_sq_res == 0:
        return 0.0
    return round(float(sum_sq_diff / sum_sq_res), 4)

def run_evaluation(model, dataloader, device) -> tuple:
    """Generates predictions and extracts ground truth for a dataloader."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())
            
    return np.concatenate(all_targets, axis=0), np.concatenate(all_preds, axis=0)

def prepare_price_level_dataloaders(
    processed_path: str,
    raw_path: str,
    window_size: int = 21,
    batch_size: int = 32,
    train_end_date: str = '2021-12-31',
    val_end_date: str = '2023-12-31'
):
    """
    Builds dataloaders for the Spurious Regression experiment:
    X: 13 features (scaled per split)
    y: Raw Gold_Price (scaled per split)
    """
    df_proc = pd.read_csv(processed_path, index_col='Date', parse_dates=True)
    df_raw = pd.read_csv(raw_path, index_col='Date', parse_dates=True)
    
    # Align dates between processed features and raw gold price
    common_idx = df_proc.index.intersection(df_raw.index)
    df_features = df_proc.loc[common_idx].copy()
    raw_prices = df_raw.loc[common_idx, ['Gold_Price']].copy()

    # Split indices chronologically
    train_mask = df_features.index <= train_end_date
    val_mask = (df_features.index > train_end_date) & (df_features.index <= val_end_date)
    test_mask = df_features.index > val_end_date

    # Scale features and target strictly on the Train set
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = scaler_x.fit_transform(df_features.loc[train_mask].values)
    y_train_s = scaler_y.fit_transform(raw_prices.loc[train_mask].values)

    X_val_s = scaler_x.transform(df_features.loc[val_mask].values)
    y_val_s = scaler_y.transform(raw_prices.loc[val_mask].values)

    X_test_s = scaler_x.transform(df_features.loc[test_mask].values)
    y_test_s = scaler_y.transform(raw_prices.loc[test_mask].values)

    # Pad boundary windows for continuity
    X_val_padded = np.vstack([X_train_s[-window_size:], X_val_s])
    y_val_padded = np.vstack([y_train_s[-window_size:], y_val_s])

    X_test_padded = np.vstack([X_val_s[-window_size:], X_test_s])
    y_test_padded = np.vstack([y_val_s[-window_size:], y_test_s])

    # Construct 3D sliding windows
    X_tr, y_tr = create_sliding_windows(np.hstack([X_train_s, y_train_s]), target_idx=-1, window_size=window_size)
    X_va, y_va = create_sliding_windows(np.hstack([X_val_padded, y_val_padded]), target_idx=-1, window_size=window_size)
    X_te, y_te = create_sliding_windows(np.hstack([X_test_padded, y_test_padded]), target_idx=-1, window_size=window_size)

    # Drop target column from X features
    X_tr = X_tr[:, :, :-1]
    X_va = X_va[:, :, :-1]
    X_te = X_te[:, :, :-1]

    dataloaders = {
        'train': DataLoader(TimeSeriesWindowDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True),
        'val': DataLoader(TimeSeriesWindowDataset(X_va, y_va), batch_size=batch_size, shuffle=False),
        'test': DataLoader(TimeSeriesWindowDataset(X_te, y_te), batch_size=batch_size, shuffle=False)
    }
    return dataloaders

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    proc_file = base_dir / "data" / "processed" / "daily_processed_features.csv"
    raw_file = base_dir / "data" / "raw" / "daily_raw_features.csv"

    if not proc_file.exists() or not raw_file.exists():
        print("Required data files not found. Check both data/processed/ and data/raw/.")
        sys.exit(1)

    results = {}

    # -------------------------------------------------------------
    # Experiment 1: Model 3 on Stationary Log-Returns
    # -------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PHASE 1: CNN-GRU on Stationary Log-Returns (Gold_Price_Return)")
    print(f"{'='*60}")
    loaders_returns, _, _ = prepare_dataloaders(
        csv_path=str(proc_file),
        target_col="Gold_Price_Return",
        window_size=21,
        batch_size=32
    )

    model_returns = CNN_GRU(input_size=13, cnn_filters=32, kernel_size=3, gru_hidden_size=64, gru_num_layers=1, dropout_rate=0.2)
    trainer_returns = ModelTrainer(model=model_returns, dataloaders=loaders_returns, device=device)
    trainer_returns.train(model_name="CNN_GRU_Returns", num_epochs=100, learning_rate=1e-3, patience=15)

    y_train_ret, p_train_ret = run_evaluation(trainer_returns.model, loaders_returns['train'], device)
    y_test_ret, p_test_ret = run_evaluation(trainer_returns.model, loaders_returns['test'], device)

    train_ret_metrics = calculate_point_metrics(y_train_ret, p_train_ret)
    test_ret_metrics = calculate_point_metrics(y_test_ret, p_test_ret)
    test_ret_metrics["Durbin_Watson"] = calculate_durbin_watson(y_test_ret, p_test_ret)

    print(f"Returns Train Metrics: {train_ret_metrics}")
    print(f"Returns Test Metrics : {test_ret_metrics}")

    results["Stationary_Returns"] = {
        "Train": train_ret_metrics,
        "Test": test_ret_metrics
    }

    # -------------------------------------------------------------
    # Experiment 2: Model 3_PriceLevel on Non-Stationary Prices
    # -------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PHASE 2: CNN-GRU on Non-Stationary Price Levels (Gold_Price)")
    print(f"{'='*60}")
    loaders_price = prepare_price_level_dataloaders(
        processed_path=str(proc_file),
        raw_path=str(raw_file),
        window_size=21,
        batch_size=32
    )

    model_price = CNN_GRU(input_size=13, cnn_filters=32, kernel_size=3, gru_hidden_size=64, gru_num_layers=1, dropout_rate=0.2)
    trainer_price = ModelTrainer(model=model_price, dataloaders=loaders_price, device=device)
    trainer_price.train(model_name="CNN_GRU_PriceLevel", num_epochs=100, learning_rate=1e-3, patience=15)

    y_train_pri, p_train_pri = run_evaluation(trainer_price.model, loaders_price['train'], device)
    y_test_pri, p_test_pri = run_evaluation(trainer_price.model, loaders_price['test'], device)

    train_pri_metrics = calculate_point_metrics(y_train_pri, p_train_pri)
    test_pri_metrics = calculate_point_metrics(y_test_pri, p_test_pri)
    test_pri_metrics["Durbin_Watson"] = calculate_durbin_watson(y_test_pri, p_test_pri)

    print(f"Price Level Train Metrics: {train_pri_metrics}")
    print(f"Price Level Test Metrics : {test_pri_metrics}")

    results["NonStationary_Prices"] = {
        "Train": train_pri_metrics,
        "Test": test_pri_metrics
    }

    # -------------------------------------------------------------
    # Save RQ1 Consolidated Report
    # -------------------------------------------------------------
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "rq1_stationarity_results.json"

    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nRQ1 experiment complete. Results written to: {out_file}")

if __name__ == "__main__":
    main()