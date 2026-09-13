"""
experiments/run_rq2_uncertainty.py

Executes the experiment for RQ2: Epistemic Uncertainty & Crisis Regimes.
Trains the core triad (GRU, Bi-LSTM, CNN-GRU), runs Monte Carlo Dropout inference (T=200),
and evaluates Prediction Interval Coverage (PICP) and Width (MPIW) across Normal and Crisis periods.
"""

import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to python path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.data.dataset import prepare_dataloaders
from src.models.recurrent import VanillaGRU, BidirectionalLSTM
from src.models.hybrid import CNN_GRU
from src.training.trainer import ModelTrainer
from src.evaluation.mc_engine import run_stochastic_inference
from src.evaluation.metrics import get_regime_masks, evaluate_all_regimes

def get_test_risk_feature(csv_path: str, test_dates: np.ndarray, feature_name: str = 'Gold_VIX_Level') -> np.ndarray:
    """
    Extracts the specified risk feature array for the Test set dates 
    to be used for regime stratification (Normal vs Crisis).
    """
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
    if feature_name not in df.columns:
        # Fallback to another risk anchor if Gold_VIX_Level is missing
        fallback = 'GPRD_THREAT_Level' if 'GPRD_THREAT_Level' in df.columns else df.columns[0]
        print(f"Warning: {feature_name} not found. Falling back to {fallback} for regime masking.")
        feature_name = fallback
        
    risk_series = df.loc[test_dates, feature_name]
    return risk_series.values

def main():
    print(f"\n{'='*80}")
    print("RUNNING EXPERIMENT RQ2: EPISTEMIC UNCERTAINTY & CRISIS REGIMES")
    print(f"{'='*80}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    data_file = base_dir / "data" / "processed" / "daily_processed_features.csv"
    if not data_file.exists():
        print(f"Error: Data file not found at {data_file}")
        sys.exit(1)

    # 1. Load Data
    dataloaders, raw_arrays, scaler = prepare_dataloaders(
        csv_path=str(data_file),
        target_col="Gold_Price_Return",
        window_size=21,
        batch_size=32
    )

    y_test_true = raw_arrays['y_test']
    test_dates = raw_arrays['test_dates']

    # 2. Generate Regime Masks (Crisis = Top 10% of Risk Feature)
    risk_array = get_test_risk_feature(str(data_file), test_dates, feature_name='Gold_VIX_Level')
    normal_mask, crisis_mask = get_regime_masks(risk_array, threshold_percentile=90.0)
    
    print(f"\nRegime Stratification Summary:")
    print(f"- Total Test Days : {len(y_test_true)}")
    print(f"- Normal Days     : {np.sum(normal_mask)}")
    print(f"- Crisis Days     : {np.sum(crisis_mask)} (>= 90th percentile of risk anchor)")

    # 3. Initialize Core Triad Models
    # Assuming 13 features in the processed dataset
    input_features = 13 
    
    models = {
        "Vanilla_GRU": VanillaGRU(input_size=input_features, hidden_size=64, num_layers=1, dropout_rate=0.2),
        "Bi_LSTM": BidirectionalLSTM(input_size=input_features, hidden_size=64, num_layers=1, dropout_rate=0.2),
        "CNN_GRU": CNN_GRU(input_size=input_features, cnn_filters=32, kernel_size=3, gru_hidden_size=64, gru_num_layers=1, dropout_rate=0.2)
    }

    results = {}

    # 4. Train, Infer, and Evaluate Each Model
    for model_name, model in models.items():
        print(f"\n{'-'*60}")
        print(f"Evaluating Model: {model_name}")
        print(f"{'-'*60}")

        # Train
        trainer = ModelTrainer(model=model, dataloaders=dataloaders, device=device)
        trainer.train(model_name=model_name, num_epochs=100, learning_rate=1e-3, patience=15)

        # MC Dropout Inference (T=200)
        y_mean, lower_bound, upper_bound = run_stochastic_inference(
            model=trainer.model, 
            dataloader=dataloaders['test'], 
            device=device, 
            T=200
        )

        # Evaluate across all regimes
        model_metrics = evaluate_all_regimes(
            y_true=y_test_true,
            y_pred=y_mean,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            normal_mask=normal_mask,
            crisis_mask=crisis_mask
        )

        results[model_name] = model_metrics

        # Print quick summary for the console
        print(f"\n[Quick Summary: {model_name}]")
        print(f"Overall PICP : {model_metrics['Overall_Uncertainty']['PICP']:.4f} (Target ~0.90)")
        print(f"Overall MPIW : {model_metrics['Overall_Uncertainty']['MPIW']:.4f}")
        print(f"Crisis PICP  : {model_metrics['Crisis_Uncertainty']['PICP']:.4f}")
        print(f"Crisis MPIW  : {model_metrics['Crisis_Uncertainty']['MPIW']:.4f}")

    # 5. Save Results
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "rq2_uncertainty_results_2.json"
    
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nRQ2 Experiment Complete. Full results saved to: {out_file}")

if __name__ == "__main__":
    main()