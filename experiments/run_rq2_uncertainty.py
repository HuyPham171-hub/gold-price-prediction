"""
experiments/run_rq2_uncertainty.py

Executes RQ2: Epistemic Uncertainty & Crisis Regimes.
Applies Frozen Thresholds to prevent look-ahead bias, evaluates UQ via empirical quantiles, 
and benchmarks conditional variance against GARCH models.
"""

import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import random
import os

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.data.dataset import prepare_dataloaders
from src.models.recurrent import VanillaGRU, UnidirectionalLSTM
from src.models.hybrid import CNN_GRU
from src.training.trainer import ModelTrainer
from src.evaluation.mc_engine import run_stochastic_inference
from src.evaluation.metrics import evaluate_all_regimes
from src.models.econometric_baselines import EconometricVolatilityBaselines, evaluate_volatility_forecasts

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_frozen_regime_masks(csv_path: str, train_dates: np.ndarray, test_dates: np.ndarray, anchor: str = 'Gold_VIX_Level') -> tuple:
    """Computes Crisis threshold exclusively on the Train set and applies it to the Test set."""
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
    
    train_vals = df.loc[train_dates, anchor].values
    frozen_thresh = float(np.percentile(train_vals, 90.0))
    
    test_vals = df.loc[test_dates, anchor].values
    crisis_mask = test_vals >= frozen_thresh
    normal_mask = ~crisis_mask
    
    return normal_mask, crisis_mask, frozen_thresh

def main():
    seed_everything(42)
    print(f"\n{'='*80}")
    print("RUNNING EXPERIMENT RQ2: UNCERTAINTY & CRISIS REGIMES")
    print(f"{'='*80}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_file = base_dir / "data" / "processed" / "daily_processed_features.csv"

    dataloaders, raw_arrays, scaler = prepare_dataloaders(str(data_file), target_col="Gold_Price_Return")
    
    # We must inverse-transform the raw arrays for the GARCH models and Ground Truth evaluation
    # Extract the target scaler parameters
    s_y = scaler.scale_[0]
    m_y = scaler.mean_[0]
    
    y_train_unscaled = (raw_arrays['y_train'] * s_y) + m_y
    y_test_unscaled = (raw_arrays['y_test'] * s_y) + m_y
    
    # 1. Generate Regime Masks (Frozen Threshold)
    normal_mask, crisis_mask, frozen_threshold = get_frozen_regime_masks(
        str(data_file), raw_arrays['train_dates'], raw_arrays['test_dates'], anchor='Gold_VIX_Level'
    )
    print(f"Frozen 90th Percentile Threshold (GVZ): {frozen_threshold:.4f}")

    # 2. Run Econometric Volatility Baselines (On UNSCALED returns)
    print("\nExecuting Volatility Baselines (GARCH family)...")
    vol_baselines = EconometricVolatilityBaselines(y_train_unscaled, y_test_unscaled)
    vol_baselines.run_garch_1_1()
    vol_baselines.run_gjr_garch()
    vol_baselines.run_egarch()
    
    garch_forecasts = vol_baselines.get_all_forecasts()
    vol_metrics = evaluate_volatility_forecasts(y_test_unscaled, garch_forecasts)

    # 3. Deep Learning Core Triad
    # Derive input size directly from the dataset
    sample_x, _ = next(iter(dataloaders['train']))
    input_size = sample_x.shape[2]
    
    models = {
        "UnidirectionalLSTM": UnidirectionalLSTM(input_size=input_size),
        "VanillaGRU": VanillaGRU(input_size=input_size),
        "CNN_GRU_Gaussian": CNN_GRU(input_size=input_size)
    }

    results = {"Volatility_Baselines": vol_metrics, "Deep_Learning": {}}

    for model_name, model in models.items():
        print(f"\nEvaluating Model: {model_name}")
        trainer = ModelTrainer(model=model, dataloaders=dataloaders, device=device)
        trainer.train(model_name=model_name, num_epochs=80, patience=15)

        # MC Dropout Inference (Inversion scaling is handled inside the engine now)
        y_mean, lower_bound, upper_bound, variance_pred = run_stochastic_inference(
            model=trainer.model, 
            dataloader=dataloaders['test'], 
            device=device, 
            scaler_target=scaler, # Pass scaler for automatic inversion
            T=200
        )
        
        # Evaluate metrics against UNSCALED ground truth
        model_metrics = evaluate_all_regimes(
            y_true=y_test_unscaled, 
            y_pred=y_mean, 
            lower_bound=lower_bound, 
            upper_bound=upper_bound, 
            normal_mask=normal_mask, 
            crisis_mask=crisis_mask, 
            variance_pred=variance_pred # Direct aleatoric variance output
        )
        results["Deep_Learning"][model_name] = model_metrics
        
        print(f"  -> Overall PICP: {model_metrics['Overall_Uncertainty']['PICP']:.4f}")
        print(f"  -> Crisis PICP:  {model_metrics['Crisis_Uncertainty'].get('PICP', 0):.4f}")

    out_file = base_dir / "results" / "rq2_uncertainty_results.json"
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nRQ2 Complete. Results saved to: {out_file}")

if __name__ == "__main__":
    main()