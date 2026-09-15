"""
experiments/run_rq3_ablation.py

Executes RQ3: Nested Feature Ablation (M0 to M4) and Variance Explainability.
Isolates the contribution of GVZ and GPR in expanding intervals during crises.
Generates SHAP and PDP plots exclusively for the predicted log-variance.
"""

import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.data.dataset import prepare_dataloaders
from src.models.hybrid import CNN_GRU
from src.training.trainer import ModelTrainer
from src.evaluation.mc_engine import run_stochastic_inference
from src.evaluation.metrics import evaluate_all_regimes
from src.evaluation.explainability import run_variance_shap, generate_variance_pdp
from experiments.run_rq2_uncertainty import get_frozen_regime_masks
import random
import os

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_nested_ablation(csv_path: str, model_name: str, keep_features: list, device: torch.device, normal_mask, crisis_mask):
    # In-memory ablation by passing keep_features to the dataloader
    dataloaders, raw_arrays, scaler = prepare_dataloaders(
        csv_path=csv_path, target_col="Gold_Price_Return", keep_cols=keep_features
    )
    
    # Infer input_size dynamically from the instantiated training tensor
    sample_x, _ = next(iter(dataloaders['train']))
    input_size = int(sample_x.shape[2])
    
    print(f"\n--- Training {model_name} (Derived Input Size: {input_size}) ---")
    
    model = CNN_GRU(input_size=input_size).to(device)
    trainer = ModelTrainer(model=model, dataloaders=dataloaders, device=device)
    
    trainer.train(model_name=model_name, num_epochs=80, patience=15)
    
    y_mean, lower_bound, upper_bound, variance_pred = run_stochastic_inference(
        model=trainer.model, dataloader=dataloaders['test'], device=device, scaler_target=scaler, T=200
    )
    
    # Invert target scaling to evaluate predictions in original return space
    s_y = scaler.scale_[0]
    m_y = scaler.mean_[0]
    y_test_unscaled = (raw_arrays['y_test'] * s_y) + m_y

    metrics = evaluate_all_regimes(
        y_true=y_test_unscaled, 
        y_pred=y_mean, 
        lower_bound=lower_bound, 
        upper_bound=upper_bound, 
        normal_mask=normal_mask, 
        crisis_mask=crisis_mask,
        variance_pred=variance_pred
    )
    actual_tensor_features = list(dict.fromkeys(keep_features + ["Gold_Price_Return"]))
    return metrics, trainer.model, dataloaders, actual_tensor_features

def main():
    seed_everything(42)
    print(f"\n{'='*80}")
    print("RUNNING EXPERIMENT RQ3: NESTED ABLATION & VARIANCE EXPLAINABILITY")
    print(f"{'='*80}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_file = base_dir / "data" / "processed" / "daily_processed_features.csv"

    # Extract actual column names from the dataset to avoid missing feature errors
    df_sample = pd.read_csv(data_file, nrows=1, index_col='Date')
    all_features = [c for c in df_sample.columns if c != 'Gold_Price_Return']

    # Automatically partition features into blocks to prevent hardcoded naming mismatches
    gvz_cols = [c for c in all_features if 'VIX' in c or 'GVZ' in c]
    gpr_cols = [c for c in all_features if 'GPR' in c]
    gold_cols = [c for c in all_features if 'Gold' in c and c not in gvz_cols]
    macro_cols = [c for c in all_features if c not in gvz_cols and c not in gpr_cols and c not in gold_cols]

    # Construct the nested ablation hierarchy (M0 through M4)
    m0_features = gold_cols if len(gold_cols) > 0 else [all_features[0]]
    m1_features = list(dict.fromkeys(m0_features + macro_cols))
    m2_features = list(dict.fromkeys(m1_features + gvz_cols))
    m3_features = list(dict.fromkeys(m1_features + gpr_cols))
    m4_features = all_features

    ablation_matrix = {
        "M0_GoldOnly": m0_features,
        "M1_Macro": m1_features,
        "M2_Macro_GVZ": m2_features,
        "M3_Macro_GPR": m3_features,
        "M4_Full": m4_features
    }

    # 1. Regime Identification using Frozen In-Sample GVZ Threshold
    _, raw_arrays_full, _ = prepare_dataloaders(str(data_file), target_col="Gold_Price_Return")
    normal_mask, crisis_mask, _ = get_frozen_regime_masks(
        str(data_file), raw_arrays_full['train_dates'], raw_arrays_full['test_dates'], anchor='Gold_VIX_Level'
    )

    results = {}
    full_model = None
    full_dataloaders = None
    actual_full_features = None

    # 2. Iterate and train through each stage of the nested ablation matrix
    for name, features in ablation_matrix.items():
        metrics, model, dl, used_features = run_nested_ablation(
            str(data_file), name, features, device, normal_mask, crisis_mask
        )
        results[name] = metrics
        
        if name == "M4_Full":
            full_model = model
            full_dataloaders = dl
            actual_full_features = used_features

    # 3. Activate explainability diagnostics targeting the variance channel (SHAP & PDP)
    if full_model is not None:
        print("\n--- Running Variance Explainability (SHAP & PDP) ---")
        train_batch_x, _ = next(iter(full_dataloaders['train']))
        test_batch_x, _ = next(iter(full_dataloaders['test']))
        
        bg_samples = train_batch_x.to(device)
        test_samples = test_batch_x[:min(50, len(test_batch_x))].to(device)
        
        explain_dir = base_dir / "results" / "explainability"
        run_variance_shap(
            full_model, bg_samples, test_samples, actual_full_features, save_dir=str(explain_dir)
        )
        
        # Generate Partial Dependence Plots for risk anchors available in the dataset
        for anchor_name in ['Gold_VIX_Level', 'GPRD_THREAT_Level']:
            if anchor_name in actual_full_features:
                idx = actual_full_features.index(anchor_name)
                generate_variance_pdp(
                    full_model, test_samples, idx, anchor_name, save_dir=str(explain_dir)
                )

    # 4. Export RQ3 empirical results to JSON
    out_file = base_dir / "results" / "rq3_ablation_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nRQ3 Experiment Complete. Full results saved to: {out_file}")

if __name__ == "__main__":
    main()