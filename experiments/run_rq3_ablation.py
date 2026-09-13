"""
experiments/run_rq3_ablation.py

Executes the experiment for RQ3: Feature Ablation (The Value of Risk Anchors).
Compares two models during Crisis Regimes:
1. Model 3 (Full Features)
2. Model 3_Ablated (Without GPR and Gold_VIX anchors)
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
from src.evaluation.metrics import get_regime_masks, evaluate_all_regimes

def get_test_risk_feature(csv_path: str, test_dates: np.ndarray, feature_name: str = 'Gold_VIX_Level') -> np.ndarray:
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
    if feature_name not in df.columns:
        fallback = [c for c in df.columns if 'GPR' in c or 'VIX' in c][0]
        feature_name = fallback
    return df.loc[test_dates, feature_name].values

def create_ablated_dataset(original_csv: Path, ablated_csv: Path):
    df = pd.read_csv(original_csv, index_col='Date', parse_dates=True)
    risk_cols = [col for col in df.columns if ('GPR' in col) or ('Gold_VIX' in col)]
    print(f"Ablating (Removing) Risk Anchors: {risk_cols}")
    df_ablated = df.drop(columns=risk_cols)
    df_ablated.to_csv(ablated_csv)

def run_ablation_pipeline(
    csv_path: Path, 
    model_name: str, 
    device: torch.device, 
    normal_mask: np.ndarray, 
    crisis_mask: np.ndarray
):
    dataloaders, raw_arrays, _ = prepare_dataloaders(
        csv_path=str(csv_path),
        target_col="Gold_Price_Return",
        window_size=21,
        batch_size=32
    )

    # Automatically derive the exact input feature count directly from the tensor
    input_size = dataloaders['train'].dataset.X.shape[2]

    print(f"\n{'-'*60}")
    print(f"Executing Pipeline: {model_name} (Derived Input Features: {input_size})")
    print(f"{'-'*60}")

    y_test_true = raw_arrays['y_test']

    model = CNN_GRU(
        input_size=input_size, 
        cnn_filters=32, 
        kernel_size=3, 
        gru_hidden_size=64, 
        gru_num_layers=1, 
        dropout_rate=0.2
    )

    trainer = ModelTrainer(model=model, dataloaders=dataloaders, device=device)
    trainer.train(model_name=model_name, num_epochs=100, learning_rate=1e-3, patience=15)

    y_mean, lower_bound, upper_bound = run_stochastic_inference(
        model=trainer.model, 
        dataloader=dataloaders['test'], 
        device=device, 
        T=200
    )

    metrics = evaluate_all_regimes(
        y_true=y_test_true,
        y_pred=y_mean,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        normal_mask=normal_mask,
        crisis_mask=crisis_mask
    )

    return metrics

def main():
    print(f"\n{'='*80}")
    print("RUNNING EXPERIMENT RQ3: FEATURE ABLATION (THE VALUE OF RISK ANCHORS)")
    print(f"{'='*80}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    proc_file = base_dir / "data" / "processed" / "daily_processed_features.csv"
    ablated_file = base_dir / "data" / "processed" / "daily_ablated_features.csv"
    
    if not proc_file.exists():
        print(f"Error: Processed file not found at {proc_file}")
        sys.exit(1)

    # 1. Regime Definition from Full Dataset
    _, raw_arrays_full, _ = prepare_dataloaders(str(proc_file), target_col="Gold_Price_Return", window_size=21, batch_size=32)
    test_dates = raw_arrays_full['test_dates']
    
    risk_array = get_test_risk_feature(str(proc_file), test_dates, feature_name='Gold_VIX_Level')
    normal_mask, crisis_mask = get_regime_masks(risk_array, threshold_percentile=90.0)
    
    # 2. Create Ablated Dataset
    create_ablated_dataset(proc_file, ablated_file)

    results = {}

    # 3. Run Full Model
    results["Model_3_Full"] = run_ablation_pipeline(
        csv_path=proc_file,
        model_name="CNN_GRU_Full_Features",
        device=device,
        normal_mask=normal_mask,
        crisis_mask=crisis_mask
    )

    # 4. Run Ablated Model
    results["Model_3_Ablated"] = run_ablation_pipeline(
        csv_path=ablated_file,
        model_name="CNN_GRU_Ablated",
        device=device,
        normal_mask=normal_mask,
        crisis_mask=crisis_mask
    )

    # 5. Display Key Findings (Delta Metrics)
    full_crisis_picp = results["Model_3_Full"]["Crisis_Uncertainty"]["PICP"]
    full_crisis_mpiw = results["Model_3_Full"]["Crisis_Uncertainty"]["MPIW"]
    
    ablated_crisis_picp = results["Model_3_Ablated"]["Crisis_Uncertainty"]["PICP"]
    ablated_crisis_mpiw = results["Model_3_Ablated"]["Crisis_Uncertainty"]["MPIW"]

    delta_picp = full_crisis_picp - ablated_crisis_picp
    delta_mpiw = full_crisis_mpiw - ablated_crisis_mpiw

    print(f"\n{'='*60}")
    print("RQ3 FINDINGS: ABLATION IMPACT DURING CRISIS REGIMES")
    print(f"{'='*60}")
    print(f"Full Model Crisis PICP    : {full_crisis_picp:.4f}")
    print(f"Ablated Model Crisis PICP : {ablated_crisis_picp:.4f}")
    print(f"-> Delta PICP (Coverage loss without anchors) : {delta_picp:+.4f}")
    print(f"\nFull Model Crisis MPIW    : {full_crisis_mpiw:.4f}")
    print(f"Ablated Model Crisis MPIW : {ablated_crisis_mpiw:.4f}")
    print(f"-> Delta MPIW (Interval shrinkage)          : {delta_mpiw:+.4f}")

    # 6. Save Results
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "rq3_ablation_results_2.json"
    
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nRQ3 Experiment Complete. Full results saved to: {out_file}")

    if ablated_file.exists():
        ablated_file.unlink()

if __name__ == "__main__":
    main()