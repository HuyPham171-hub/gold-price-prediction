"""
experiments/run_mc_convergence.py

Executes the Monte Carlo Dropout Convergence Test.
Validates the stability of predictive mean, PICP, and MPIW across increasing 
stochastic forward passes (T = 20, 50, 100, 200, 500).
Generates a convergence plot to empirically justify the choice of T=200 
as the optimal trade-off between computational cost and uncertainty stability.
"""

import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.data.dataset import prepare_dataloaders
from src.models.hybrid import CNN_GRU
from src.training.trainer import ModelTrainer
from src.evaluation.mc_engine import run_stochastic_inference
from src.evaluation.metrics import calculate_uncertainty_metrics

def main():
    print(f"\n{'='*80}")
    print("RUNNING EXPERIMENT: MC DROPOUT CONVERGENCE TEST")
    print(f"{'='*80}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_file = base_dir / "data" / "processed" / "daily_processed_features.csv"

    # 1. Load Data
    dataloaders, raw_arrays, scaler = prepare_dataloaders(str(data_file), target_col="Gold_Price_Return")
    input_size = dataloaders['train'].dataset.X.shape[2]
    
    # Create a subset of the test dataloader (e.g., first 100 samples) to speed up T=500
    subset_size = min(100, len(dataloaders['test'].dataset))
    subset_indices = list(range(subset_size))
    test_subset = torch.utils.data.Subset(dataloaders['test'].dataset, subset_indices)
    test_subset_loader = torch.utils.data.DataLoader(test_subset, batch_size=32, shuffle=False)
    
    # Strictly align true labels directly from the subset dataset to avoid index-shift bugs
    s_y = scaler.scale_[0]
    m_y = scaler.mean_[0]
    y_test_subset_scaled = np.array([test_subset[i][1].item() for i in range(len(test_subset))])
    y_test_true = (y_test_subset_scaled * s_y) + m_y

    # 2. Train a base model
    print("Training base CNN-GRU model for convergence testing...")
    model = CNN_GRU(input_size=input_size).to(device)
    trainer = ModelTrainer(model=model, dataloaders=dataloaders, device=device)
    
    trainer.train(model_name="CNN_GRU_Convergence", num_epochs=30, patience=5)

    # 3. Convergence Testing Loop
    T_list = [20, 50, 100, 200, 500]
    results = {
        'T': [],
        'PICP': [],
        'MPIW': [],
        'Time_Seconds': []
    }

    print("\nStarting Stochastic Inference across different T values...")
    for T in T_list:
        print(f"Testing T = {T}...")
        start_time = time.time()
        
        y_mean, lower_bound, upper_bound, _ = run_stochastic_inference(
            model=trainer.model, 
            dataloader=test_subset_loader, 
            device=device, 
            scaler_target=scaler,
            T=T,
            confidence_level=0.90
        )
        
        exec_time = time.time() - start_time
        
        metrics = calculate_uncertainty_metrics(
            y_true=y_test_true, 
            lower_bound=lower_bound, 
            upper_bound=upper_bound,
            variance_pred=None 
        )
        
        results['T'].append(T)
        results['PICP'].append(metrics['PICP'])
        results['MPIW'].append(metrics['MPIW'])
        results['Time_Seconds'].append(exec_time)
        
        print(f"  -> PICP: {metrics['PICP']:.4f} | MPIW: {metrics['MPIW']:.4f} | Time: {exec_time:.2f}s")

    # 4. Generate Academic Dual-Panel Convergence Plot
    print("\nGenerating Convergence Plot...")
    results_dir = base_dir / "results" / "convergence"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # Panel 1: Uncertainty Metrics Stability (PICP and MPIW)
    color1 = '#1f77b4' # Blue
    color2 = '#2ca02c' # Green
    
    ax1.set_ylabel('PICP (Coverage)', color=color1, fontsize=11)
    ln1 = ax1.plot(results['T'], results['PICP'], marker='o', color=color1, label='PICP (Nominal = 0.90)', linewidth=2)
    ax1.axhline(0.90, color='gray', linestyle=':', label='Target 90% Bound')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel('MPIW (Interval Width)', color=color2, fontsize=11)
    ln2 = ax1_twin.plot(results['T'], results['MPIW'], marker='s', linestyle='--', color=color2, label='MPIW (Sharpness)', linewidth=2)
    ax1_twin.tick_params(axis='y', labelcolor=color2)

    # Combine legends for Panel 1
    lns1 = ln1 + ln2
    labs1 = [l.get_label() for l in lns1]
    ax1.legend(lns1, labs1, loc='center right')
    ax1.set_title('Monte Carlo Dropout Convergence Test (T vs. Predictive Uncertainty)', fontsize=12)

    # Panel 2: Computational Cost
    color3 = '#d62728' # Red
    ax2.plot(results['T'], results['Time_Seconds'], marker='^', linestyle='-', color=color3, label='Runtime (s)', linewidth=2)
    ax2.set_xlabel('Number of Stochastic Passes (T)', fontsize=11)
    ax2.set_ylabel('Inference Time (Seconds)', color=color3, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=color3)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left')

    fig.tight_layout()
    
    plot_path = results_dir / "mc_convergence_plot.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Convergence plot saved to: {plot_path}")
    print("Experiment Complete.")

if __name__ == "__main__":
    main()