"""
src/evaluation/explainability.py

Advanced explainability module for Dual-Head Heteroskedastic networks.
Isolates the Variance (Aleatoric Uncertainty) head to quantify the exact contribution 
of risk anchors (e.g., GVZ, GPR) to the dynamic expansion of prediction intervals.
Includes SHAP (GradientExplainer), Partial Dependence Plots (PDP), and Permutation Importance.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# =====================================================================
# 1. MODEL WRAPPER (ISOLATING THE VARIANCE HEAD)
# =====================================================================

class VarianceHeadWrapper(nn.Module):
    """
    Wraps the Dual-Head model to output ONLY the predicted log-variance.
    This is required for SHAP and standard explainability tools which expect a scalar output.
    """
    def __init__(self, base_model: nn.Module):
        super(VarianceHeadWrapper, self).__init__()
        self.base_model = base_model
        self.base_model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the base model
        _, sigma_sq = self.base_model(x)
        # Return logvar to ensure symmetric and numerically stable explanations
        logvar = torch.log(sigma_sq)
        return logvar

# =====================================================================
# 2. SHAP ANALYSIS
# =====================================================================

def run_variance_shap(
    model: nn.Module, 
    background_data: torch.Tensor, 
    test_data: torch.Tensor, 
    feature_names: List[str],
    save_dir: str = "results/explainability"
):
    """
    Applies SHAP (GradientExplainer) strictly to the variance head to rank feature importance.
    Since input is 3D (Batch, Timesteps, Features), SHAP values are aggregated across time.
    """
    logger.info("Initializing SHAP GradientExplainer for the Variance Head...")
    os.makedirs(save_dir, exist_ok=True)
    
    wrapper = VarianceHeadWrapper(model)
    
    # GradientExplainer is more stable than DeepExplainer for PyTorch RNN/GRU architectures
    explainer = shap.GradientExplainer(wrapper, background_data)
    
    logger.info(f"Computing SHAP values for {test_data.size(0)} test samples...")
    
    # =========================================================================
    # STRICT FIX: Disable cuDNN for RNN backward pass during eval() mode.
    # This is required for PyTorch GRU/LSTM layers when calculating SHAP gradients.
    # =========================================================================
    with torch.backends.cudnn.flags(enabled=False):
        shap_values = explainer.shap_values(test_data)
        
    # Handle cases where GradientExplainer wraps outputs in a list
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        
    shap_values = np.array(shap_values)

    # Squeeze the trailing singleton dimension if present: (Batch, Timesteps, Features, 1) -> (Batch, Timesteps, Features)
    if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
        shap_values = np.squeeze(shap_values, axis=-1)

    # Aggregate across the temporal axis (axis 1): -> (Batch, Features)
    shap_values_2d = np.sum(shap_values, axis=1)
    
    # Prepare 2D test feature matrix for summary plotting: -> (Batch, Features)
    test_data_2d = torch.mean(test_data, dim=1).detach().cpu().numpy()
    if test_data_2d.ndim == 3 and test_data_2d.shape[-1] == 1:
        test_data_2d = np.squeeze(test_data_2d, axis=-1)

    # 1. Generate SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_2d, 
        features=test_data_2d, 
        feature_names=feature_names, 
        show=False
    )
    plt.title("SHAP Summary: Drivers of Predictive Log-Variance (Uncertainty)", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_variance_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Flatten global feature importance array to 1D: (Features,)
    global_importance = np.mean(np.abs(shap_values_2d), axis=0).ravel()
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': global_importance
    }).sort_values(by='Mean_Abs_SHAP', ascending=False)
    
    logger.info("SHAP Global Feature Importance (Variance Head):")
    for _, row in importance_df.iterrows():
        logger.info(f"  {row['Feature']:>20}: {row['Mean_Abs_SHAP']:.4f}")
        
    importance_df.to_csv(os.path.join(save_dir, "shap_variance_importance.csv"), index=False)
    return importance_df

# =====================================================================
# 3. PARTIAL DEPENDENCE PLOTS (PDP)
# =====================================================================

def generate_variance_pdp(
    model: nn.Module, 
    test_data: torch.Tensor, 
    feature_idx: int, 
    feature_name: str, 
    num_points: int = 20,
    save_dir: str = "results/explainability"
):
    """
    Generates a 1D Partial Dependence Plot for the variance head.
    Simulates how increasing a specific risk feature (e.g., GVZ) across all timesteps 
    forces the network to exponentially expand its predicted variance.
    """
    logger.info(f"Generating PDP for feature: {feature_name} (Index: {feature_idx})...")
    os.makedirs(save_dir, exist_ok=True)
    
    wrapper = VarianceHeadWrapper(model)
    device = test_data.device
    
    # Determine the empirical range of the feature in the test set
    feat_min = torch.min(test_data[:, :, feature_idx]).item()
    feat_max = torch.max(test_data[:, :, feature_idx]).item()
    
    grid_values = np.linspace(feat_min, feat_max, num_points)
    pdp_results = []
    
    with torch.no_grad():
        for val in grid_values:
            # Clone test data to avoid modifying the original tensor
            perturbed_data = test_data.clone()
            # Intervene: Set the feature to 'val' across all timesteps for all batches
            perturbed_data[:, :, feature_idx] = val
            
            # Forward pass through the Variance wrapper
            logvar_preds = wrapper(perturbed_data)
            mean_logvar = torch.mean(logvar_preds).item()
            pdp_results.append(mean_logvar)
            
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(grid_values, pdp_results, marker='o', linewidth=2, color='#d62728')
    plt.title(f"Partial Dependence Plot (Variance Head)\nEffect of {feature_name} on Log-Variance")
    plt.xlabel(f"{feature_name} (Standardized Value)")
    plt.ylabel("Average Predicted Log-Variance")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path = os.path.join(save_dir, f"pdp_variance_{feature_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"PDP saved to {save_path}")
    return grid_values, pdp_results

# =====================================================================
# 4. PERMUTATION IMPORTANCE (VARIANCE CHANNEL)
# =====================================================================

def compute_variance_permutation_importance(
    model: nn.Module, 
    test_dataloader: torch.utils.data.DataLoader, 
    feature_names: List[str],
    device: torch.device,
    n_repeats: int = 5
) -> pd.DataFrame:
    """
    Computes permutation importance strictly evaluating how shuffling a feature 
    disrupts the model's ability to predict high/low variance accurately.
    Uses Mean Predicted Variance Absolute Error as a proxy.
    """
    logger.info("Computing Variance-Head Permutation Importance...")
    wrapper = VarianceHeadWrapper(model).to(device)
    
    # 1. Compute baseline average log-variance
    baseline_logvars = []
    all_X = []
    with torch.no_grad():
        for X_batch, _ in test_dataloader:
            X_batch = X_batch.to(device)
            all_X.append(X_batch)
            logvars = wrapper(X_batch)
            baseline_logvars.append(logvars.cpu())
            
    baseline_logvars = torch.cat(baseline_logvars, dim=0)
    baseline_mean_logvar = torch.mean(baseline_logvars).item()
    
    # 2. Iterate and permute each feature
    X_full = torch.cat(all_X, dim=0) # Shape: (N, Timesteps, Features)
    importance_scores = []
    
    for feat_idx, feat_name in enumerate(feature_names):
        feature_deltas = []
        for _ in range(n_repeats):
            X_permuted = X_full.clone()
            
            # Shuffle the specific feature across the batch dimension
            # We shuffle the temporal blocks for each batch independently
            perm_idx = torch.randperm(X_full.size(0))
            X_permuted[:, :, feat_idx] = X_full[perm_idx, :, feat_idx]
            
            with torch.no_grad():
                permuted_logvars = wrapper(X_permuted)
                
            # Importance is measured by the absolute shift in average variance prediction.
            # If a risk anchor is destroyed, the network usually collapses its variance estimation.
            delta = torch.abs(torch.mean(permuted_logvars) - baseline_mean_logvar).item()
            feature_deltas.append(delta)
            
        mean_delta = np.mean(feature_deltas)
        importance_scores.append(mean_delta)
        logger.info(f"  {feat_name:>20}: {mean_delta:.4f} (Shift in Logvar)")
        
    # Compile Results
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Logvar_Shift_Importance': importance_scores
    }).sort_values(by='Logvar_Shift_Importance', ascending=False)
    
    return importance_df