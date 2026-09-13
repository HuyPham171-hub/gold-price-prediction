"""
src/evaluation/mc_engine.py

Monte Carlo Dropout Inference Engine for Dual-Head Architectures.
Combines Epistemic Uncertainty (model weight dispersion) with dynamically learned 
Aleatoric Uncertainty (predicted data variance) using the Law of Total Variance.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def activate_mc_dropout(model: nn.Module) -> None:
    """
    Forces all Dropout layers to remain active during evaluation for Epistemic uncertainty.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            module.train()

def run_stochastic_inference(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    device: torch.device, 
    T: int = 200
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs T forward passes. Calculates Total Variance by fusing Epistemic and Aleatoric components.
    
    Returns:
        y_mean: Deterministic point forecast.
        lower_bound: 5th percentile of the combined distribution (90% CI).
        upper_bound: 95th percentile of the combined distribution (90% CI).
    """
    logger.info(f"Starting Dual-Head MC Dropout inference with T={T} passes...")
    
    activate_mc_dropout(model)
    model.to(device)
    
    all_mu = []
    all_sigma_sq = []
    
    with torch.no_grad():
        for t in range(T):
            batch_mu = []
            batch_sigma_sq = []
            
            for X_batch, _ in dataloader:
                X_batch = X_batch.to(device)
                
                # Model now returns mean (mu) and variance (sigma_sq)
                mu, sigma_sq = model(X_batch)
                
                batch_mu.append(mu.cpu().numpy())
                batch_sigma_sq.append(sigma_sq.cpu().numpy())
            
            all_mu.append(np.concatenate(batch_mu, axis=0))
            all_sigma_sq.append(np.concatenate(batch_sigma_sq, axis=0))
            
            if (t + 1) % 50 == 0:
                logger.info(f"Completed {t + 1}/{T} stochastic passes.")

    # Stack matrices into shape (T, N_samples)
    mu_matrix = np.array(all_mu).squeeze(-1)
    sigma_sq_matrix = np.array(all_sigma_sq).squeeze(-1)
    
    # 1. Expected Point Forecast (Mean of all predicted means)
    y_mean = np.mean(mu_matrix, axis=0)
    
    # 2. Epistemic Variance (Dispersion of the predicted means over T passes)
    epistemic_var = np.var(mu_matrix, axis=0)
    
    # 3. Aleatoric Variance (Average of the predicted data variances over T passes)
    aleatoric_var = np.mean(sigma_sq_matrix, axis=0)
    
    # 4. Law of Total Variance
    total_var = epistemic_var + aleatoric_var
    total_std = np.sqrt(total_var)
    
    # 5. Gaussian Bounds for 90% Confidence Interval (Z-score = 1.645)
    lower_bound = y_mean - 1.645 * total_std
    upper_bound = y_mean + 1.645 * total_std
    
    logger.info("Inference complete. Heteroskedastic bounds generated.")
    
    return y_mean, lower_bound, upper_bound

def extract_true_labels(dataloader: torch.utils.data.DataLoader) -> np.ndarray:
    all_labels = []
    for _, y_batch in dataloader:
        all_labels.append(y_batch.numpy())
    return np.concatenate(all_labels, axis=0).squeeze(-1)