"""
src/evaluation/mc_engine.py

Monte Carlo Dropout Inference Engine for Dual-Head Architectures.
Combines Epistemic Uncertainty (model weight dispersion) with dynamically learned 
Aleatoric Uncertainty (predicted data variance). Generates empirical quantiles 
from the combined predictive mixture to capture non-Gaussian tail behavior.
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
    scaler_target=None, 
    T: int = 200,
    confidence_level: float = 0.90
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs T forward passes. Calculates empirical quantiles by sampling from the 
    predictive mixture of Gaussians.
    
    Returns:
        y_mean: Deterministic point forecast.
        lower_bound: Lower empirical quantile of the combined distribution.
        upper_bound: Upper empirical quantile of the combined distribution.
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

    # Stack matrices into shape (T, N_samples, 1) and squeeze
    mu_matrix = np.array(all_mu).squeeze(-1)       # Shape: (T, N_samples)
    sigma_sq_matrix = np.array(all_sigma_sq).squeeze(-1)
    
    # 1. Expected Point Forecast
    y_mean = np.mean(mu_matrix, axis=0)
    
    # 2. Empirical Sampling
    std_matrix = np.sqrt(sigma_sq_matrix)
    sampled_y = np.random.normal(loc=mu_matrix, scale=std_matrix)
    
    # 3. Extract Empirical Quantiles
    lower_q = ((1.0 - confidence_level) / 2.0) * 100
    upper_q = (1.0 - (1.0 - confidence_level) / 2.0) * 100
    lower_bound = np.percentile(sampled_y, lower_q, axis=0)
    upper_bound = np.percentile(sampled_y, upper_q, axis=0)
    
    # 4. Aleatoric Variance extraction for Volatility tracking
    aleatoric_var = np.mean(sigma_sq_matrix, axis=0)
    
    # =========================================================================
    # STRICT FIX: Variance & Point Scaling Inversion (RQ2 Requirement)
    # =========================================================================
    if scaler_target is not None:
        logger.info("Inverting predictions back to original financial scale...")
        # Get scaling standard deviation (s_y) and mean (m_y)
        s_y = scaler_target.scale_[0]
        m_y = scaler_target.mean_[0]
        
        # Invert Point Predictions & Bounds
        y_mean = (y_mean * s_y) + m_y
        lower_bound = (lower_bound * s_y) + m_y
        upper_bound = (upper_bound * s_y) + m_y
        
        # Invert Variance: sigma^2_original = (s_y^2) * sigma^2_scaled
        aleatoric_var = aleatoric_var * (s_y ** 2)
    # =========================================================================
    
    logger.info(f"Inference complete. Empirical {confidence_level*100}% bounds generated.")
    return y_mean, lower_bound, upper_bound, aleatoric_var

def extract_true_labels(dataloader: torch.utils.data.DataLoader) -> np.ndarray:
    all_labels = []
    for _, y_batch in dataloader:
        all_labels.append(y_batch.numpy())
    return np.concatenate(all_labels, axis=0).squeeze(-1)