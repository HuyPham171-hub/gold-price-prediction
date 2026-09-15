"""
src/training/loss.py

Advanced loss functions for Heteroskedastic and Quantile Regression modeling.
Includes Gaussian NLL (optimized for logvar), Student-t NLL (with learnable degrees of freedom), 
and Pinball Loss for distribution-free interval bounds.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianNLLLogvar(nn.Module):
    """
    Gaussian Negative Log-Likelihood Loss.
    Optimized to directly accept 'logvar' (log-variance) to prevent numerical instability 
    (variance collapse or explosion).
    
    Formula: L = 0.5 * (log(2*pi) + logvar + (y - mu)^2 / exp(logvar))
    """
    def __init__(self, eps: float = 1e-6):
        super(GaussianNLLLogvar, self).__init__()
        self.eps = eps
        # Constant log(2 * pi)
        self.c = math.log(2 * math.pi)

    def forward(self, mu: torch.Tensor, y: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Ensure y has the same shape as mu
        if y.dim() == 1:
            y = y.unsqueeze(1)
            
        variance = torch.exp(logvar) + self.eps
        loss = 0.5 * (self.c + logvar + ((y - mu) ** 2) / variance)
        return torch.mean(loss)


class StudentTNLL(nn.Module):
    """
    Student-t Negative Log-Likelihood Loss.
    Models fat-tailed financial returns by learning the degrees of freedom (nu).
    
    The parameter 'nu' is constrained to be strictly > 2 (to ensure finite variance)
    using the transformation: nu = 2 + softplus(eta), where eta is a learnable parameter.
    """
    def __init__(self, eps: float = 1e-6):
        super(StudentTNLL, self).__init__()
        self.eps = eps
        # Learnable parameter eta. Initialized to 0.0 (nu ≈ 2.69)
        self.eta = nn.Parameter(torch.tensor(0.0))

    def forward(self, mu: torch.Tensor, y: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if y.dim() == 1:
            y = y.unsqueeze(1)
            
        # 1. Enforce constraint: nu > 2
        nu = 2.0 + F.softplus(self.eta)
        variance = torch.exp(logvar) + self.eps
        
        # 2. Compute NLL terms for Student-t distribution
        # Log Gamma terms
        term1 = -torch.lgamma((nu + 1) / 2.0)
        term2 = torch.lgamma(nu / 2.0)
        
        # Log scale terms
        term3 = 0.5 * torch.log(math.pi * nu)
        term4 = 0.5 * logvar
        
        # Polynomial tail term
        term5 = ((nu + 1) / 2.0) * torch.log(1.0 + ((y - mu) ** 2) / (nu * variance))
        
        loss = term1 + term2 + term3 + term4 + term5
        return torch.mean(loss)


class PinballLoss(nn.Module):
    """
    Pinball Loss (Quantile Loss) for distribution-free tail bounds.
    Used for Quantile Regression models to directly predict specific percentiles 
    (e.g., 5th and 95th for a 90% confidence interval) without assuming a Gaussian mixture.
    
    Formula for quantile q: L_q = max(q * (y - y_hat), (q - 1) * (y - y_hat))
    """
    def __init__(self, quantiles: list[float] = [0.05, 0.50, 0.95]):
        super(PinballLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, predictions: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Tensor of shape (Batch, Num_Quantiles)
            y: Ground truth tensor of shape (Batch, 1) or (Batch,)
        """
        if y.dim() == 1:
            y = y.unsqueeze(1)
            
        if predictions.size(1) != len(self.quantiles):
            raise ValueError(f"Expected {len(self.quantiles)} predictions per sample, got {predictions.size(1)}")

        losses = []
        for i, q in enumerate(self.quantiles):
            # Extract the prediction for the i-th quantile
            pred_q = predictions[:, i].unsqueeze(1)
            error = y - pred_q
            
            # Pinball loss calculation for this specific quantile
            loss_q = torch.max(q * error, (q - 1.0) * error)
            losses.append(loss_q)
            
        # Average loss across all quantiles and the batch
        total_loss = torch.mean(torch.cat(losses, dim=1))
        return total_loss