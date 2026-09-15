"""
src/training/trainer.py

Standardized training engine for Dual-Head Heteroskedastic models.
Incorporates Heuristic Variance Initialization to prevent collapse, 
and Calibration-Aware Early Stopping (monitoring NLL, PICP, and Winkler Score) 
to ensure robust out-of-sample uncertainty quantification.
"""

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class CalibrationEarlyStopping:
    """
    Stops training if the composite validation score (NLL + Winkler Penalty) 
    does not improve after a given patience. Saves the best model weights.
    """
    def __init__(self, patience: int = 15, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = np.inf
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_composite_score: float, model: nn.Module):
        if val_composite_score < self.best_score - self.min_delta:
            self.best_score = val_composite_score
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class ModelTrainer:
    """
    Manages the training and validation loop for Dual-Head PyTorch models.
    """
    def __init__(
        self, 
        model: nn.Module, 
        dataloaders: Dict[str, torch.utils.data.DataLoader], 
        device: torch.device,
        save_dir: str = "models_saved"
    ):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(self.save_dir, exist_ok=True)

    def _heuristic_variance_initialization(self):
        """
        Forces the initial predicted variance to match the empirical variance 
        of the training target to prevent variance collapse/explosion in early epochs.
        """
        # 1. Calculate empirical variance of the Training set
        all_y = []
        for _, y_batch in self.dataloaders['train']:
            all_y.append(y_batch)
        all_y = torch.cat(all_y)
        
        # Empirical variance (Var(y))
        emp_var = torch.var(all_y).item()
        target_logvar = np.log(emp_var + 1e-6)
        
        # 2. Inject into the logvar head
        initialized = False
        for name, module in self.model.named_modules():
            if 'fc_logvar' in name and isinstance(module, nn.Linear):
                # Set bias to target logvar, initialize weights to small random values (or zero)
                nn.init.constant_(module.bias, target_logvar)
                nn.init.zeros_(module.weight) 
                logger.info(f"Heuristic Init applied to '{name}': bias set to {target_logvar:.4f} (Empirical Var: {emp_var:.4f})")
                initialized = True
                
        if not initialized:
            logger.warning("Could not find 'fc_logvar' layer. Heuristic initialization skipped.")

    def train(
        self, 
        model_name: str, 
        num_epochs: int = 100, 
        learning_rate: float = 1e-3, 
        patience: int = 15,
        winkler_weight: float = 0.05 # Trade-off hyperparameter for Early Stopping
    ) -> Dict[str, Any]:
        
        # 1. Apply Heuristic Initialization
        self._heuristic_variance_initialization()
        
        # --- GAUSSIAN NLL LOSS ---
        criterion = nn.GaussianNLLLoss(eps=1e-6)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        early_stopping = CalibrationEarlyStopping(patience=patience)
        history = {
            'train_loss': [], 
            'val_loss': [], 
            'val_picp': [], 
            'val_winkler': [],
            'val_composite_score': []
        }

        logger.info(f"Starting Calibration-Aware Training for {model_name} on {self.device}...")

        for epoch in range(num_epochs):
            # ==========================================
            # TRAINING PHASE
            # ==========================================
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in self.dataloaders['train']:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                # Reshape y_batch for loss computation if necessary
                if y_batch.dim() == 1:
                    y_batch = y_batch.unsqueeze(1)
                
                optimizer.zero_grad()
                
                # Dual-head models output BOTH mean (mu) and variance (sigma_sq)
                mu, sigma_sq = self.model(X_batch)
                
                loss = criterion(mu, y_batch, sigma_sq)
                
                loss.backward()
                # Gradient clipping to prevent exploding gradients from variance head
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
                
            train_loss /= len(self.dataloaders['train'].dataset)
            history['train_loss'].append(train_loss)

            # ==========================================
            # VALIDATION PHASE (NLL + Calibration)
            # ==========================================
            self.model.eval()
            val_loss = 0.0
            val_picp = 0.0
            val_winkler = 0.0
            
            # 90% Confidence Interval Multiplier for Gaussian
            z_score = 1.645 
            alpha = 0.10 # Target error rate for Winkler Score
            
            with torch.no_grad():
                for X_batch, y_batch in self.dataloaders['val']:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    
                    if y_batch.dim() == 1:
                        y_batch = y_batch.unsqueeze(1)
                    
                    mu, sigma_sq = self.model(X_batch)
                    loss = criterion(mu, y_batch, sigma_sq)
                    val_loss += loss.item() * X_batch.size(0)
                    
                    # --- Calibration Metrics (Interval Score & Coverage) ---
                    sigma = torch.sqrt(sigma_sq)
                    L = mu - z_score * sigma
                    U = mu + z_score * sigma
                    
                    # 1. PICP (Prediction Interval Coverage Probability)
                    covered = ((y_batch >= L) & (y_batch <= U)).float()
                    val_picp += covered.sum().item()
                    
                    # 2. Winkler Score (Penalizes width and non-coverage)
                    delta = U - L
                    penalty_lower = (2.0 / alpha) * (L - y_batch) * (y_batch < L).float()
                    penalty_upper = (2.0 / alpha) * (y_batch - U) * (y_batch > U).float()
                    winkler = delta + penalty_lower + penalty_upper
                    val_winkler += winkler.sum().item()
                    
            n_val = len(self.dataloaders['val'].dataset)
            val_loss /= n_val
            val_picp /= n_val
            val_winkler /= n_val
            
            # Composite Score for Early Stopping (Balances pure likelihood with practical calibration)
            val_composite_score = val_loss + (winkler_weight * val_winkler)
            
            history['val_loss'].append(val_loss)
            history['val_picp'].append(val_picp)
            history['val_winkler'].append(val_winkler)
            history['val_composite_score'].append(val_composite_score)

            scheduler.step(val_composite_score)
            early_stopping(val_composite_score, self.model)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1:03d}/{num_epochs} | "
                    f"Train NLL: {train_loss:.4f} | "
                    f"Val NLL: {val_loss:.4f} | "
                    f"Val PICP: {val_picp*100:.1f}% | "
                    f"Val Winkler: {val_winkler:.4f}"
                )

            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch+1}. Restoring best weights.")
                break

        logger.info("Saving best calibrated model weights...")
        self.model.load_state_dict(early_stopping.best_model_wts)
        
        save_path = os.path.join(self.save_dir, f"{model_name}_best.pt")
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model successfully saved to {save_path}")

        return history