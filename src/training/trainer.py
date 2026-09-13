"""
src/training/trainer.py

Standardized training engine for Dual-Head Heteroskedastic models.
Utilizes Gaussian Negative Log-Likelihood (NLL) Loss to simultaneously optimize 
the point prediction (mean) and dynamically quantify the aleatoric variance (sigma squared).
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

class EarlyStopping:
    """
    Stops training if validation loss does not improve after a given patience.
    Saves the best model weights.
    """
    def __init__(self, patience: int = 15, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
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

    def train(
        self, 
        model_name: str, 
        num_epochs: int = 100, 
        learning_rate: float = 1e-3, 
        patience: int = 15
    ) -> Dict[str, Any]:
        
        # --- GAUSSIAN NLL LOSS ---
        # The key to Heteroskedasticity. It forces the network to increase 'var'
        # when it suspects the error (input - target) will be large.
        criterion = nn.GaussianNLLLoss(eps=1e-6)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        early_stopping = EarlyStopping(patience=patience)
        history = {'train_loss': [], 'val_loss': []}

        logger.info(f"Starting NLL training for {model_name} on {self.device}...")

        for epoch in range(num_epochs):
            # --- Training Phase ---
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in self.dataloaders['train']:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                
                # Dual-head models output BOTH mean (mu) and variance (sigma_sq)
                mu, sigma_sq = self.model(X_batch)
                
                # criterion(input, target, var)
                loss = criterion(mu, y_batch, sigma_sq)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
                
            train_loss /= len(self.dataloaders['train'].dataset)
            history['train_loss'].append(train_loss)

            # --- Validation Phase ---
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for X_batch, y_batch in self.dataloaders['val']:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    
                    mu, sigma_sq = self.model(X_batch)
                    loss = criterion(mu, y_batch, sigma_sq)
                    val_loss += loss.item() * X_batch.size(0)
                    
            val_loss /= len(self.dataloaders['val'].dataset)
            history['val_loss'].append(val_loss)

            scheduler.step(val_loss)
            early_stopping(val_loss, self.model)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss (NLL): {train_loss:.4f} - Val Loss (NLL): {val_loss:.4f}")

            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                break

        logger.info("Loading best model weights...")
        self.model.load_state_dict(early_stopping.best_model_wts)
        
        save_path = os.path.join(self.save_dir, f"{model_name}_best.pt")
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")

        return history