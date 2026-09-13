"""
src/evaluation/metrics.py

Provides functions to calculate point-forecast metrics, uncertainty metrics,
and to generate boolean masks for regime stratification (Normal vs. Crisis).
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple

def calculate_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates standard deterministic metrics including Directional Accuracy (DA).
    """
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()

    rmse = float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)))
    mae = float(mean_absolute_error(y_true_flat, y_pred_flat))
    r2 = float(r2_score(y_true_flat, y_pred_flat))
    
    # Directional Accuracy: checks if the predicted sign matches the actual sign
    sign_true = np.sign(y_true_flat)
    sign_pred = np.sign(y_pred_flat)
    # Ignore days where true return is exactly 0 to avoid false penalties
    valid_mask = sign_true != 0
    da = float(np.mean(sign_true[valid_mask] == sign_pred[valid_mask]))

    return {
        "RMSE": round(rmse, 6),
        "MAE": round(mae, 6),
        "R2": round(r2, 6),
        "DA": round(da, 4)
    }

def calculate_uncertainty_metrics(
    y_true: np.ndarray, 
    lower_bound: np.ndarray, 
    upper_bound: np.ndarray,
    alpha: float = 0.10
) -> Dict[str, float]:
    """
    Calculates metrics for probabilistic forecasting (Prediction Intervals).
    Default alpha=0.10 corresponds to a 90% confidence interval.
    """
    y_true_flat = y_true.ravel()
    lower_flat = lower_bound.ravel()
    upper_flat = upper_bound.ravel()

    # 1. PICP: Prediction Interval Coverage Probability (Target: ~90%)
    covered = (y_true_flat >= lower_flat) & (y_true_flat <= upper_flat)
    picp = float(np.mean(covered))

    # 2. MPIW: Mean Prediction Interval Width (Measures sharpness)
    widths = upper_flat - lower_flat
    mpiw = float(np.mean(widths))

    # 3. Winkler Score: Penalizes intervals that fail to cover the true value
    winkler_scores = np.where(
        covered,
        widths,
        np.where(
            y_true_flat < lower_flat,
            widths + (2.0 / alpha) * (lower_flat - y_true_flat),
            widths + (2.0 / alpha) * (y_true_flat - upper_flat)
        )
    )
    mean_winkler = float(np.mean(winkler_scores))

    return {
        "PICP": round(picp, 4),
        "MPIW": round(mpiw, 4),
        "Winkler": round(mean_winkler, 4)
    }

def get_regime_masks(
    risk_feature_array: np.ndarray, 
    threshold_percentile: float = 90.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits the evaluation timeline into Normal and Crisis regimes based on a risk feature.
    
    Args:
        risk_feature_array: 1D array of a risk metric (e.g., GPR or Gold_VIX values) for the Test set.
        threshold_percentile: The percentile above which a day is considered a Crisis.
        
    Returns:
        Tuple of boolean masks (normal_mask, crisis_mask).
    """
    risk_feature_flat = risk_feature_array.ravel()
    threshold_value = np.percentile(risk_feature_flat, threshold_percentile)
    
    crisis_mask = risk_feature_flat >= threshold_value
    normal_mask = ~crisis_mask
    
    return normal_mask, crisis_mask

def evaluate_all_regimes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    normal_mask: np.ndarray,
    crisis_mask: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Helper function to calculate all metrics across Overall, Normal, and Crisis periods.
    """
    results = {}
    
    # 1. Overall Metrics
    results["Overall_Point"] = calculate_point_metrics(y_true, y_pred)
    results["Overall_Uncertainty"] = calculate_uncertainty_metrics(y_true, lower_bound, upper_bound)
    
    # 2. Normal Regime Metrics
    results["Normal_Point"] = calculate_point_metrics(y_true[normal_mask], y_pred[normal_mask])
    results["Normal_Uncertainty"] = calculate_uncertainty_metrics(
        y_true[normal_mask], lower_bound[normal_mask], upper_bound[normal_mask]
    )
    
    # 3. Crisis Regime Metrics
    results["Crisis_Point"] = calculate_point_metrics(y_true[crisis_mask], y_pred[crisis_mask])
    results["Crisis_Uncertainty"] = calculate_uncertainty_metrics(
        y_true[crisis_mask], lower_bound[crisis_mask], upper_bound[crisis_mask]
    )
    
    return results