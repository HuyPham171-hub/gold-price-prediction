"""
src/evaluation/metrics.py

Provides rigorous econometric and UQ metrics for probabilistic forecasting.
Includes Clark-West tests, PP/KPSS unit root testing, empirical confidence intervals 
for coverage (Wilson Score), and Tail-Weighted interval scores.
"""

import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron
from typing import Dict, Tuple, Optional

# =====================================================================
# 1. ECONOMETRIC FORECAST & STATIONARITY TESTS
# =====================================================================

def clark_west_test(y_true: np.ndarray, y_pred_baseline: np.ndarray, y_pred_nested: np.ndarray) -> Tuple[float, float]:
    """
    Clark-West Test for Nested Models (e.g., Deep Learning vs Naive Random Walk).
    H0: The nested model (baseline) has equal MSE to the larger model.
    H1: The larger model has lower MSE.
    Returns: (CW Statistic, one-sided p-value).
    """
    e1 = y_true - y_pred_baseline
    e2 = y_true - y_pred_nested
    
    # Clark-West adjustment term
    adj = (y_pred_baseline - y_pred_nested) ** 2
    f_t = e1**2 - (e2**2 - adj)
    
    mean_f = np.mean(f_t)
    var_f = np.var(f_t, ddof=1)
    n = len(f_t)
    
    if var_f == 0:
        return 0.0, 1.0
        
    cw_stat = mean_f / np.sqrt(var_f / n)
    p_value = 1.0 - stats.norm.cdf(cw_stat) # One-sided test
    
    return float(cw_stat), float(p_value)

def run_stationarity_tests(series: np.ndarray) -> Dict[str, float]:
    """Runs a trifecta of residual stationarity tests."""
    results = {}
    
    # 1. ADF Test (Null: Unit Root)
    adf_res = adfuller(series, autolag='AIC')
    results['ADF_pval'] = adf_res[1]
    
    # 2. Phillips-Perron (Null: Unit Root)
    pp_test = PhillipsPerron(series)
    results['PP_pval'] = pp_test.pvalue
    
    # 3. KPSS Test (Null: Stationary)
    # Ignore the warning about p-values outside table range
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p, _, _ = kpss(series, regression='c', nlags="auto")
        results['KPSS_pval'] = kpss_p
        
    return results

# =====================================================================
# 2. POINT & UNCERTAINTY METRICS
# =====================================================================

def wilson_score_interval(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes the Wilson Score empirical confidence interval for a proportion (PICP)."""
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n
    denominator = 1 + z**2 / n
    center = p_hat + z**2 / (2 * n)
    spread = z * np.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))
    lower = (center - spread) / denominator
    upper = (center + spread) / denominator
    return lower, upper

def calculate_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true_flat, y_pred_flat = y_true.ravel(), y_pred.ravel()
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    sign_true, sign_pred = np.sign(y_true_flat), np.sign(y_pred_flat)
    valid_mask = sign_true != 0
    da = np.mean(sign_true[valid_mask] == sign_pred[valid_mask]) if np.sum(valid_mask) > 0 else 0.0

    return {"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2), "DA": float(da)}

def calculate_uncertainty_metrics(
    y_true: np.ndarray, 
    lower_bound: np.ndarray, 
    upper_bound: np.ndarray,
    variance_pred: Optional[np.ndarray] = None,
    alpha: float = 0.10
) -> Dict[str, float]:
    """
    Comprehensive UQ metrics including Tail-Weighted Winkler and QLIKE loss.
    """
    y, l, u = y_true.ravel(), lower_bound.ravel(), upper_bound.ravel()
    n = len(y)

    # 1. PICP & Confidence Interval
    covered = (y >= l) & (y <= u)
    successes = int(np.sum(covered))
    picp = successes / n if n > 0 else 0.0
    picp_l, picp_u = wilson_score_interval(successes, n, confidence=0.95)

    # 2. MPIW (Sharpness)
    widths = u - l
    mpiw = float(np.mean(widths))

    # 3. Tail-Weighted Winkler Score
    # Penalizes non-coverage heavily, separated into lower/upper tail penalties
    penalty_lower = (2.0 / alpha) * (l - y) * (y < l).astype(float)
    penalty_upper = (2.0 / alpha) * (y - u) * (y > u).astype(float)
    winkler_scores = widths + penalty_lower + penalty_upper
    mean_winkler = float(np.mean(winkler_scores))
    
    metrics = {
        "PICP": float(picp),
        "PICP_95CI_Lower": float(picp_l),
        "PICP_95CI_Upper": float(picp_u),
        "MPIW": float(mpiw),
        "Winkler": float(mean_winkler)
    }

    # 4. Volatility Forecast Evaluation (if variance predictions are provided)
    if variance_pred is not None:
        var_pred = variance_pred.ravel()
        realized_var = y ** 2
        eps = 1e-8
        rv_safe = np.maximum(realized_var, eps)
        var_safe = np.maximum(var_pred, eps)
        
        metrics["Vol_RMSE"] = float(np.sqrt(np.mean((realized_var - var_safe) ** 2)))
        metrics["QLIKE"] = float(np.mean((rv_safe / var_safe) - np.log(rv_safe / var_safe) - 1))
        
    return metrics

def evaluate_all_regimes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    normal_mask: np.ndarray,
    crisis_mask: np.ndarray,
    variance_pred: Optional[np.ndarray] = None
) -> Dict[str, Dict[str, float]]:
    
    results = {}
    
    # 1. Overall
    results["Overall_Point"] = calculate_point_metrics(y_true, y_pred)
    results["Overall_Uncertainty"] = calculate_uncertainty_metrics(y_true, lower_bound, upper_bound, variance_pred)
    
    # 2. Normal Regime
    results["Normal_Point"] = calculate_point_metrics(y_true[normal_mask], y_pred[normal_mask])
    var_norm = variance_pred[normal_mask] if variance_pred is not None else None
    results["Normal_Uncertainty"] = calculate_uncertainty_metrics(
        y_true[normal_mask], lower_bound[normal_mask], upper_bound[normal_mask], var_norm
    )
    
    # 3. Crisis Regime
    if np.sum(crisis_mask) > 0:
        results["Crisis_Point"] = calculate_point_metrics(y_true[crisis_mask], y_pred[crisis_mask])
        var_crisis = variance_pred[crisis_mask] if variance_pred is not None else None
        results["Crisis_Uncertainty"] = calculate_uncertainty_metrics(
            y_true[crisis_mask], lower_bound[crisis_mask], upper_bound[crisis_mask], var_crisis
        )
    else:
        results["Crisis_Point"] = {}
        results["Crisis_Uncertainty"] = {}
        
    return results