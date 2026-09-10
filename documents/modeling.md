# Chapter 3: Modeling & Evaluation - From Statistics to Deep Learning

## Overview
In this chapter, we systematically test 11 different models across 3 algorithm families to find the optimal approach for gold price prediction. Rather than jumping to complex solutions, we start with simple baselines and progressively add complexity, understanding what each model contributes.

### What We'll Explore
- **Simple vs Multiple:** Does each feature work alone, or do they need to work together?
- **Linear vs Non-linear:** Are relationships straight lines, or do we need curves and interactions?
- **Time Series vs Feature-based:** Should we focus on temporal patterns or cross-variable relationships?
- **Simple vs Complex:** When does added complexity actually improve predictions?

*The journey ahead: We'll start with simple linear regression to establish a baseline, then explore traditional machine learning methods to handle non-linearity, and finally test deep learning architectures to capture temporal dependencies. Each step reveals what matters most for gold price prediction.*

## Modeling Philosophy: Start Simple, Add Complexity
Rather than jumping straight to complex solutions, we follow a **systematic approach**: start with **simple baselines**, understand their limitations, then **progressively add complexity**. This ensures we understand what each model contributes and **avoid unnecessary sophistication**.

1. **Baseline (Linear Models & Time Series):** Establish minimum acceptable performance. If simple works, why complicate?
2. **Traditional ML (Non-linear & Ensemble Methods):** Handle non-linearity and feature interactions. Test if advanced methods outperform linear baselines.
3. **Deep Learning (Neural Networks & Sequence Models):** Capture temporal dependencies and complex patterns. Maximum predictive power.

### Evaluation Criteria
- **R² (Coefficient of Determination):** How much of the gold price's movement can our model explain? (Scale: 0 to 1. Higher is better).
- **RMSE (Root Mean Squared Error):** Average prediction error in dollars. Lower is better.
- **MAE (Mean Absolute Error):** What is our average error, in dollars? (e.g., 'On average, the model is off by $35').

## Simple Linear Regression: Testing Each Feature
Before building Multiple models, we tested each of the 13 features individually to understand their standalone predictive power. This reveals which features have strong linear relationships with gold prices.

### Summary Table
| Feature | R² | RMSE | MAE | Interpretation |
|---------|----|------|-----|----------------|
| **CPI** | 0.720 | $266.74 | $210.61 | Strongest single predictor |
| **S&P_500** | 0.619 | $311.12 | $240.87 | Stock market correlation |
| **Silver_Futures** | 0.526 | $346.97 | $274.55 | Precious metal co-movement |
| USD_Index | 0.361 | $402.80 | $326.55 | Currency strength impact |
| GPR | 0.193 | $452.57 | $368.78 | Geopolitical risk factor |
| GPRA | 0.083 | $482.24 | $382.14 | Action-based risk |
| Real_Interest_Rate | 0.079 | $483.29 | $352.45 | Moderate predictive power |
| Treasury_Yield_10Y | 0.053 | $490.13 | $374.42 | Weak linear relationship |
| VIX | -0.020 | $508.61 | $403.34 | Near-zero linear fit |
| Unemployment | -0.002 | $504.21 | $400.82 | Near-zero linear fit |
| Crude_Oil | 0.001 | $503.37 | $391.79 | Near-zero linear fit |
| Fed_Funds_Rate | -0.043 | $514.37 | $400.19 | Weak negative fit |

### Top 3 Features
1. **CPI (Inflation) (R² = 0.720, RMSE: $266.74, MAE: $210.61):** Consumer Price Index explains 72% of gold price variance. When inflation rises, gold prices follow as investors seek inflation hedge. This is the single most predictive feature. (Formula: Gold = 13.41 x CPI - 1876.60)
2. **S&P 500 (R² = 0.619, RMSE: $311.12, MAE: $240.87):** Stock market index explains 62% of variance. Surprising positive correlation: both rise in liquidity-driven markets. Challenges 'gold vs stocks' narrative. (Formula: Gold = 0.30 x S&P500 + 686.66)
3. **Silver Futures (R² = 0.526, RMSE: $346.97, MAE: $274.55):** Precious metals move together. Silver and gold share similar drivers (inflation hedge, safe haven). 53% of gold variance explained by silver alone. (Formula: Gold = 50.17 x Silver + 382.50)

### Classification by Predictive Power
Not all features work well alone. Low R² doesn't mean irrelevant — it means the relationship is non-linear, lagged, or requires interaction with other variables.
- **Weak/Insignificant (R² < 0.08):** VIX (-0.020), Crude Oil (0.001), Unemployment (-0.002), Fed Funds (-0.043), Treasury Yield (0.053), Real Interest (0.079).
- **Moderate Predictors (R² = 0.08–0.36):** USD Index (0.361), GPR (0.193), GPRA (0.083).

**Solution: Multiple Models**
These 'weak' features become valuable in Multiple models through interactions. Example: Real Interest Rate (R² = 0.079 alone) + CPI + Fed Funds jointly capture the real cost of holding gold vs interest-bearing assets.

## Multiple Linear Regression: Combining All Features
Now we use all 13 features simultaneously. This allows the model to capture interactions between variables (e.g., inflation and interest rates together affecting gold). Results: **R² = 0.947**, **RMSE = $115.88**, **MAE = $77.06**.

### Coefficients & Significance
Features with p < 0.05 are statistically significant.
| Feature | Coefficient (β) | p-value | 95% CI | Significance |
|---------|-----------------|---------|--------|--------------|
| **Intercept** | -1009.58 | 0.000 | [-1554, -465] | Highly significant |
| **Silver_Futures** | +25.49 | 0.000 | [21.35, 29.64] | Very strong |
| **Unemployment** | +32.04 | 0.000 | [19.63, 44.45] | Positive |
| **CPI** | +10.20 | 0.000 | [7.46, 12.95] | Inflation hedge |
| **S&P_500** | +0.103 | 0.000 | [0.050, 0.156] | Market linkage |
| **USD_Index** | -7.84 | 0.010 | [-13.76, -1.91] | Currency inverse |
| **Crude_Oil** | -2.20 | 0.022 | [-4.08, -0.33] | Negative (multicollinearity) |
| VIX | +1.51 | 0.239 | [-1.01, 4.02] | Not significant |
| Treasury_Yield_10Y | -52.38 | 0.171 | [-127.63, 22.87] | Not significant |
| Real_Interest_Rate | +24.05 | 0.525 | [-50.55, 98.66] | Not significant |
| Fed_Funds_Rate | +5.12 | 0.609 | [-14.62, 24.87] | Not significant |
| GPR | +0.22 | 0.682 | [-0.86, 1.31] | Not significant |
| GPRA | +0.08 | 0.867 | [-0.84, 0.99] | Not significant |

**Key Findings:**
- **6 significant features (p < 0.05):** Silver, Unemployment, CPI, S&P500, USD Index, Crude Oil
- **Unemployment coefficient (+32.04):** When unemployment increase -> gold increase (safe haven during economic stress)
- **Crude Oil negative (-2.20):** Counterintuitive, likely due to multicollinearity with CPI
- **7 non-significant features:** VIX, interest rates, geopolitical indices (redundant in Multiple context)

### Model Diagnostics & Assumptions
- **Overall Model Fit:** F-statistic: 312.9. Prob (F) = 1.89e-110 ~ 0.000. Model is highly significant.
- **Durbin-Watson:** DW statistic: 2.221. Target: near 2.0. No autocorrelation.
- **Omnibus Test:** Prob: 0.000. Skew: 0.928 | Kurtosis: 10.38. Residuals not normal.