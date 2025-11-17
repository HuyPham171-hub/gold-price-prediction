# GoldSight: Gold Price Forecasting System

## Project Overview

GoldSight is a comprehensive machine learning system designed to forecast gold prices using historical market data and macroeconomic indicators. The project combines advanced deep learning architectures with traditional econometric analysis to provide accurate short-term and medium-term gold price predictions.

## Objectives

- Develop a robust forecasting model for gold spot prices using multivariate time series analysis
- Identify and quantify the impact of key economic drivers on gold price movements
- Provide data-driven insights for investment decision-making in precious metals markets
- Create an accessible web interface for real-time price forecasting and historical analysis

## Methodology

### Data Collection
The system integrates multiple data sources spanning 2006-2025:
- Gold spot prices across major currencies
- Market indicators: S&P 500, USD Index, Silver Futures, Crude Oil
- Macroeconomic variables: CPI, Federal Funds Rate, Treasury Yields, Unemployment
- Risk metrics: VIX, Geopolitical Risk Index (GPR)

### Modeling Approach
Three modeling strategies were evaluated:
1. **Univariate Linear Regression**: Individual feature analysis with 13 separate models
2. **Multivariate Linear Regression**: Combined feature modeling with regularization techniques
3. **Deep Learning**: LSTM, GRU, and RNN architectures for temporal pattern recognition

### Key Findings
- GRU Multivariate model achieves optimal performance (R² = 0.990, MAE = $34.94)
- Macroeconomic indicators demonstrate stronger predictive power than market-based features
- Multicollinearity is present but manageable through Ridge regression techniques
- Temporal modeling captures long-term trends and seasonal patterns effectively

## Technical Implementation

### Core Technologies
- **Machine Learning**: TensorFlow/Keras for deep learning, scikit-learn for baseline models
- **Data Processing**: pandas, NumPy for data manipulation and feature engineering
- **Visualization**: Plotly, Matplotlib, Seaborn for exploratory analysis
- **Web Application**: Reflex framework for interactive dashboard deployment

### Model Performance
| Model Type | R² Score | MAE (USD) | Use Case |
|------------|----------|-----------|----------|
| GRU Multivariate | 0.990 | $34.94 | Primary forecasting model |
| LSTM Multivariate | 0.989 | $36.20 | Alternative temporal model |
| Linear Regression | 0.947 | $78.89 | Baseline comparison |

## Deliverables

1. **Exploratory Data Analysis**: Comprehensive statistical analysis and feature correlation studies
2. **Forecasting Models**: Pre-trained deep learning models with documented hyperparameters
3. **Web Application**: Interactive dashboard for price forecasting and historical trend visualization
4. **Documentation**: Technical report detailing methodology, results, and model evaluation metrics

## Business Value

GoldSight enables stakeholders to:
- Forecast gold prices with high accuracy (mean error < $35)
- Understand macroeconomic influences on precious metal valuations
- Assess risk exposure through volatility and geopolitical risk indicators
- Make data-informed trading and hedging decisions

## Future Enhancements

- Real-time data integration via financial APIs
- Extended forecast horizons (30, 60, 90-day predictions)
- Multi-asset portfolio optimization incorporating gold positions
- Automated retraining pipeline for model drift mitigation
