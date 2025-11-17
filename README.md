# GoldSight: Gold Price Forecasting System

A comprehensive machine learning system for forecasting gold prices using historical market data and macroeconomic indicators. This project implements deep learning architectures and traditional econometric models to deliver accurate short-term and medium-term gold price predictions through an interactive web application.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Web Application](#web-application)
- [Technical Stack](#technical-stack)
- [License](#license)

## Overview

GoldSight combines multiple forecasting methodologies to predict gold spot prices with high accuracy (R² = 0.990, MAE = $34.94). The system analyzes 13 features including market indicators, macroeconomic variables, and risk metrics spanning 2006-2025.

### Key Objectives

- Develop robust forecasting models using multivariate time series analysis
- Quantify economic drivers influencing gold price movements
- Provide actionable insights for investment decision-making
- Deploy an accessible web interface for real-time forecasting

## Features

- **Multiple Modeling Strategies**: Univariate regression, multivariate regression, and deep learning (LSTM, GRU, RNN)
- **Comprehensive Data Analysis**: Exploratory data analysis with correlation studies and distribution analysis
- **Interactive Dashboard**: Web-based application built with Reflex framework
- **Real-time Forecasting**: Pre-trained models for immediate price predictions
- **Performance Metrics**: Detailed model evaluation with R², MAE, RMSE, and MAPE
- **Visualization Suite**: Static and interactive charts for data exploration and results presentation

## Project Structure

```
Project/
├── data/                          # Dataset directory
│   ├── raw/                       # Original data files
│   │   ├── gold_spot_WGC.csv
│   │   ├── market_data.csv
│   │   ├── macro_monthly.csv
│   │   ├── real_interest_rate.csv
│   │   ├── usd_index.csv
│   │   ├── vix_data.csv
│   │   └── data_gpr_export.xls
│   ├── processed/                 # Cleaned and transformed data
│   │   ├── market_data_ffill.csv
│   │   ├── RIR_ffill.csv
│   │   ├── usd_ffill.csv
│   │   └── vix_ffill.csv
│   ├── combined_data.csv          # Merged dataset
│   └── filtered_data.csv          # Final feature set
│
├── notebooks/                     # Jupyter notebooks
│   ├── collect_data.ipynb         # Data collection pipeline
│   ├── explore.ipynb              # Exploratory data analysis
│   ├── modeling_new.ipynb         # Model training and evaluation
│   └── plots_case1/               # Generated visualizations
│
├── models/                        # Trained model artifacts
│   ├── best_gru_multivariate.keras
│   ├── best_lstm_multivariate.keras
│   ├── best_rnn_multivariate.keras
│   ├── best_gru.keras
│   ├── best_lstm.keras
│   ├── best_rnn.keras
│   └── best_mlp.keras
│
├── goldsight/                     # Reflex web application
│   ├── goldsight.py               # Main application entry
│   ├── pages/                     # Application pages
│   │   ├── home.py
│   │   ├── data_collection.py
│   │   ├── eda.py
│   │   ├── modeling.py
│   │   └── forecast.py
│   ├── components/                # Reusable UI components
│   │   ├── navbar.py
│   │   ├── buttons.py
│   │   ├── card.py
│   │   ├── chart.py
│   │   ├── chapter_nav.py
│   │   └── layout.py
│   ├── services/                  # Business logic
│   │   ├── data_collector.py
│   │   ├── data_preprecessor.py
│   │   └── forecast_pipeline.py
│   ├── utils/                     # Utility functions
│   │   └── design_system.py
│   └── data/cache/                # Cached visualizations
│
├── results/                       # Analysis outputs
│   ├── explore/                   # EDA visualizations
│   └── LR_Univariate/             # Regression analysis plots
│
├── requirements.txt               # Python dependencies
├── requirements-dev.txt           # Development dependencies
├── rxconfig.py                    # Reflex configuration
├── README.md                      # Project documentation
└── PRODUCT_DESCRIPTION.md         # Product overview

```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- 4GB+ RAM (8GB recommended for model training)

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/HuyPham171-hub/gold-price-prediction.git
cd gold-price-prediction/Project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify installation:
```bash
python -c "import reflex; import tensorflow; import sklearn; print('Installation successful')"
```

## Usage

### Data Preparation

1. Run data collection notebook:
```bash
jupyter notebook notebooks/collect_data.ipynb
```

2. Execute exploratory data analysis:
```bash
jupyter notebook notebooks/explore.ipynb
```

### Model Training

Train models using the modeling notebook:
```bash
jupyter notebook notebooks/modeling_new.ipynb
```

Pre-trained models are available in the `models/` directory for immediate use.

### Web Application

Launch the Reflex web application:
```bash
reflex run
```

Access the application at `http://localhost:3000`

## Data Sources

### Market Indicators
- **S&P 500**: U.S. stock market performance
- **USD Index**: U.S. dollar strength
- **Silver Futures**: Precious metal correlation
- **Crude Oil**: Commodity market trends

### Macroeconomic Variables
- **CPI**: Consumer Price Index (inflation)
- **Federal Funds Rate**: Monetary policy indicator
- **Treasury Yield 10Y**: Long-term interest rates
- **Real Interest Rate**: Inflation-adjusted rates
- **Unemployment**: Labor market conditions

### Risk Metrics
- **VIX**: Market volatility index
- **GPR**: Geopolitical Risk Index
- **GPRA**: Geopolitical Risk Acts

### Target Variable
- **Gold Spot Price (USD)**: Daily gold prices from World Gold Council

**Data Range**: January 2006 - December 2024 (6,800+ observations)

## Modeling Approach

### 1. Univariate Linear Regression
- Individual feature analysis with 13 separate models
- Baseline performance evaluation
- Feature importance ranking

### 2. Multivariate Linear Regression
- Combined feature modeling with OLS and Ridge regularization
- Variance Inflation Factor (VIF) analysis for multicollinearity
- R² = 0.947, MAE = $78.89

### 3. Deep Learning Models

#### Architecture Comparison

| Model | Architecture | Parameters | R² Score | MAE (USD) |
|-------|--------------|------------|----------|-----------|
| GRU Multivariate | 128-64-32 units, 3 layers | 45,312 | 0.990 | $34.94 |
| LSTM Multivariate | 128-64-32 units, 3 layers | 60,416 | 0.989 | $36.20 |
| RNN Multivariate | 128-64-32 units, 3 layers | 30,208 | 0.987 | $38.75 |

#### Training Configuration
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Mean Squared Error
- **Regularization**: Dropout (0.2), Early Stopping (patience: 20)
- **Batch Size**: 32
- **Epochs**: 200 (with early stopping)
- **Validation Split**: 80/20 train-test

### Feature Engineering
- Forward-fill for missing values
- Daily resampling for monthly indicators
- Min-Max normalization (0-1 scale)
- Sequence length: 60 days for temporal models

## Results

### Model Performance Summary

**Best Model: GRU Multivariate**
- R² Score: 0.990
- Mean Absolute Error: $34.94
- Root Mean Squared Error: $47.23
- Mean Absolute Percentage Error: 2.1%

### Key Findings

1. **Macroeconomic Dominance**: CPI, interest rates, and unemployment demonstrate stronger predictive power than market indicators
2. **Temporal Patterns**: GRU architecture effectively captures long-term trends and seasonal variations
3. **Multicollinearity Management**: Ridge regression maintains performance despite high VIF values (CPI: 1805.21, USD Index: 1012.78)
4. **Feature Correlation**: Gold shows strong positive correlation with CPI (0.85) and negative correlation with real interest rates (-0.40)

### Model Comparison

| Category | Model | R² | MAE | RMSE | MAPE |
|----------|-------|-----|-----|------|------|
| Baseline | Linear Regression | 0.947 | $78.89 | $99.45 | 4.8% |
| Deep Learning | GRU Multivariate | 0.990 | $34.94 | $47.23 | 2.1% |
| Deep Learning | LSTM Multivariate | 0.989 | $36.20 | $48.90 | 2.2% |
| Deep Learning | RNN Multivariate | 0.987 | $38.75 | $51.34 | 2.4% |

## Web Application

### Pages

1. **Home**: Project overview and navigation
2. **Data Collection**: Data sources and preprocessing methodology
3. **Exploratory Data Analysis**: Interactive visualizations and statistical summaries
4. **Modeling**: Model architecture, training process, and performance metrics
5. **Forecast**: Real-time price predictions with trained models

### Features

- **Interactive Charts**: Plotly-based visualizations for data exploration
- **Model Selection**: Compare predictions across different models
- **Historical Analysis**: Time series plots with major economic events annotated
- **Distribution Analysis**: Feature distributions by category (market, macro, volatility)
- **Correlation Matrix**: Heatmap visualization of feature relationships

### Deployment

**Local Development**:
```bash
reflex run
```

**Production Deployment** (Render/Vercel):
```bash
reflex export
```

Configuration file: `rxconfig.py`

## Technical Stack

### Core Technologies

- **Machine Learning**: TensorFlow 2.x, Keras, scikit-learn
- **Data Processing**: pandas, NumPy, statsmodels
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Web Framework**: Reflex 0.8.9
- **Development**: Jupyter Notebook, Python 3.10+

### Dependencies

See `requirements.txt` for complete dependency list. Key packages:
- tensorflow >= 2.10.0
- scikit-learn >= 1.7.0
- pandas >= 2.3.0
- plotly >= 6.2.0
- reflex >= 0.8.9

## Contributing

This project is part of an academic research study. For questions or collaboration inquiries, please contact the repository owner.

## License

This project is developed for academic purposes. All rights reserved.

## Acknowledgments

- World Gold Council for historical gold price data
- Federal Reserve Economic Data (FRED) for macroeconomic indicators
- Geopolitical Risk Index database
- Yahoo Finance for market data

## Contact

**Project Repository**: [https://github.com/HuyPham171-hub/gold-price-prediction](https://github.com/HuyPham171-hub/gold-price-prediction)

**Author**: HuyPham171-hub

**Institution**: Greenwich University

**Course**: COMP1682.1 Final Project

---

**Last Updated**: November 2025