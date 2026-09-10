# GoldSight: An AI-Powered Gold Price Prediction Journey

## Why Predict Gold?
Gold is a critical asset in financial markets, serving as a hedge against inflation and economic instability. Predicting its price helps investors make better decisions. Traditional forecasting methods often fail to capture the complex market dynamics, making Machine Learning an ideal approach.

## Our Objective
This project aims to develop a system using Machine Learning to forecast gold prices. We collected historical gold data and 13+ related economic indicators to train predictive models. This interactive platform visualizes trends and provides forecasts, allowing users to analyze price models and make informed decisions.

## System Architecture & Data Pipeline
Our end-to-end system integrates data collection, preprocessing, model training, and deployment. Below are the high-level architecture and detailed data pipeline workflows.

### High-Level System Architecture
*The system consists of four main components: Data Sources (yfinance, FRED, manual CSV), Processing Environment (data cleaning & model training), Storage (trained models & preprocessed datasets), and Application Layer (interactive web interface built with Reflex).*

- **Data Sources (APIs & CSV)** ➔ **Processing Environment (Jupyter Notebooks)** ➔ **Storage (Models & Datasets)** ➔ **Application Layer (Reflex Web App)**

### Data Pipeline Workflow
*Our data pipeline automates the complete workflow: raw data ingestion from multiple sources, frequency alignment to monthly intervals, forward-fill imputation for missing values, VIF-based feature engineering with standardization, resulting in a clean multivariate dataset ready for modeling.*

1. **Raw Data Ingestion (yfinance, FRED, Manual CSV)**
2. **Data Synchronization (Frequency Alignment)**
3. **Preprocessing (Forward-fill Imputation)**
4. **Feature Engineering (VIF Analysis & Scaling)**
5. **Multivariable Dataset (Ready to Train)**

## Explore Our Research Journey
Follow our step-by-step process, from data to deployment.

- **Chapter 1: The Data:** See the 13+ market and macro indicators we collected.
- **Chapter 2: The Exploration:** Discover the key correlations and insights from our EDA.
- **Chapter 3: The Models:** Our journey comparing 11 models, from ARIMA to LSTM.
- **Final App: The Forecast Tool:** Try our best-performing model to get live forecasts.
- **Source Code:** View the complete source code and notebooks on GitHub.