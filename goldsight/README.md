# GoldSight - Gold Price Prediction Web Application# Gold Price Prediction Web App - Project Structure



**GoldSight** is an advanced web application for gold price forecasting using Machine Learning and Deep Learning techniques. Built with Reflex Python framework, it provides real-time predictions, interactive visualizations, and comprehensive analysis of economic factors affecting gold prices.## 📁 Cấu trúc Thư Mục



## 🎯 Features```

Project/

- **Real-time Data Collection**: Fetch gold prices and economic indicators from Yahoo Finance and FRED API├── __init__.py                    # Package initialization

- **Exploratory Data Analysis**: Interactive charts and statistical analysis├── app.py                         # Main Reflex application (entry point)

- **Multiple ML/DL Models**: Linear Regression, Ridge, ARIMA, SVR, Random Forest, XGBoost, MLP, RNN, LSTM, GRU│

- **Multivariate Forecasting**: Predict using 12 economic indicators├── components/                    # Reusable UI components

- **Interactive Dashboard**: User-friendly interface for model comparison and predictions│   ├── __init__.py

│   ├── navbar.py                 # Navigation bar component

## 📁 Project Structure│   ├── footer.py                 # Footer component

│   ├── chart_wrapper.py          # Plotly chart wrapper

```│   └── data_table.py             # Interactive data table

goldsight/│

├── __init__.py                    # Package initialization├── pages/                         # Page components (routes)

├── app.py                         # Main Reflex application (entry point)│   ├── __init__.py

││   ├── home.py                   # Home page (/)

├── components/                    # Reusable UI components│   ├── data_collection.py        # Data collection page (/data-collection)

│   ├── __init__.py│   ├── eda.py                    # EDA page (/eda)

│   ├── navbar.py                 # Navigation bar component│   ├── modeling.py               # Modeling page (/modeling)

│   ├── footer.py                 # Footer component│   └── forecast.py               # Forecast page (/forecast)

│   ├── chart_wrapper.py          # Plotly chart wrapper│

│   └── data_table.py             # Interactive data table├── services/                      # Business logic & data pipeline

││   ├── __init__.py

├── pages/                         # Page components (routes)│   ├── data_collector.py         # Fetch data from yfinance, FRED, GPR

│   ├── __init__.py│   ├── data_preprocessor.py      # Data cleaning, resampling, feature engineering

│   ├── home.py                   # Home page (/)│   ├── model_loader.py           # Load trained models (.keras, .pkl)

│   ├── data_collection.py        # Data collection page (/data-collection)│   └── forecast_pipeline.py      # Generate forecasts (7/21/30 days)

│   ├── eda.py                    # EDA page (/eda)│

│   ├── modeling.py               # Modeling page (/modeling)├── models/                        # Saved ML/DL models

│   └── forecast.py               # Forecast page (/forecast)│   ├── best_lstm_multivariate.keras

││   ├── best_gru_multivariate.keras

├── services/                      # Business logic & data pipeline│   ├── best_rnn_multivariate.keras

│   ├── __init__.py│   ├── lstm_daily_7d.keras       # Short-term model (daily features)

│   ├── data_collector.py         # Fetch data from yfinance, FRED, GPR│   ├── scaler_X.pkl              # StandardScaler for features

│   ├── data_preprocessor.py      # Data cleaning, resampling, feature engineering│   └── scaler_y.pkl              # StandardScaler for target

│   ├── model_loader.py           # Load trained models (.keras, .pkl)│

│   └── forecast_pipeline.py      # Generate forecasts (7/21/30 days)├── data/                          # Data storage

││   ├── raw/                      # Raw data from APIs

├── utils/                         # Helper functions│   │   ├── gold_spot.csv

│   ├── __init__.py│   │   ├── market_data.csv

│   ├── date_utils.py             # Date handling utilities│   │   └── macro_data.csv

│   ├── plot_utils.py             # Plotly visualization helpers│   ├── processed/                # Processed data (cleaned, aligned)

│   └── metrics.py                # Evaluation metrics (R², RMSE, MAE)│   │   ├── combined_data.csv

││   │   └── features_engineered.csv

├── models/                        # Trained ML/DL models│   └── cache/                    # Cache for API calls (reduce requests)

│   ├── best_lstm_multivariate.keras│       └── yfinance_cache.json

│   ├── best_gru_multivariate.keras│

│   ├── best_rnn_multivariate.keras└── utils/                         # Helper functions

│   └── scaler.pkl                # Data scaler    ├── __init__.py

│    ├── date_utils.py             # Date manipulation, is_mid_month(), etc.

└── data/                          # Data storage    ├── plot_utils.py             # Plotly/Matplotlib helper functions

    ├── raw/                      # Raw data from APIs    └── metrics.py                # Evaluation metrics (R², RMSE, MAE)

    ├── processed/                # Cleaned and engineered data```

    └── cache/                    # Cached API responses

```## 🚀 Cách Chạy



## 🚀 Quick Start```bash

# Activate virtual environment (nếu có)

### 1. Installation.venv\Scripts\activate



```bash# Install dependencies

# Navigate to project directorypip install -r requirements.txt

cd c:\Users\huy\Documents\GreenWich\COMP1682.1_FinalProject\Project

# Run Reflex app

# Activate virtual environment (if not already)reflex run

.venv\Scripts\activate

# Hoặc với debug mode

# Install dependencies (Reflex already installed)reflex run --loglevel debug

pip install -r requirements.txt```

```

## 📝 Lưu Ý

### 2. Run the Application

1. **Import Path**: Vì app module là `Project`, import phải dùng:

```bash   ```python

# Navigate to goldsight folder   from Project.pages.home import home_page

cd goldsight   from Project.services.data_collector import fetch_latest_data

   ```

# Initialize Reflex (first time only)

reflex init2. **Assets**: Đặt CSS, images, plots trong thư mục `assets/` ở root (ngoài `Project/`)



# Run the application3. **Data**: Không commit data thô nặng, thêm vào `.gitignore`:

reflex run   ```

```   Project/data/cache/

   Project/models/*.keras

The app will be available at: **http://localhost:3000**   ```



## 📊 Data Sources4. **Environment Variables**: Tạo file `.env` cho API keys:

   ```

- **Yahoo Finance (yfinance)**: Gold spot prices, S&P 500, USD Index, Silver prices   FRED_API_KEY=your_key_here

- **FRED API**: CPI, Real Interest Rate, GDP, Unemployment Rate, Federal Funds Rate   ```

- **GPR Index**: Geopolitical Risk Index

## 📦 Dependencies

## 🤖 Models Implemented

Thêm vào `requirements.txt`:

### Traditional Models```

- Linear Regressionreflex>=0.5.0

- Ridge Regressionpandas>=2.0.0

- ARIMA/SARIMAnumpy>=1.24.0

yfinance>=0.2.0

### Machine Learning Modelsfredapi>=0.5.0

- Support Vector Regression (SVR)scikit-learn>=1.3.0

- Random Foresttensorflow>=2.15.0

- XGBoostplotly>=5.18.0

```

### Deep Learning Models (Multivariate)

- Multi-Layer Perceptron (MLP)## 🔄 Next Steps

- Recurrent Neural Network (RNN)

- Long Short-Term Memory (LSTM) ⭐1. ✅ Tạo cấu trúc thư mục (done)

- Gated Recurrent Unit (GRU)2. ⏳ Implement components (navbar, charts)

3. ⏳ Develop services (data_collector, forecast_pipeline)

## 🎨 Tech Stack4. ⏳ Copy models từ `notebooks/` → `Project/models/`

5. ⏳ Populate pages với nội dung từ notebooks

- **Framework**: Reflex Python 0.8.96. ⏳ Test & Deploy

- **Deep Learning**: TensorFlow 2.15+
- **Machine Learning**: scikit-learn 1.7.1
- **Data Handling**: pandas, numpy
- **Visualization**: Plotly 6.2.0
- **API Integration**: yfinance, fredapi

## 📖 Pages

### 1. Home (`/`)
- Project introduction
- Key objectives
- Navigation to main sections

### 2. Data Collection (`/data-collection`)
- Data sources overview
- Real-time data fetching
- Data quality checks

### 3. Exploratory Data Analysis (`/eda`)
- Time series visualizations
- Correlation analysis
- Statistical summaries
- Feature importance

### 4. Model Training (`/modeling`)
- Model comparison table
- Performance metrics (R², RMSE, MAE)
- Training history visualization
- Hyperparameter tuning results

### 5. Price Forecast (`/forecast`)
- Real-time gold price prediction
- Forecast horizon selection (7/21/30 days)
- Confidence intervals
- Economic indicator impact analysis

## 🔧 Configuration

Edit `rxconfig.py` to customize:
- App name
- Theme settings
- Plugins

## 📝 Environment Variables

Create a `.env` file in the project root:

```
FRED_API_KEY=your_fred_api_key_here
```

## 🧪 Development

### Adding New Pages

1. Create new file in `pages/`
2. Define page function
3. Import and add route in `app.py`

```python
from goldsight.pages.new_page import new_page_component
app.add_page(new_page_component, route="/new-page", title="New Page")
```

### Adding Services

Create service files in `services/` for:
- API integrations
- Data processing pipelines
- Model training/inference

## 🐛 Troubleshooting

### Reflex not found
```bash
pip install reflex
```

### Import errors
Ensure you're using correct module name:
```python
from goldsight.pages.home import home_page  # ✅ Correct
from Project.pages.home import home_page     # ❌ Old name
```

### Module not found
```bash
pip install -r requirements.txt --upgrade
```

## 📦 Deployment

```bash
# Build for production
reflex export

# Deploy to Reflex Cloud
reflex deploy
```

## 📄 License

This project is part of COMP1682.1 Final Project at Greenwich University.

## 👥 Contributors

- **Student**: Huy Pham
- **Course**: COMP1682.1 - Machine Learning & AI

## 🔗 Links

- [Reflex Documentation](https://reflex.dev/docs/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [GitHub Repository](https://github.com/HuyPham171-hub/gold-price-prediction)

---

**GoldSight** - Illuminate Your Investment Decisions 💡✨
