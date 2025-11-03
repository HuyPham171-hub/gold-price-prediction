# GoldSight - Quick Start Guide

## ✅ Setup Complete

Your **GoldSight** application is ready to run!

```
goldsight/
├── app.py                      ✅ Main Reflex app (updated)
├── components/                 ✅ UI components folder
├── pages/                      ✅ 5 pages (home, data, eda, modeling, forecast)
├── services/                   ✅ Business logic folder
├── utils/                      ✅ Helper functions folder
├── models/                     ✅ Model storage
└── data/                       ✅ Data storage (raw, processed, cache)
```

## 🚀 Run the Application

```powershell
# You are already in the right directory!
cd goldsight
reflex init
reflex run
```

**Access at**: http://localhost:3000

## 📄 Available Routes

- **Home** - `/` - Landing page with project overview
- **Data Collection** - `/data-collection` - Data fetching from APIs
- **EDA** - `/eda` - Exploratory Data Analysis
- **Modeling** - `/modeling` - Model training and comparison
- **Forecast** - `/forecast` - Gold price predictions

## 🔄 Next Steps

### 1. Copy Trained Models

```powershell
# Copy models from parent directory
Copy-Item ..\models\best_lstm_multivariate.keras goldsight\models\
Copy-Item ..\models\best_gru_multivariate.keras goldsight\models\
Copy-Item ..\models\best_rnn_multivariate.keras goldsight\models\
```

### 2. Create Navigation Bar Component

**Create `components/navbar.py`:**

```python
import reflex as rx

def navbar() -> rx.Component:
    """Navigation bar component."""
    return rx.box(
        rx.hstack(
            rx.heading("GoldSight", size="7", color="white"),
            rx.spacer(),
            rx.hstack(
                rx.link("Home", href="/", color="white", _hover={"color": "yellow.300"}),
                rx.link("Data", href="/data-collection", color="white", _hover={"color": "yellow.300"}),
                rx.link("EDA", href="/eda", color="white", _hover={"color": "yellow.300"}),
                rx.link("Modeling", href="/modeling", color="white", _hover={"color": "yellow.300"}),
                rx.link("Forecast", href="/forecast", color="white", _hover={"color": "yellow.300"}),
                spacing="6"
            ),
            width="100%",
            max_width="1200px",
            padding_x="2rem",
            padding_y="1rem",
        ),
        bg="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        width="100%",
        position="sticky",
        top="0",
        z_index="999",
        box_shadow="0 2px 10px rgba(0,0,0,0.1)"
    )
```

### 3. Add Navbar to All Pages

Update each page file:

```python
from goldsight.components.navbar import navbar

def home_page() -> rx.Component:
    return rx.container(
        navbar(),  # Add this line
        rx.vstack(
            # ... existing content
        )
    )
```

### 4. Create Data Collection Service

**Create `services/data_collector.py`:**

```python
"""Data collection service for fetching gold price and economic indicators."""
import yfinance as yf
from fredapi import Fred
import pandas as pd
from datetime import datetime, timedelta

def fetch_gold_price(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch gold spot price from Yahoo Finance."""
    gold = yf.download("GC=F", start=start_date, end=end_date)
    return gold[['Close']].rename(columns={'Close': 'Gold_Spot'})

def fetch_economic_indicators(api_key: str, start_date: str) -> pd.DataFrame:
    """Fetch economic indicators from FRED API."""
    fred = Fred(api_key=api_key)
    
    indicators = {
        'CPI': 'CPIAUCSL',
        'Real_Interest_Rate': 'REAINTRATREARAT10Y',
        'USD_Index': 'DTWEXBGS',
        'VIX': 'VIXCLS',
        'SP500': 'SP500',
    }
    
    data = {}
    for name, series_id in indicators.items():
        try:
            data[name] = fred.get_series(series_id, start_date)
        except Exception as e:
            print(f"Error fetching {name}: {e}")
    
    return pd.DataFrame(data)

def fetch_latest_data(api_key: str, months: int = 12) -> pd.DataFrame:
    """Fetch latest data for prediction."""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=months*30)).strftime("%Y-%m-%d")
    
    gold = fetch_gold_price(start_date, end_date)
    indicators = fetch_economic_indicators(api_key, start_date)
    
    # Combine data
    combined = pd.concat([gold, indicators], axis=1)
    return combined.ffill()
```

### 5. Create Forecast Pipeline

**Create `services/forecast_pipeline.py`:**

```python
"""Forecast pipeline for gold price predictions."""
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def generate_forecast(model_path: str, data: pd.DataFrame, horizon: int = 7) -> dict:
    """
    Generate gold price forecast.
    
    Args:
        model_path: Path to trained model (.keras)
        data: Historical data (last 60 days minimum)
        horizon: Forecast horizon in days
    
    Returns:
        Dictionary with forecast dates and predicted prices
    """
    # Load model
    model = load_model(model_path)
    
    # Prepare data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create sequence (last 60 days)
    sequence_length = 60
    last_sequence = scaled_data[-sequence_length:]
    
    # Generate predictions
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(horizon):
        # Predict next value
        pred = model.predict(current_sequence.reshape(1, sequence_length, -1), verbose=0)
        predictions.append(pred[0, 0])
        
        # Update sequence (rolling window)
        current_sequence = np.append(current_sequence[1:], pred, axis=0)
    
    # Inverse transform predictions
    dummy = np.zeros((len(predictions), data.shape[1]))
    dummy[:, 0] = predictions
    predictions_original = scaler.inverse_transform(dummy)[:, 0]
    
    # Generate dates
    last_date = data.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
    
    return {
        'dates': forecast_dates.tolist(),
        'prices': predictions_original.tolist()
    }
```

### 6. Create Visualization Utilities

**Create `utils/plot_utils.py`:**

```python
"""Plotting utilities for GoldSight."""
import plotly.graph_objects as go
import pandas as pd

def plot_time_series(df: pd.DataFrame, column: str = 'Gold_Spot', title: str = 'Gold Price Over Time'):
    """Create time series line chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[column],
        mode='lines',
        name=column,
        line=dict(color='gold', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def plot_forecast(historical: pd.DataFrame, forecast_dates: list, forecast_prices: list):
    """Plot historical data with forecast."""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical.index,
        y=historical['Gold_Spot'],
        mode='lines',
        name='Historical',
        line=dict(color='blue', width=2)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_prices,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Gold Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def plot_correlation_matrix(df: pd.DataFrame):
    """Create correlation heatmap."""
    corr = df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        template='plotly_white'
    )
    
    return fig
```

## 🎨 Customization

### Change Theme Color

Edit `app.py`:

```python
app = rx.App(
    theme=rx.theme(
        appearance="dark",  # or "light"
        accent_color="gold",  # gold theme for GoldSight!
    )
)
```

### Add Custom Styles

Create `assets/custom.css`:

```css
.gold-gradient {
    background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
}

.card-hover:hover {
    transform: translateY(-5px);
    transition: transform 0.3s ease;
    box-shadow: 0 10px 25px rgba(0,0,0,0.2);
}
```

## 🐛 Common Issues

### "reflex: command not found"
```powershell
# Ensure virtual environment is activated
.venv\Scripts\activate

# Reinstall Reflex
pip install reflex --upgrade
```

### Import Errors
```powershell
# Verify you're in the correct directory
cd goldsight

# Check if __init__.py exists in all folders
Get-ChildItem -Recurse -Filter "__init__.py"
```

### Port Already in Use
```powershell
# Change port in rxconfig.py
config = rx.Config(
    app_name="goldsight",
    port=8000  # Change from default 3000
)
```

## 📦 Production Deployment

### Build for Production

```powershell
reflex export
```

### Deploy to Reflex Cloud

```powershell
reflex deploy
```

## 📚 Learn More

- [Reflex Documentation](https://reflex.dev/docs/)
- [Plotly Python](https://plotly.com/python/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)

---

**Happy Forecasting with GoldSight!** 🌟📈
