"""Forecast page - Real-time gold price prediction using GRU Multivariate model."""
import os
# Suppress TensorFlow information messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from typing import Any, Dict, List, Optional
import reflex as rx
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import joblib
from goldsight.components import page_layout, chapter_progress
from goldsight.utils.design_system import (
    Colors, Typography, Spacing, Layout,
    card_style, section_divider
)

# ======================================================================
# STATE MANAGEMENT
# ======================================================================

class ForecastState(rx.State):
    """State management for the forecast page."""
    
    # Forecast mode
    forecast_months: int = 6  # Fixed: Predict 6 months (Jun-Nov 2025)
    
    # Input variables (Default values from May 2025 data)
    cpi: float = 320.58
    silver: float = 32.89
    sp500: float = 5911.69
    usd_index: float = 122.11
    real_interest_rate: float = 2.07
    unemployment: float = 4.2
    vix: float = 18.57
    crude_oil: float = 60.79
    fed_funds_rate: float = 4.33
    tbill_10y: float = 4.42
    gpr: float = 100.0  # Default (missing in May 2025)
    gpra: float = 100.0  # Default (missing in May 2025)
    
    # Prediction outputs (for multiple months)
    forecasts: list[Dict[str, Any]] = []
    is_loading: bool = False
    error_message: str = ""

    # --- Explicit Setters (No auto-trigger, manual Run Forecast button) ---
    def set_cpi(self, value: float): 
        self.cpi = value
    
    def set_silver(self, value: float): 
        self.silver = value
    
    def set_sp500(self, value: float): 
        self.sp500 = value
    
    def set_usd_index(self, value: float): 
        self.usd_index = value
    
    def set_real_interest_rate(self, value: float): 
        self.real_interest_rate = value
    
    def set_fed_funds_rate(self, value: float): 
        self.fed_funds_rate = value
    
    def set_unemployment(self, value: float): 
        self.unemployment = value
    
    def set_vix(self, value: float): 
        self.vix = value
    
    def set_crude_oil(self, value: float): 
        self.crude_oil = value
    
    def set_treasury_10y(self, value: float): 
        self.tbill_10y = value
    
    def set_gpr(self, value: float): 
        self.gpr = value
    
    def set_gpra(self, value: float): 
        self.gpra = value
    # ---------------------------------------------

    @rx.var(cache=True)
    def model(self) -> object | None:
        """Load pre-trained GRU model (cached)."""
        try:
            import tensorflow as tf
            # Resolve absolute path to ensure file is found
            base_dir = Path(__file__).resolve().parent.parent.parent
            model_path = base_dir / "models" / "best_gru_multivariate.keras"
            
            if not model_path.exists():
                print(f"System Error: Model file not found at {model_path}")
                return None
            return tf.keras.models.load_model(str(model_path))
        except Exception as e:
            print(f"System Error: Failed to load model. {e}")
            return None
    
    @rx.var(cache=True)
    def scaler_X(self) -> object | None:
        """Load Feature Scaler (cached)."""
        try:
            base_dir = Path(__file__).resolve().parent.parent.parent
            path = base_dir / "models" / "scaler_X.pkl"
            if not path.exists(): return None
            return joblib.load(str(path))
        except Exception: return None

    @rx.var(cache=True)
    def scaler_y(self) -> object | None:
        """Load Target Scaler (cached)."""
        try:
            base_dir = Path(__file__).resolve().parent.parent.parent
            path = base_dir / "models" / "scaler_y.pkl"
            if not path.exists(): return None
            return joblib.load(str(path))
        except Exception: return None
    
    @rx.var(cache=True)
    def historical_data(self) -> pd.DataFrame:
        """Load filtered historical data (cached)."""
        try:
            base_dir = Path(__file__).resolve().parent.parent.parent
            data_path = base_dir / "data" / "filtered_data.csv"
            if not data_path.exists(): return pd.DataFrame()
            df = pd.read_csv(data_path, parse_dates=["Date"])
            return df.sort_values("Date")
        except Exception: return pd.DataFrame()
    
    def run_forecast(self):
        """Execute multi-month prediction (Jun-Nov 2025) using GRU model."""
        self.is_loading = True
        self.error_message = ""
        self.forecasts = []
        yield
        
        try:
            # Validation checks
            if self.model is None or self.scaler_X is None or self.scaler_y is None:
                self.error_message = "System Error: Model artifacts not loaded."
                self.is_loading = False
                return
            
            if self.historical_data.empty:
                self.error_message = "System Error: Historical data missing."
                self.is_loading = False
                return
            
            # Define feature columns (match training order)
            feature_cols = [
                "CPI", "Silver_Futures", "S&P_500", "USD_Index",
                "Real_Interest_Rate", "Unemployment_Rate", "VIX",
                "Crude_Oil", "Fed_Funds_Rate", "10Y_Treasury", "GPR", "GPRA"
            ]

            # Column mapping from CSV to feature names
            csv_to_features = {
                "CPI": "CPI",
                "Silver_Futures": "Silver_Futures",
                "SP_500": "S&P_500",
                "USD_Index": "USD_Index",
                "Real_Interest_Rate": "Real_Interest_Rate",
                "Unemployment": "Unemployment_Rate",
                "^VIX": "VIX",
                "Crude_Oil": "Crude_Oil",
                "Fed_Funds_Rate": "Fed_Funds_Rate",
                "Treasury_Yield_10Y": "10Y_Treasury",
                "GPR": "GPR",
                "GPRA": "GPRA"
            }

            # Get last 12 months from historical data (up to May 2025)
            hist_df = self.historical_data.tail(12).copy()
            
            # Rename columns to match feature names
            hist_df = hist_df.rename(columns=csv_to_features)
            
            # Ensure all feature columns exist
            for col in feature_cols:
                if col not in hist_df.columns:
                    hist_df[col] = 0.0
                # Fill NaN values (e.g., GPR/GPRA in Apr/May 2025)
                hist_df[col] = hist_df[col].fillna(100.0 if col in ["GPR", "GPRA"] else hist_df[col].mean())
            
            # Extract features in correct order
            hist_features = hist_df[feature_cols].values  # Shape: (12, 12)
            
            # Prepare user inputs (will be used as "current conditions" for forecasting)
            current_input = np.array([[
                self.cpi,
                self.silver,
                self.sp500,
                self.usd_index,
                self.real_interest_rate,
                self.unemployment,
                self.vix,
                self.crude_oil,
                self.fed_funds_rate,
                self.tbill_10y,
                self.gpr,
                self.gpra
            ]])  # Shape: (1, 12)
            
            # Initialize forecasts list
            forecasts_data = []
            
            # Rolling forecast for 6 months (Jun-Nov 2025)
            # Starting point: May 2025 data (last row in historical_data)
            last_date = pd.to_datetime(self.historical_data["Date"].iloc[-1])
            may_2025_gold_price = self.historical_data["Gold_Spot"].iloc[-1]
            
            for month_ahead in range(1, self.forecast_months + 1):
                # Build 12-month sequence
                if month_ahead == 1:
                    # First forecast (Jun 2025): use last 11 historical + current user input
                    sequence = np.vstack([hist_features[-11:], current_input])
                else:
                    # Subsequent forecasts (Jul-Nov 2025): roll window forward
                    # Use last 11 from sequence + current user input (assuming stable conditions)
                    sequence = np.vstack([sequence[-11:], current_input])
                
                # Scale the sequence
                X_scaled = self.scaler_X.transform(sequence)  # Shape: (12, 12)
                
                # Reshape for GRU: (1, 12, 12)
                X_input = X_scaled.reshape(1, 12, 12)
                
                # Predict
                y_pred_scaled = self.model.predict(X_input, verbose=0)
                
                # Inverse transform
                y_pred = self.scaler_y.inverse_transform(y_pred_scaled)[0, 0]
                
                # Calculate confidence interval (using RMSE from test set)
                rmse = 45.92
                confidence_lower = y_pred - 1.96 * rmse
                confidence_upper = y_pred + 1.96 * rmse
                
                # Calculate forecast date (starting from Jun 2025)
                forecast_date = last_date + pd.DateOffset(months=month_ahead)
                
                # Store forecast
                forecasts_data.append({
                    "month": forecast_date.strftime("%b %Y"),
                    "date": forecast_date,
                    "price": round(float(y_pred), 2),
                    "lower": round(float(confidence_lower), 2),
                    "upper": round(float(confidence_upper), 2),
                    "change_pct": round(((y_pred - may_2025_gold_price) / may_2025_gold_price) * 100, 2)
                })
                
                # Update sequence for next iteration (use prediction as last value)
                # Note: We keep current_input the same (stable scenario assumption)
            
            self.forecasts = forecasts_data
            
        except Exception as e:
            self.error_message = f"Prediction failed: {str(e)}"
            print(f"Prediction Exception: {e}")
            import traceback
            traceback.print_exc()
            self.forecasts = []
            
        self.is_loading = False

    @rx.var
    def forecast_chart(self) -> go.Figure:
        """Generate Plotly chart showing historical trend and 6-month forecast."""
        if self.historical_data.empty: 
            return go.Figure()
        
        # Get all historical data (full timeline to show complete context)
        hist = self.historical_data.copy()
        
        fig = go.Figure()
        
        # 1. Historical Line (show all data including May 2025)
        if "Gold_Spot" in hist.columns:
            fig.add_trace(go.Scatter(
                x=hist["Date"], 
                y=hist["Gold_Spot"],
                mode="lines+markers", 
                name="Historical Data",
                line=dict(color="#D97706", width=2),
                marker=dict(size=4, color="#D97706")
            ))
        
        # 2. Forecast Line & Points (6 months)
        if len(self.forecasts) > 0:
            last_date = pd.to_datetime(hist["Date"].iloc[-1])
            last_price = hist["Gold_Spot"].iloc[-1] if "Gold_Spot" in hist.columns else self.forecasts[0]["price"]
            
            # Connector line from last historical point to first forecast
            forecast_dates = [last_date] + [f["date"] for f in self.forecasts]
            forecast_prices = [last_price] + [f["price"] for f in self.forecasts]
            
            # Forecast line
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_prices,
                mode="lines+markers",
                name="6-Month Forecast",
                line=dict(color="#DC2626", width=3, dash="dash"),
                marker=dict(size=8, color="#DC2626", symbol="diamond")
            ))
            
            # Confidence Interval (shaded area)
            forecast_dates_only = [f["date"] for f in self.forecasts]
            lower_bounds = [f["lower"] for f in self.forecasts]
            upper_bounds = [f["upper"] for f in self.forecasts]
            
            fig.add_trace(go.Scatter(
                x=forecast_dates_only + forecast_dates_only[::-1],
                y=upper_bounds + lower_bounds[::-1],
                fill='toself',
                fillcolor='rgba(220, 38, 38, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence',
                showlegend=True,
                hoverinfo='skip'
            ))
            
        fig.update_layout(
            title={
                'text': "Gold Price Historical & Forecast: 2006-2025 (USD/oz)",
                'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top',
                'font': {'size': 18, 'weight': 'bold'}
            },
            xaxis_title="Date",
            yaxis_title="Gold Price (USD/oz)",
            margin=dict(l=60, r=40, t=80, b=60),
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="right", 
                x=1,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            font=dict(family="Inter, sans-serif", size=12)
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        
        return fig


# ======================================================================
# UI HELPER COMPONENTS
# ======================================================================

def slider_control(
    label: str, 
    value_var: rx.Var, 
    set_fn: callable, 
    min_val: float, 
    max_val: float, 
    step: float, 
    unit: str = ""
) -> rx.Component:
    """Reusable slider component with label and value badge."""
    return rx.vstack(
        rx.hstack(
            rx.text(label, size="2", weight="medium", color="var(--gray-11)"),
            rx.spacer(),
            rx.badge(
                f"{value_var}{unit}", 
                color_scheme="gray", 
                variant="soft",
                size="1"
            ),
            width="100%",
            align="center"
        ),
        rx.slider(
            default_value=[value_var],
            min=min_val,
            max=max_val,
            step=step,
            on_value_commit=lambda v: set_fn(v[0]),
            color_scheme="amber",
            width="100%",
            size="1"
        ),
        spacing="1",
        width="100%"
    )

def result_stat_card(label: str, value: str, subtext: str, color: str) -> rx.Component:
    """Statistic display card."""
    return rx.vstack(
        rx.text(label, size="1", weight="medium", color="var(--gray-10)", text_transform="uppercase", letter_spacing="0.05em"),
        rx.heading(value, size="6", weight="bold", color=rx.color(color, 10)),
        rx.text(subtext, size="1", color="var(--gray-9)"),
        spacing="1",
        align="center",
        padding="1em",
        border="1px solid",
        border_color=rx.color("gray", 4),
        border_radius="var(--radius-3)",
        background="white",
        width="100%"
    )

# ======================================================================
# MAIN PAGE LAYOUT
# ======================================================================

def forecast_page() -> rx.Component:
    return page_layout(
        rx.flex(
            rx.vstack(
                # 1. Navigation Progress
                chapter_progress(current=4),
                
                # 2. Page Header
                rx.vstack(
                    rx.heading("Interactive Forecast Tool", size="8", color_scheme="amber"),
                    rx.text(
                        "6-Month Gold Price Forecast (Jun-Nov 2025) based on May 2025 Economic Data", 
                        color="var(--gray-11)",
                        size="3"
                    ),
                    spacing="2",
                    align="center",
                    margin_bottom="2em",
                    width="100%"
                ),

                # 3. Main Grid Layout
                rx.grid(
                    
                    # --- LEFT COLUMN: Chart & Results ---
                    rx.vstack(
                        # Forecast Table (6 months)
                        rx.cond(
                            ForecastState.forecasts.length() > 0,
                            rx.vstack(
                                rx.heading("Forecast Summary", size="4", weight="bold", margin_bottom="0.5em"),
                                rx.table.root(
                                    rx.table.header(
                                        rx.table.row(
                                            rx.table.column_header_cell("Month"),
                                            rx.table.column_header_cell("Predicted Price"),
                                            rx.table.column_header_cell("Lower Bound (95%)"),
                                            rx.table.column_header_cell("Upper Bound (95%)"),
                                            rx.table.column_header_cell("Change vs May '25"),
                                        )
                                    ),
                                    rx.table.body(
                                        rx.foreach(
                                            ForecastState.forecasts,
                                            lambda forecast: rx.table.row(
                                                rx.table.cell(forecast["month"]),
                                                rx.table.cell(
                                                    rx.heading(f"${forecast['price']}", size="3", color=rx.color("amber", 10))
                                                ),
                                                rx.table.cell(f"${forecast['lower']}"),
                                                rx.table.cell(f"${forecast['upper']}"),
                                                rx.table.cell(
                                                    rx.text(
                                                        f"{forecast['change_pct']}%",
                                                        weight="bold",
                                                        color=rx.color("green", 10)
                                                    )
                                                ),
                                            )
                                        )
                                    ),
                                    variant="surface",
                                    size="2",
                                    width="100%"
                                ),
                                spacing="3",
                                width="100%",
                                padding="1.5em",
                                background="white",
                                border=f"1px solid {rx.color('gray', 4)}",
                                border_radius="var(--radius-4)",
                                margin_bottom="1em"
                            ),
                            # Empty State
                            rx.box(
                                rx.text(
                                    "Configure economic scenarios on the right panel and click 'Run Forecast' to generate predictions.",
                                    color="var(--gray-10)",
                                    style={"font_style": "italic"}
                                ),
                                padding="2em",
                                width="100%",
                                align="center",
                                text_align="center",
                                border="1px dashed var(--gray-5)",
                                border_radius="var(--radius-3)"
                            )
                        ),
                        
                        # Chart Container
                        rx.box(
                            rx.plotly(
                                data=ForecastState.forecast_chart, 
                                style={"width": "100%", "height": "550px"}
                            ),
                            width="100%",
                            padding="1em",
                            background="white",
                            border=f"1px solid {rx.color('gray', 4)}",
                            border_radius="var(--radius-4)",
                            box_shadow="0 4px 20px rgba(0,0,0,0.05)"
                        ),
                        width="100%",
                        spacing="4"
                    ),

                    # --- RIGHT COLUMN: Control Panel ---
                    rx.box(
                        rx.vstack(
                            rx.heading("Scenario Inputs", size="3", weight="bold"),
                            rx.text(
                                "Adjust sliders to configure economic scenarios, then click 'Run Forecast' below.",
                                size="1",
                                color="var(--gray-10)",
                                line_height="1.5"
                            ),
                            rx.divider(),
                            
                            # Scrollable Area for Sliders
                            rx.scroll_area(
                                rx.vstack(
                                    rx.text("Key Economic Drivers", size="1", weight="bold", color="var(--gray-9)"),
                                    slider_control("CPI (Inflation)", ForecastState.cpi, ForecastState.set_cpi, 250, 350, 0.5),
                                    slider_control("Silver Futures", ForecastState.silver, ForecastState.set_silver, 20, 50, 0.5, "$"),
                                    slider_control("S&P 500", ForecastState.sp500, ForecastState.set_sp500, 4000, 7000, 50),
                                    
                                    rx.divider(),
                                    
                                    rx.text("Monetary & Fiscal", size="1", weight="bold", color="var(--gray-9)"),
                                    slider_control("Fed Funds Rate", ForecastState.fed_funds_rate, ForecastState.set_fed_funds_rate, 0, 6, 0.25, "%"),
                                    slider_control("10Y Treasury Yield", ForecastState.tbill_10y, ForecastState.set_treasury_10y, 2, 6, 0.1, "%"),
                                    slider_control("Real Interest Rate", ForecastState.real_interest_rate, ForecastState.set_real_interest_rate, -2, 5, 0.1, "%"),
                                    
                                    rx.divider(),
                                    
                                    rx.text("Market Conditions", size="1", weight="bold", color="var(--gray-9)"),
                                    slider_control("USD Index", ForecastState.usd_index, ForecastState.set_usd_index, 90, 135, 0.5),
                                    slider_control("Crude Oil", ForecastState.crude_oil, ForecastState.set_crude_oil, 50, 120, 1, "$"),
                                    slider_control("VIX (Fear Index)", ForecastState.vix, ForecastState.set_vix, 10, 50, 0.5),
                                    slider_control("Unemployment", ForecastState.unemployment, ForecastState.set_unemployment, 3, 8, 0.1, "%"),
                                    
                                    rx.divider(),
                                    
                                    rx.text("Geopolitical Risk", size="1", weight="bold", color="var(--gray-9)"),
                                    slider_control("GPR Index", ForecastState.gpr, ForecastState.set_gpr, 50, 200, 1),
                                    slider_control("GPRA Index", ForecastState.gpra, ForecastState.set_gpra, 50, 200, 1),
                                    
                                    spacing="4",
                                    width="100%",
                                    padding_right="1em"
                                ),
                                type="auto",
                                scrollbars="vertical",
                                style={"height": "450px"}
                            ),
                            
                            rx.divider(),
                            
                            # Run Forecast Button
                            rx.button(
                                "Run Forecast",
                                on_click=ForecastState.run_forecast,
                                loading=ForecastState.is_loading,
                                width="100%",
                                size="3",
                                color_scheme="amber",
                                variant="solid",
                                style={"cursor": "pointer"}
                            ),
                            
                            # Error Display
                            rx.cond(
                                ForecastState.error_message != "",
                                rx.callout(
                                    ForecastState.error_message,
                                    icon="triangle-alert",
                                    color_scheme="red",
                                    size="1",
                                    width="100%"
                                )
                            ),
                            
                            spacing="4",
                            width="100%",
                            align="stretch"
                        ),
                        padding="1.5em",
                        border=f"1px solid {rx.color('gray', 4)}",
                        border_radius="var(--radius-4)",
                        background="white",
                        width="100%"
                    ),

                    # Grid Configuration
                    columns="3fr 1fr",
                    spacing="6",
                    width="100%",
                    align_items="start"
                ),
                
                section_divider(),
                
                # Footer
                rx.box(
                    rx.text("GoldSight System v1.0 - Final Year Project", size="2", color="var(--gray-9)"),
                    width="100%",
                    text_align="center",
                    padding_y="2em"
                ),
                
                spacing="6",
                width="100%",
                align="center"
            ),
            justify="center",
            width="100%",
            max_width=Layout.MAX_WIDTH_WIDE,
            margin_x="auto",
            padding_x="2em",
            padding_y="2em"
        )
    )