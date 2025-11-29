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
    
    # Input variables (Default values approximating current market conditions)
    cpi: float = 315.0
    silver: float = 32.5
    sp500: float = 5200.0
    usd_index: float = 105.0
    real_interest_rate: float = 2.1
    unemployment: float = 3.8
    vix: float = 14.5
    crude_oil: float = 82.0
    fed_funds_rate: float = 5.25
    tbill_10y: float = 4.5
    gpr: float = 100.0
    gpra: float = 100.0
    
    # Prediction outputs
    predicted_price: float = 0.0
    confidence_lower: float = 0.0
    confidence_upper: float = 0.0
    price_change_pct: float = 0.0
    is_loading: bool = False
    error_message: str = ""

    # --- Explicit Setters (Required by Reflex) ---
    def set_cpi(self, value: float): self.cpi = value
    def set_silver(self, value: float): self.silver = value
    def set_sp500(self, value: float): self.sp500 = value
    def set_usd_index(self, value: float): self.usd_index = value
    def set_real_interest_rate(self, value: float): self.real_interest_rate = value
    def set_fed_funds_rate(self, value: float): self.fed_funds_rate = value
    def set_unemployment(self, value: float): self.unemployment = value
    def set_vix(self, value: float): self.vix = value
    def set_crude_oil(self, value: float): self.crude_oil = value
    def set_treasury_10y(self, value: float): self.tbill_10y = value
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
        """Execute prediction using the loaded GRU model."""
        self.is_loading = True
        self.error_message = ""
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
            
            # 1. Prepare Historical Context (Last 11 months)
            # Define exact feature order used during training
            feature_cols = [
                "CPI", "Silver_Futures", "S&P_500", "USD_Index",
                "Real_Interest_Rate", "Unemployment_Rate", "VIX",
                "Crude_Oil", "Fed_Funds_Rate", "10Y_Treasury", "GPR", "GPRA"
            ]

            # Map state variables to dataframe columns
            current_input = {
                "CPI": self.cpi,
                "Silver_Futures": self.silver,
                "S&P_500": self.sp500,
                "USD_Index": self.usd_index,
                "Real_Interest_Rate": self.real_interest_rate,
                "Unemployment_Rate": self.unemployment,
                "VIX": self.vix,
                "Crude_Oil": self.crude_oil,
                "Fed_Funds_Rate": self.fed_funds_rate,
                "10Y_Treasury": self.tbill_10y,
                "GPR": self.gpr,
                "GPRA": self.gpra
            }
            
            # Handle potential missing columns in historical CSV (if any)
            available_cols = [c for c in feature_cols if c in self.historical_data.columns]
            hist_features = self.historical_data[available_cols].tail(11).copy()

            # If CSV is missing columns, fill with 0.0
            for col in feature_cols:
                if col not in hist_features.columns:
                    hist_features[col] = 0.0

            # Create 12th row (Current User Scenario)
            current_df = pd.DataFrame([current_input])

            # Combine to get a 12-month sequence
            hist_features = hist_features[feature_cols]
            current_df = current_df[feature_cols]
            full_seq_df = pd.concat([hist_features, current_df], ignore_index=True)

            # 2. Scale Data
            X_raw = full_seq_df.values # Shape (12, 12)
            X_scaled = self.scaler_X.transform(X_raw)

            # 3. Reshape for GRU (1 sample, 12 steps, 12 features)
            X_input = X_scaled.reshape(1, 12, 12)
            
            # 4. Predict
            y_pred_scaled = self.model.predict(X_input, verbose=0)
            
            # 5. Inverse Transform
            # scaler_y expects shape (n, 1)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)[0, 0]
            
            # 6. Post-processing results
            # Using test set RMSE from report for confidence interval
            rmse = 45.92 
            self.predicted_price = round(float(y_pred), 2)
            self.confidence_lower = round(float(y_pred - 1.96 * rmse), 2)
            self.confidence_upper = round(float(y_pred + 1.96 * rmse), 2)
            
            last_actual = self.historical_data["Gold_Spot"].iloc[-1] if "Gold_Spot" in self.historical_data.columns else 2000.0
            self.price_change_pct = round(((y_pred - last_actual) / last_actual) * 100, 2)
            
        except Exception as e:
            self.error_message = f"Prediction failed: {str(e)}"
            print(f"Prediction Exception: {e}")
            import traceback
            traceback.print_exc()
            self.predicted_price = 0.0
            
        self.is_loading = False

    @rx.var
    def forecast_chart(self) -> go.Figure:
        """Generate Plotly chart showing historical trend and forecast point."""
        if self.historical_data.empty: 
            return go.Figure()
        
        # Get recent history (last 24 months) for context
        hist = self.historical_data.tail(24)
        
        fig = go.Figure()
        
        # 1. Historical Line
        if "Gold_Spot" in hist.columns:
            fig.add_trace(go.Scatter(
                x=hist["Date"], y=hist["Gold_Spot"],
                mode="lines", name="Historical Data",
                line=dict(color="#D97706", width=3) # Amber-600
            ))
        
        # 2. Forecast Point & CI (Only if prediction exists)
        if self.predicted_price > 0:
            last_date = pd.to_datetime(hist["Date"].iloc[-1])
            next_date = last_date + pd.DateOffset(months=1)
            
            # Connector line (dotted)
            last_val = hist["Gold_Spot"].iloc[-1] if "Gold_Spot" in hist.columns else self.predicted_price
            fig.add_trace(go.Scatter(
                x=[last_date, next_date],
                y=[last_val, self.predicted_price],
                mode="lines",
                name="Projection",
                line=dict(color="#D97706", width=2, dash="dot"),
                showlegend=False
            ))

            # The prediction point
            fig.add_trace(go.Scatter(
                x=[next_date], y=[self.predicted_price],
                mode="markers", name="Forecast",
                marker=dict(color="#DC2626", size=12, symbol="circle") # Red-600
            ))
            
            # Confidence Interval Area (Error Bars)
            fig.add_trace(go.Scatter(
                x=[next_date, next_date],
                y=[self.confidence_lower, self.confidence_upper],
                mode="lines",
                name="95% Confidence",
                line=dict(color="#DC2626", width=3),
                showlegend=True
            ))
            
        fig.update_layout(
            title={
                'text': "Gold Price Projection (USD/oz)",
                'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
            },
            margin=dict(l=40, r=20, t=60, b=40),
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="Inter, sans-serif")
        )
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
                        "Real-time scenario analysis using the Multivariate GRU Model.", 
                        color="var(--gray-11)",
                        size="3"
                    ),
                    spacing="2",
                    align="center",
                    margin_bottom="2em",
                    width="100%"
                ),

                # 3. Main Grid Layout
                # 3fr (Left: Visualization) | 1fr (Right: Controls)
                rx.grid(
                    
                    # --- LEFT COLUMN: Chart & Results ---
                    rx.vstack(
                        # Results Summary Row
                        rx.cond(
                            ForecastState.predicted_price > 0,
                            rx.grid(
                                result_stat_card("Forecast Price", f"${ForecastState.predicted_price}", "Next Month Prediction", "amber"),
                                result_stat_card("Lower Bound", f"${ForecastState.confidence_lower}", "Conservative (95%)", "gray"),
                                result_stat_card("Upper Bound", f"${ForecastState.confidence_upper}", "Optimistic (95%)", "gray"),
                                result_stat_card("Change", f"{ForecastState.price_change_pct}%", "Vs. Last Close", "blue"),
                                columns="4",
                                spacing="4",
                                width="100%"
                            ),
                            # Empty State (Placeholder)
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
                                style={"width": "100%", "height": "500px"} # Fixed height prevents overflow
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
                            rx.divider(),
                            
                            # Scrollable Area for Sliders
                            rx.scroll_area(
                                rx.vstack(
                                    rx.text("Key Drivers", size="1", weight="bold", color="var(--gray-9)"),
                                    slider_control("CPI (Inflation)", ForecastState.cpi, ForecastState.set_cpi, 250, 350, 1),
                                    slider_control("Silver Futures", ForecastState.silver, ForecastState.set_silver, 20, 50, 0.5, "$"),
                                    slider_control("S&P 500", ForecastState.sp500, ForecastState.set_sp500, 4000, 6500, 50),
                                    
                                    rx.divider(),
                                    
                                    rx.text("Secondary Factors", size="1", weight="bold", color="var(--gray-9)"),
                                    slider_control("USD Index", ForecastState.usd_index, ForecastState.set_usd_index, 90, 115, 0.1),
                                    slider_control("Real Interest Rate", ForecastState.real_interest_rate, ForecastState.set_real_interest_rate, -2, 4, 0.1, "%"),
                                    slider_control("Crude Oil", ForecastState.crude_oil, ForecastState.set_crude_oil, 60, 100, 1, "$"),
                                    slider_control("VIX (Fear Index)", ForecastState.vix, ForecastState.set_vix, 10, 40, 0.5),
                                    slider_control("Unemployment", ForecastState.unemployment, ForecastState.set_unemployment, 3, 6, 0.1, "%"),
                                    
                                    spacing="4",
                                    width="100%",
                                    padding_right="1em"
                                ),
                                type="auto",
                                scrollbars="vertical",
                                style={"height": "450px"} # Match chart height approx
                            ),
                            
                            rx.divider(),
                            
                            # Action Button
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