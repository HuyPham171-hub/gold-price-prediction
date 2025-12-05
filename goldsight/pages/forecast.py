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
    
    # Prediction outputs (for multiple months)
    forecasts: list[Dict[str, Any]] = []
    is_loading: bool = False
    error_message: str = ""

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
    
    @rx.var
    def historical_data(self) -> pd.DataFrame:
        """Load filtered historical data (NOT cached - always fresh)."""
        try:
            base_dir = Path(__file__).resolve().parent.parent.parent
            data_path = base_dir / "data" / "filtered_data.csv"
            if not data_path.exists(): return pd.DataFrame()
            # Load without parsing dates first
            df = pd.read_csv(data_path)
            # Parse dates with mixed format (file contains both M/D/YYYY and YYYY-MM-DD)
            df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=False)
            df.set_index('Date', inplace=True)
            return df.sort_index()
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
            
            # Debug: Print data info
            print(f"\n=== FORECAST DEBUG ===")
            print(f"Total rows in historical_data: {len(self.historical_data)}")
            print(f"Date range: {self.historical_data.index[0]} to {self.historical_data.index[-1]}")
            print(f"Last date: {self.historical_data.index[-1]}")
            print(f"Last Gold_Spot price: ${self.historical_data['Gold_Spot'].iloc[-1]:.2f}")
            
            # Get last 12 months data (same as notebook)
            last_12_rows = self.historical_data.tail(12)
            
            # Extract features (exclude Gold_Spot which is the target)
            X_last_12 = last_12_rows.drop(columns=["Gold_Spot"]).values  # Shape: (12, 12)
            
            # Get baseline gold price (May 2025)
            baseline_price = self.historical_data["Gold_Spot"].iloc[-1]
            baseline_date = pd.to_datetime(self.historical_data.index[-1])
            
            print(f"Baseline date: {baseline_date}")
            print(f"Baseline price: ${baseline_price:.2f}")
            print(f"X_last_12 shape: {X_last_12.shape}")
            print(f"======================\n")
            
            # Rolling forecast for 6 months
            forecasts_data = []
            rmse = 45.92
            
            for i in range(self.forecast_months):
                # Scale features
                X_scaled = self.scaler_X.transform(X_last_12)
                
                # Reshape for GRU: (1, 12, 12)
                X_input = X_scaled.reshape(1, 12, 12)
                
                # Predict
                y_pred_scaled = self.model.predict(X_input, verbose=0)
                y_pred = self.scaler_y.inverse_transform(y_pred_scaled)[0, 0]
                
                # Calculate confidence interval
                confidence_lower = y_pred - 1.96 * rmse
                confidence_upper = y_pred + 1.96 * rmse
                
                # Calculate forecast date
                forecast_date = baseline_date + pd.DateOffset(months=i+1)
                change_pct = ((y_pred - baseline_price) / baseline_price) * 100
                
                # Store forecast
                forecasts_data.append({
                    "month": forecast_date.strftime("%b %Y"),
                    "date": forecast_date,
                    "price": round(float(y_pred), 2),
                    "lower": round(float(confidence_lower), 2),
                    "upper": round(float(confidence_upper), 2),
                    "change_pct": round(float(change_pct), 2)
                })
                
                # Roll window: drop first row, append last row (stable scenario)
                X_last_12 = np.vstack([X_last_12[1:], X_last_12[-1]])
            
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
        
        # Get historical data from 2020 onwards (reduced range for clarity)
        hist = self.historical_data.copy()
        hist = hist[hist.index >= '2020-01-01']
        
        fig = go.Figure()
        
        # 1. Historical Line (2020 to May 2025)
        if "Gold_Spot" in hist.columns:
            fig.add_trace(go.Scatter(
                x=hist.index, 
                y=hist["Gold_Spot"],
                mode="lines", 
                name="Historical Data (2020-2025)",
                line=dict(color="#D97706", width=2),
                hovertemplate="<b>%{x|%b %Y}</b><br>Price: $%{y:.2f}<extra></extra>"
            ))
        
        # 2. Forecast Line & Points (6 months: Jun-Nov 2025)
        if len(self.forecasts) > 0:
            last_date = pd.to_datetime(hist.index[-1])
            last_price = hist["Gold_Spot"].iloc[-1] if "Gold_Spot" in hist.columns else self.forecasts[0]["price"]
            
            # Connector line from last historical point to first forecast
            forecast_dates = [last_date] + [f["date"] for f in self.forecasts]
            forecast_prices = [last_price] + [f["price"] for f in self.forecasts]
            
            # Forecast line
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_prices,
                mode="lines+markers",
                name="6-Month Forecast (Jun-Nov 2025)",
                line=dict(color="#DC2626", width=3, dash="dash"),
                marker=dict(size=10, color="#DC2626", symbol="diamond"),
                hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: $%{y:.2f}<extra></extra>"
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
                name='95% Confidence Interval',
                showlegend=True,
                hoverinfo='skip'
            ))
            
        fig.update_layout(
            title={
                'text': "Gold Price Historical (2020-2025) & 6-Month Forecast",
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
                    rx.heading("Gold Price Forecast Tool", size="8", color_scheme="amber"),
                    rx.text(
                        "6-Month Forecast: Jun-Nov 2025 using GRU Multivariate Model", 
                        color="var(--gray-11)",
                        size="3",
                        weight="medium"
                    ),
                    rx.text(
                        "Based on 12 months of historical data (Jun 2024 - May 2025)",
                        color="var(--gray-10)",
                        size="2",
                        style={"font_style": "italic"}
                    ),
                    spacing="2",
                    align="center",
                    margin_bottom="2em",
                    width="100%"
                ),

                # 3. Main Grid Layout
                rx.grid(
                    
                    # --- Main Content: Chart & Results ---
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
                                    "Click 'Run Forecast' button below to generate 6-month gold price predictions.",
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

                    # Grid Configuration
                    columns="1",
                    spacing="6",
                    width="100%",
                    align_items="start"
                ),
                
                # Run Forecast Button (centered)
                rx.box(
                    rx.button(
                        "Run Forecast",
                        on_click=ForecastState.run_forecast,
                        loading=ForecastState.is_loading,
                        size="3",
                        color_scheme="amber",
                        variant="solid",
                        style={"cursor": "pointer"}
                    ),
                    width="100%",
                    text_align="center",
                    margin_y="2em"
                ),
                
                # Error Display
                rx.cond(
                    ForecastState.error_message != "",
                    rx.callout(
                        ForecastState.error_message,
                        icon="triangle-alert",
                        color_scheme="red",
                        size="2",
                        width="100%"
                    )
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