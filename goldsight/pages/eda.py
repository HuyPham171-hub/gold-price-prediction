"""Chapter 2: Exploratory Data Analysis - Understanding the Gold Price Landscape"""

import reflex as rx
import json
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from goldsight.components import page_layout, chapter_progress

# ======================================================================
# HELPER COMPONENTS
# ======================================================================

def section_divider() -> rx.Component:
    """A simple visual divider for sections."""
    return rx.divider(margin_y="1.5em")


def load_plotly_chart(chart_name: str) -> go.Figure:
    """
    Generic function to load any Plotly chart from JSON cache.
    
    Args:
        chart_name: Name of the chart file (without .json extension)
                   e.g., "gold_currency_heatmap", "correlation_matrix"
    
    Returns:
        Plotly Figure object, or empty figure with error message if failed
    
    Usage:
        rx.plotly(data=load_plotly_chart("gold_currency_heatmap"), width="900px")
    """
    try:
        # Construct path to cache
        cache_path = Path(__file__).parent.parent / "data" / "cache" / f"{chart_name}.json"
        
        if not cache_path.exists():
            return go.Figure().update_layout(
                title=f"Chart '{chart_name}' not found",
                annotations=[{
                    "text": f"Run explore.ipynb to generate {chart_name}.json",
                    "showarrow": False,
                    "x": 0.5,
                    "y": 0.5,
                    "font": {"size": 16, "color": "gray"}
                }],
                height=400
            )
        
        # Load JSON and convert to Figure (keep original layout from notebook)
        with open(cache_path, 'r') as f:
            fig_dict = json.load(f)
        
        fig = go.Figure(fig_dict)
        
        return fig
        
    except Exception as e:
        return go.Figure().update_layout(
            title=f"Error loading '{chart_name}'",
            annotations=[{
                "text": f"Error: {str(e)}",
                "showarrow": False,
                "x": 0.5,
                "y": 0.5,
                "font": {"size": 14, "color": "red"}
            }],
            height=400
        )


def metric_card(label: str, value: str, icon: str, color_scheme: str = "amber") -> rx.Component:
    """Display a metric card with icon."""
    return rx.box(
        rx.vstack(
            rx.icon(icon, size=32, color=rx.color(color_scheme, 9)),
            rx.heading(value, size="7", weight="bold"),
            rx.text(label, size="2", color="var(--gray-12)", text_align="center"),
            spacing="2",
            align="center"
        ),
        padding="1.5em",
        border="1px solid",
        border_color=rx.color("gray", 5),
        border_radius="var(--radius-4)",
        background_color=rx.color("gray", 1),
        width="100%",
        _hover={
            "border_color": rx.color(color_scheme, 6),
            "background_color": rx.color(color_scheme, 1),
            "transform": "translateY(-2px)",
            "box_shadow": "0 4px 12px rgba(0, 0, 0, 0.1)"
        },
        transition="all 0.2s ease"
    )


def insight_card(title: str, description: str, icon: str = "lightbulb") -> rx.Component:
    """Display an insight card with icon and description."""
    return rx.box(
        rx.hstack(
            rx.icon(icon, size=48, color=rx.color("blue", 9)),
            rx.vstack(
                rx.heading(title, size="4", weight="bold"),
                rx.text(description, size="3", color="var(--gray-12)", line_height="1.7"),
                spacing="2",
                align="start"
            ),
            spacing="3",
            align="start"
        ),
        padding="1.25em",
        background=rx.color("blue", 2),
        border_left=f"4px solid {rx.color('blue', 9)}",
        border_radius="var(--radius-3)",
        margin_bottom="1em"
    )


# ======================================================================
# MAIN PAGE SECTIONS
# ======================================================================

def executive_summary() -> rx.Component:
    """Executive summary with key findings."""
    return rx.vstack(
        rx.heading("Executive Summary", size="6", weight="bold", color_scheme="blue"),
        rx.box(
            rx.vstack(
                rx.heading("What We Discovered", size="5", weight="bold", margin_bottom="0.5em"),
                rx.unordered_list(
                    rx.list_item(
                        rx.text.strong("Strong Inflation Link: "),
                        "CPI explains 75% of gold price variance (r=0.87) – confirming gold's role as inflation hedge"
                    ),
                    rx.list_item(
                        rx.text.strong("Inverse Rate Relationship: "),
                        "Higher real interest rates lead to lower gold prices (r=-0.26) due to opportunity cost"
                    ),
                    rx.list_item(
                        rx.text.strong("Equity Market Surprise: "),
                        "Gold and S&P 500 move together (r=0.82), challenging the 'safe haven' narrative"
                    ),
                    rx.list_item(
                        rx.text.strong("VIX Paradox: "),
                        "Market volatility shows NO correlation with gold (r≈0.00) – unexpected finding!"
                    ),
                    spacing="2",
                    padding_left="1.5em"
                ),
                rx.text(
                    "In this chapter, we'll explore 17 features across 19.5 years (2006-2025) to understand "
                    "the economic forces that drive gold prices. Through correlation analysis, distribution studies, "
                    "and interactive visualizations, we'll identify the 13 most predictive features for our models.",
                    size="3",
                    color="var(--gray-12)",
                    margin_top="1em",
                    line_height="1.7"
                ),
                spacing="3",
                align="start"
            ),
            padding="1.5em",
            background=rx.color("blue", 2),
            border_left=f"4px solid {rx.color('blue', 9)}",
            border_radius="var(--radius-3)"
        ),
        spacing="3",
        align="start",
        width="100%",
        margin_bottom="2em"
    )


def data_collection_journey() -> rx.Component:
    """Explain how we found the common start date and aligned datasets."""
    return rx.vstack(
        rx.heading("Data Collection Journey: Finding Common Ground", size="7", weight="bold", margin_bottom="1em"),
        
        rx.text(
            "Imagine trying to combine 7 different historical records, each starting at different points in time. "
            "Our first challenge was finding when ALL these records overlap - the moment where every piece of data becomes available. "
            "We use Gold Spot Price as our foundation (measured monthly), and match all other indicators to its timeline. "
            "This ensures our analysis uses consistent monthly snapshots from start to finish.",
            size="4",
            color="var(--gray-12)",
            line_height="1.7",
            margin_bottom="1.5em"
        ),
        
        rx.box(
            rx.vstack(
                rx.heading("Data Availability by Source", size="5", weight="bold", margin_bottom="1em"),
                
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.table.column_header_cell("Dataset"),
                            rx.table.column_header_cell("Earliest Valid Date"),
                            rx.table.column_header_cell("Latest Date"),
                            rx.table.column_header_cell("Original Frequency"),
                        )
                    ),
                    rx.table.body(
                        rx.table.row(
                            rx.table.cell("Market Data (S&P 500, NASDAQ, etc.)"),
                            rx.table.cell("2004-11-18"),
                            rx.table.cell("2025-05-30"),
                            rx.table.cell(rx.badge("Daily", color_scheme="blue", size="2"))
                        ),
                        rx.table.row(
                            rx.table.cell(rx.text.strong("USD Index")),
                            rx.table.cell(rx.text.strong("2006-01-02", color=rx.color("red", 10))),
                            rx.table.cell("2025-05-30"),
                            rx.table.cell(rx.badge("Daily", color_scheme="blue", size="2"))
                        ),
                        rx.table.row(
                            rx.table.cell("Macroeconomic (CPI, Unemployment, M2)"),
                            rx.table.cell("2000-01-01"),
                            rx.table.cell("2025-05-01"),
                            rx.table.cell(rx.badge("Monthly", color_scheme="amber", size="2"))
                        ),
                        rx.table.row(
                            rx.table.cell("Real Interest Rate"),
                            rx.table.cell("2003-01-02"),
                            rx.table.cell("2025-05-01"),
                            rx.table.cell(rx.badge("Monthly", color_scheme="amber", size="2"))
                        ),
                        rx.table.row(
                            rx.table.cell("VIX (Volatility Index)"),
                            rx.table.cell("2000-01-03"),
                            rx.table.cell("2025-05-30"),
                            rx.table.cell(rx.badge("Daily", color_scheme="blue", size="2"))
                        ),
                        rx.table.row(
                            rx.table.cell("GPR (Geopolitical Risk)"),
                            rx.table.cell("1985-01-01"),
                            rx.table.cell("2025-05-01"),
                            rx.table.cell(rx.badge("Monthly", color_scheme="amber", size="2"))
                        ),
                        rx.table.row(
                            rx.table.cell("Gold Spot Price (WGC)"),
                            rx.table.cell("1978-01-31"),
                            rx.table.cell("2025-05-31"),
                            rx.table.cell(rx.badge("Monthly", color_scheme="amber", size="2"))
                        ),
                    ),
                    variant="surface",
                    size="3"
                ),
                
                rx.box(
                    rx.hstack(
                        rx.icon("calendar-days", size=36, color=rx.color("amber", 9)),
                        rx.vstack(
                            rx.text.strong("Common Start Date: January 31, 2006", size="4"),
                            rx.text(
                                "This is the first month-end after USD Index becomes available (Jan 2, 2006). "
                                "Since we join all data to Gold Spot Price (monthly), the timeline starts at the nearest month-end. "
                                "From this date forward, we have 19.5 years (233 months) of complete data across all features.",
                                size="3",
                                color="var(--gray-12)",
                                line_height="1.6"
                            ),
                            spacing="1",
                            align="start"
                        ),
                        spacing="3",
                        align="start"
                    ),
                    padding="1em",
                    background=rx.color("amber", 2),
                    border_radius="var(--radius-3)",
                    margin_top="1em"
                ),
                
                rx.box(
                    rx.vstack(
                        rx.heading("How We Combined the Data", size="5", weight="bold", margin_bottom="0.5em"),
                        rx.unordered_list(
                            rx.list_item(
                                rx.text.strong("Step 1: "),
                                "Find the 'starting line', when the last dataset begins (USD Index on January 2, 2006)"
                            ),
                            rx.list_item(
                                rx.text.strong("Step 2: "),
                                "Use Gold Spot Price as the anchor, it's measured monthly (end of each month)"
                            ),
                            rx.list_item(
                                rx.text.strong("Step 3: "),
                                "Match all other data to these monthly dates, even if they're originally tracked daily"
                            ),
                            rx.list_item(
                                rx.text.strong("Step 4: "),
                                "Fill any small gaps, ensuring we have complete information for every month"
                            ),
                            rx.list_item(
                                rx.text.strong("Step 5: "),
                                "Final result, one unified monthly timeline from 2006 to 2025"
                            ),
                            spacing="2",
                            padding_left="1.5em"
                        ),
                        spacing="2",
                        align="start"
                    ),
                    padding="1.25em",
                    background=rx.color("blue", 1),
                    border="1px solid",
                    border_color=rx.color("blue", 5),
                    border_radius="var(--radius-3)",
                    margin_top="1em"
                ),
                
                spacing="3",
                align="start"
            ),
            padding="1.5em",
            background=rx.color("gray", 1),
            border="1px solid",
            border_color=rx.color("gray", 5),
            border_radius="var(--radius-4)"
        ),
        
        spacing="3",
        align="start",
        width="100%",
        margin_bottom="2em"
    )


def dataset_overview() -> rx.Component:
    """Dataset overview section with metrics."""
    
    # Load combined_data.csv
    try:
        data_path = Path(__file__).parent.parent.parent / "data" / "combined_data.csv"
        df = pd.read_csv(data_path)
        
        # Get first 10 rows for preview
        df_preview = df.head(10)
        
        # Format numbers to 2 decimal places for better readability
        df_display = df_preview.copy()
        for col in df_display.columns:
            if df_display[col].dtype in ['float64', 'float32']:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        
        # Convert to lists for table display
        columns = df_display.columns.tolist()
        rows = df_display.values.tolist()
        
        data_loaded = True
        total_rows = len(df)
        total_cols = len(df.columns)
        
    except Exception as e:
        columns = ["Error"]
        rows = [[f"Could not load data: {str(e)}"]]
        data_loaded = False
        total_rows = 0
        total_cols = 0
    
    return rx.vstack(
        rx.heading("Dataset Overview", size="7", weight="bold", margin_bottom="1em"),
        
        rx.text(
            "After alignment and preprocessing, we have a unified monthly dataset spanning 19.5 years. "
            "All 17 features are synchronized to end-of-month dates",
            size="4",
            color="var(--gray-12)",
            line_height="1.7",
            margin_bottom="1.5em"
        ),
        
        rx.grid(
            metric_card("Total Features", str(total_cols) if data_loaded else "17", "database", "amber"),
            metric_card("Time Span", "19.5 Years", "calendar", "blue"),
            metric_card("Monthly Observations", str(total_rows) if data_loaded else "233", "bar-chart", "green"),
            columns="4",
            spacing="3",
            width="100%"
        ),
        
        # DataFrame Preview Section
        rx.cond(
            data_loaded,
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.icon("table-2", size=24, color=rx.color("purple", 9)),
                        rx.heading("Dataset Preview (First 10 Rows)", size="5", weight="bold"),
                        spacing="2",
                        align="center",
                        margin_bottom="0.5em"
                    ),
                    
                    rx.text(
                        f"Showing 10 of {total_rows} rows • {total_cols} columns",
                        size="2",
                        color="var(--gray-11)",
                        margin_bottom="1em"
                    ),
                    
                    rx.box(
                        rx.table.root(
                            rx.table.header(
                                rx.table.row(
                                    *[rx.table.column_header_cell(
                                        col,
                                        style={
                                            "font_weight": "bold",
                                            "background": rx.color("purple", 3),
                                            "white_space": "nowrap"
                                        }
                                    ) for col in columns]
                                )
                            ),
                            rx.table.body(
                                *[
                                    rx.table.row(
                                        *[rx.table.cell(
                                            str(cell) if cell != "" else "-",
                                            style={"white_space": "nowrap"}
                                        ) for cell in row]
                                    )
                                    for row in rows
                                ]
                            ),
                            variant="surface",
                            size="2"
                        ),
                        overflow_x="auto",
                        width="100%",
                        border="1px solid",
                        border_color=rx.color("gray", 4),
                        border_radius="var(--radius-3)"
                    ),
                    
                    spacing="3",
                    align="start",
                    width="100%"
                ),
                padding="1.5em",
                background=rx.color("purple", 1),
                border="1px solid",
                border_color=rx.color("purple", 5),
                border_radius="var(--radius-4)",
                margin_top="1.5em",
                width="100%"
            ),
            rx.box()  # Empty box if data not loaded
        ),
        
        spacing="3",
        align="start",
        width="100%",
        margin_bottom="2em"
    )


def gold_spot_currency_analysis() -> rx.Component:
    """Analyze Gold Spot across different currencies and explain USD choice."""
    content_vstack = rx.vstack(
        rx.heading("Why USD for Gold Spot? Understanding the Target Variable", size="7", weight="bold", margin_bottom="1em"),
        
        rx.text(
            "Gold prices are quoted in multiple currencies globally (USD, EUR, GBP, JPY, CHF, etc.). "
            "Choosing the right currency for our target variable is crucial. We analyzed correlations "
            "across all available currencies and selected USD for three compelling reasons:",
            size="4",
            color="var(--gray-12)",
            line_height="1.7",
            margin_bottom="1.5em"
        ),
        
        rx.grid(
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.icon("trending-up", size=24, color=rx.color("green", 9)),
                        rx.heading("Near-Perfect Correlation", size="4", weight="bold"),
                        spacing="2",
                        align="center"
                    ),
                    rx.text(
                        "Correlation matrix shows r > 0.99 between USD, EUR, GBP gold prices. "
                        "The differences are purely exchange rate scaling, the underlying gold movement is identical. "
                        "Some currencies show slightly lower correlations (0.74-0.79) due to shorter historical data.",
                        size="3",
                        color="var(--gray-12)",
                        line_height="1.6"
                    ),
                    spacing="2",
                    align="start"
                ),
                padding="1.25em",
                background=rx.color("green", 1),
                border="1px solid",
                border_color=rx.color("green", 5),
                border_radius="var(--radius-3)"
            ),
            
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.icon("globe", size=24, color=rx.color("blue", 9)),
                        rx.heading("Global Standard", size="4", weight="bold"),
                        spacing="2",
                        align="center"
                    ),
                    rx.text(
                        "USD is the international pricing standard. Major gold exchanges (COMEX, LBMA) quote in USD/oz. "
                        "Most economic indicators (CPI, Fed rates, S&P 500) are USD-based, ensuring consistency. "
                        "Using USD eliminates currency conversion noise in our analysis.",
                        size="3",
                        color="var(--gray-12)",
                        line_height="1.6"
                    ),
                    spacing="2",
                    align="start"
                ),
                padding="1.25em",
                background=rx.color("blue", 1),
                border="1px solid",
                border_color=rx.color("blue", 5),
                border_radius="var(--radius-3)"
            ),
            
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.icon("database", size=24, color=rx.color("purple", 9)),
                        rx.heading("Most Complete Data", size="4", weight="bold"),
                        spacing="2",
                        align="center"
                    ),
                    rx.text(
                        "USD gold prices have the longest uninterrupted time series (1978-2025 from WGC). "
                        "Other currencies have gaps or shorter histories. Gold_Spot_USD provides 47 years of data, "
                        "though we only use 19.5 years (2006-2025) due to USD Index limitations.",
                        size="3",
                        color="var(--gray-12)",
                        line_height="1.6"
                    ),
                    spacing="2",
                    align="start"
                ),
                padding="1.25em",
                background=rx.color("purple", 1),
                border="1px solid",
                border_color=rx.color("purple", 5),
                border_radius="var(--radius-3)"
            ),
            
            columns="3",
            spacing="3",
            width="100%",
            margin_bottom="1.5em"
        ),
        
        rx.box(
            rx.vstack(
                rx.heading("Gold Spot vs Gold Futures vs Gold ETF", size="5", weight="bold", margin_bottom="1em"),
                
                rx.text(
                    "Beyond currency selection, we also chose between three gold price representations. "
                    "All three have correlation near 1.0, but differ in what they represent:",
                    size="3",
                    color="var(--gray-12)",
                    margin_bottom="1em",
                    line_height="1.6"
                ),
                
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.table.column_header_cell("Product"),
                            rx.table.column_header_cell("What It Represents"),
                            rx.table.column_header_cell("Market Characteristics"),
                            rx.table.column_header_cell("Decision"),
                        )
                    ),
                    rx.table.body(
                        rx.table.row(
                            rx.table.cell(rx.text.strong("Gold Spot", color=rx.color("amber", 10))),
                            rx.table.cell("Physical gold price for immediate delivery on OTC markets"),
                            rx.table.cell("Reflects real supply/demand of physical gold; less speculative"),
                            rx.table.cell(rx.badge("SELECTED", color_scheme="green", size="2")),
                            style={"background": rx.color("green", 2), "font_weight": "bold"}
                        ),
                        rx.table.row(
                            rx.table.cell(rx.text.strong("Gold Futures")),
                            rx.table.cell("Contracts for future delivery on COMEX exchange"),
                            rx.table.cell("High liquidity, many speculators; more volatile short-term"),
                            rx.table.cell(rx.badge("Rejected", color_scheme="gray", size="2"))
                        ),
                        rx.table.row(
                            rx.table.cell(rx.text.strong("Gold ETF (GLD, IAU)")),
                            rx.table.cell("Exchange-traded fund tracking gold; traded like stocks"),
                            rx.table.cell("Includes fund management fees; affected by equity flows"),
                            rx.table.cell(rx.badge("Rejected", color_scheme="gray", size="2"))
                        ),
                    ),
                    variant="surface",
                    size="3"
                ),
                
                rx.box(
                    rx.vstack(
                        rx.text.strong("Why Gold Spot Selected:", size="4"),
                        rx.unordered_list(
                            rx.list_item("Represents actual physical gold value without speculative noise"),
                            rx.list_item("Long-term stability better reflects gold's role as store of value"),
                            rx.list_item("No contract expiration dates or rollover costs (unlike futures)"),
                            rx.list_item("No management fees or tracking errors (unlike ETFs)"),
                            rx.list_item("Best proxy for 'true' gold price in economic analyses"),
                            spacing="1",
                            padding_left="1.5em"
                        ),
                        spacing="2",
                        align="start"
                    ),
                    padding="1em",
                    background=rx.color("amber", 2),
                    border_radius="var(--radius-3)",
                    margin_top="1em"
                ),
                
                spacing="3",
                align="start"
            ),
            padding="1.5em",
            background=rx.color("gray", 1),
            border="1px solid",
            border_color=rx.color("gray", 5),
            border_radius="var(--radius-4)"
        ),
        
        spacing="3",
        align="start",
        width="100%",
        margin_bottom="1em"
    )
    # Chart placed outside vstack for better centering
    chart_box = rx.box(
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.icon("bar-chart-3", size=20, color=rx.color("amber", 9)),
                    rx.heading("Gold Price Correlation Heatmap (All Currencies & Products)", size="5", weight="bold"),
                    spacing="2",
                    align="center"
                ),
                rx.text(
                    "Heatmap showing r > 0.99 between USD, EUR, GBP gold spot prices, and high correlation with futures/ETFs",
                    size="3",
                    color="var(--gray-12)"
                ),
                rx.plotly(data=load_plotly_chart("gold_currency_heatmap"), width="1000px"),
                spacing="3",
                align="start",
                width="100%"
            ),
            padding="1.5em",
            border="1px solid",
            border_color=rx.color("gray", 5),
            border_radius="var(--radius-4)",
            background_color=rx.color("gray", 1),
        ),
        display="flex",
        justify_content="center",
        width="100%",
        margin_bottom="2em"
    )
    
    return rx.vstack(
        content_vstack,
        chart_box,
        spacing="0",
        align="start",
        width="100%",
        margin_bottom="2em"
    )


def target_variable_section() -> rx.Component:
    """Gold Spot price analysis section."""
    content_vstack = rx.vstack(
        rx.heading("Target Variable: Gold Spot Price", size="7", weight="bold", margin_bottom="1em"),
        
        rx.text(
            "Gold spot price represents the current market price for immediate physical delivery. "
            "As our target variable, this time series (2006-2025) reveals long-term trends and volatility patterns. "
            "Gold grew from ~$600/oz in 2006 to over $2,700/oz in 2024, driven by financial crises, inflation fears, and geopolitical tensions.",
            size="4",
            color="var(--gray-12)",
            line_height="1.7",
            margin_bottom="1.5em"
        ),
        
        spacing="3",
        align="start",
        width="100%",
        margin_bottom="1em"
    )
    
    # Chart placed outside vstack for better centering
    chart_box = rx.box(
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.icon("trending-up", size=20, color=rx.color("amber", 9)),
                    rx.heading("Gold Spot Price Historical Trend (2006-2025)", size="5", weight="bold"),
                    spacing="2",
                    align="center"
                ),
                rx.text(
                    "Interactive time series showing gold price evolution with major economic events",
                    size="3",
                    color="var(--gray-12)"
                ),
                rx.plotly(data=load_plotly_chart("gold_price_timeseries"), width="1200px"),
                spacing="3",
                align="start",
                width="100%"
            ),
            padding="1.5em",
            border="1px solid",
            border_color=rx.color("gray", 5),
            border_radius="var(--radius-4)",
            background_color=rx.color("gray", 1),
        ),
        display="flex",
        justify_content="center",
        width="100%",
        margin_bottom="2em"
    )
    
    # Additional content after chart
    additional_content = rx.vstack(
        rx.grid(
            insight_card(
                "2008 Financial Crisis",
                "Gold surged from $800 to $1,900 as investors fled to safe havens during the Great Recession"
            ),
            insight_card(
                "COVID-19 Pandemic (2020)",
                "Gold hit record high of $2,067 amid unprecedented monetary stimulus and economic uncertainty"
            ),
            insight_card(
                "2024 All-Time High",
                "Gold broke $2,700/oz driven by inflation fears, Fed rate cuts, and geopolitical tensions"
            ),
            columns="3",
            spacing="3",
            width="100%"
        ),
        
        spacing="3",
        align="start",
        width="100%"
    )
    
    return rx.vstack(
        content_vstack,
        chart_box,
        additional_content,
        spacing="0",
        align="start",
        width="100%",
        margin_bottom="2em"
    )


def distribution_analysis_section() -> rx.Component:
    """Feature distribution analysis with tabbed interface"""
    content_vstack = rx.vstack(
        rx.heading("Feature Distributions", size="7", weight="bold", margin_bottom="1em"),
        
        rx.text(
            "Understanding value distributions helps detect outliers, skewness, and identify potential transformations needed for modeling. "
            "Each tab shows boxplots (for quartiles and outliers) alongside histograms (for distribution shape) of different feature groups.",
            size="4",
            color="var(--gray-12)",
            line_height="1.7",
            margin_bottom="1.5em"
        ),
        
        spacing="3",
        align="start",
        width="100%",
        margin_bottom="1em"
    )
    
    # Chart box placed outside vstack for better centering
    chart_box = rx.box(
        rx.box(
            rx.tabs.root(
                rx.tabs.list(
                    rx.tabs.trigger("Target Variable", value="target"),
                    rx.tabs.trigger("Market Indicators", value="market"),
                    rx.tabs.trigger("Macroeconomic", value="macro"),
                    rx.tabs.trigger("Volatility & Risk", value="volatility"),
                ),
                
                rx.tabs.content(
                    rx.vstack(
                        rx.text(
                            "Gold Spot price distribution shows right skewness with increasing trend over time. "
                            "The boxplot reveals several outliers during crisis periods (2008, 2020, 2024).",
                            size="3",
                            color="var(--gray-12)",
                            margin_bottom="1em"
                        ),
                        rx.plotly(data=load_plotly_chart("gold_distributions"), width="1200px"),
                        spacing="3",
                        align="start",
                        width="100%"
                    ),
                    value="target"
                ),
                
                rx.tabs.content(
                    rx.vstack(
                        rx.text(
                            "Market indicators (S&P 500, USD Index, Silver, Crude Oil) show varying distribution patterns. "
                            "Stock indices show strong upward trends, while commodities exhibit higher volatility.",
                            size="3",
                            color="var(--gray-12)",
                            margin_bottom="1em"
                        ),
                        rx.plotly(data=load_plotly_chart("market_distributions"), width="1200px"),
                        spacing="3",
                        align="start",
                        width="100%"
                    ),
                    value="market"
                ),
                
                rx.tabs.content(
                    rx.vstack(
                        rx.text(
                            "Macroeconomic variables (CPI, Unemployment, Interest Rates) reflect major policy shifts. "
                            "CPI shows steady inflation growth, while rates fluctuated dramatically during QE periods.",
                            size="3",
                            color="var(--gray-12)",
                            margin_bottom="1em"
                        ),
                        rx.plotly(data=load_plotly_chart("macro_distributions"), width="1200px"),
                        spacing="3",
                        align="start",
                        width="100%"
                    ),
                    value="macro"
                ),
                
                rx.tabs.content(
                    rx.vstack(
                        rx.text(
                            "Volatility indicators (VIX, GPR, GPRA) capture market fear and geopolitical tensions. "
                            "VIX spikes during crises (2008, 2020), while GPR shows elevated levels during conflicts.",
                            size="3",
                            color="var(--gray-12)",
                            margin_bottom="1em"
                        ),
                        rx.plotly(data=load_plotly_chart("volatility_distributions"), width="1000px"),
                        spacing="3",
                        align="start",
                        width="100%"
                    ),
                    value="volatility"
                ),
                
                default_value="target",
                width="100%"
            ),
            padding="1.5em",
            background=rx.color("gray", 1),
            border="1px solid",
            border_color=rx.color("gray", 5),
            border_radius="var(--radius-4)"
        ),
        display="flex",
        justify_content="center",
        width="100%",
        margin_bottom="2em"
    )
    
    return rx.vstack(
        content_vstack,
        chart_box,
        spacing="0",
        align="start",
        width="100%",
        margin_bottom="2em"
    )


def correlation_analysis_section() -> rx.Component:
    """Correlation analysis section."""
    content_vstack = rx.vstack(
        rx.heading("Correlation & Relationships", size="7", weight="bold", margin_bottom="1em"),
        
        rx.text(
            "Understanding feature relationships reveals which variables move together and helps identify multicollinearity. "
            "This interactive heatmap shows correlations between all 17 features from 2006-2025. "
            "Red cells indicate strong positive correlations (variables rising together), while blue shows negative correlations (inverse relationships).",
            size="4",
            color="var(--gray-12)",
            line_height="1.7",
            margin_bottom="1.5em"
        ),
        
        spacing="3",
        align="start",
        width="100%",
        margin_bottom="1em"
    )
    
    # Chart placed outside vstack for better centering
    chart_box = rx.box(
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.icon("bar-chart-3", size=20, color=rx.color("blue", 9)),
                    rx.heading("Full Feature Correlation Heatmap", size="5", weight="bold"),
                    spacing="2",
                    align="center"
                ),
                rx.text(
                    "Interactive correlation matrix showing relationships between all 17 features in our dataset",
                    size="3",
                    color="var(--gray-12)"
                ),
                rx.plotly(data=load_plotly_chart("correlation_heatmap"), width="1000px"),
                spacing="3",
                align="start",
                width="100%"
            ),
            padding="1.5em",
            border="1px solid",
            border_color=rx.color("gray", 5),
            border_radius="var(--radius-4)",
            background_color=rx.color("gray", 1),
        ),
        display="flex",
        justify_content="center",
        width="100%",
        margin_bottom="2em"
    )
    
    # Additional content after chart
    additional_content = rx.vstack(
        rx.box(
            rx.vstack(
                rx.heading("Top Correlations with Gold Spot", size="5", weight="bold", margin_bottom="1em"),
                
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.table.column_header_cell("Feature"),
                            rx.table.column_header_cell("Correlation"),
                            rx.table.column_header_cell("Interpretation"),
                        )
                    ),
                    rx.table.body(
                        rx.table.row(
                            rx.table.cell(rx.text.strong("CPI (Inflation)")),
                            rx.table.cell(rx.badge("+0.85", color_scheme="green", size="2")),
                            rx.table.cell("Strong positive – inflation drives gold demand")
                        ),
                        rx.table.row(
                            rx.table.cell(rx.text.strong("M2 Money Supply")),
                            rx.table.cell(rx.badge("+0.82", color_scheme="green", size="2")),
                            rx.table.cell("Strong positive – monetary expansion boosts gold")
                        ),
                        rx.table.row(
                            rx.table.cell(rx.text.strong("S&P 500")),
                            rx.table.cell(rx.badge("+0.80", color_scheme="green", size="2")),
                            rx.table.cell("Surprising! Both rise in liquidity-driven markets")
                        ),
                        rx.table.row(
                            rx.table.cell(rx.text.strong("Silver Futures")),
                            rx.table.cell(rx.badge("+0.71", color_scheme="green", size="2")),
                            rx.table.cell("Strong positive – precious metals move together")
                        ),
                        rx.table.row(
                            rx.table.cell(rx.text.strong("Real Interest Rate")),
                            rx.table.cell(rx.badge("-0.40", color_scheme="red", size="2")),
                            rx.table.cell("Negative – higher rates reduce gold appeal")
                        ),
                        rx.table.row(
                            rx.table.cell(rx.text.strong("Treasury Yield 10Y")),
                            rx.table.cell(rx.badge("-0.34", color_scheme="red", size="2")),
                            rx.table.cell("Negative – bonds compete with gold")
                        ),
                        rx.table.row(
                            rx.table.cell(rx.text.strong("VIX (Volatility)")),
                            rx.table.cell(rx.badge("0.00", color_scheme="gray", size="2")),
                            rx.table.cell("No correlation – unexpected finding!")
                        ),
                    ),
                    variant="surface",
                    size="3"
                ),
                
                spacing="3",
                align="start"
            ),
            padding="1.5em",
            background=rx.color("gray", 1),
            border="1px solid",
            border_color=rx.color("gray", 5),
            border_radius="var(--radius-4)",
            margin_bottom="1.5em"
        ),
        
        spacing="3",
        align="start",
        width="100%"
    )
    
    # 3D Network Graph section
    network_3d_content = rx.vstack(
        rx.heading("3D Feature Correlation Network", size="7", weight="bold", margin_bottom="1em"),
        
        rx.text(
            "Before deciding which features to keep, we need to understand how all 17 original features relate to each other. "
            "This network visualization maps correlations in 3D space—features that move together cluster closely, "
            "while independent features stand apart. Lines connect features with correlation above 0.3, "
            "revealing natural groupings and redundancies.",
            size="4",
            color="var(--gray-12)",
            line_height="1.7",
            margin_bottom="1.5em"
        ),
        
        rx.box(
            rx.vstack(
                rx.heading("Reading the Network", size="5", weight="bold", margin_bottom="1em"),
                
                rx.accordion.root(
                    rx.accordion.item(
                        header="Node Size = Volatility",
                        content=rx.text(
                            "Larger circles indicate features with higher variance, meaning they changed more dramatically "
                            "over our 19.5-year period. Gold prices, stock indices, and crude oil typically show larger nodes "
                            "due to their significant price swings during economic crises and booms.",
                            size="3",
                            line_height="1.7",
                            color="var(--gray-12)"
                        )
                    ),
                    rx.accordion.item(
                        header="Position = Similarity",
                        content=rx.text(
                            "The algorithm pulls correlated features together like magnets. "
                            "Tight clusters mean these features tell similar stories. For example, CPI and M2 Money Supply "
                            "sit close together because both reflect inflationary pressures. We only need one of them.",
                            size="3",
                            line_height="1.7",
                            color="var(--gray-12)"
                        )
                    ),
                    rx.accordion.item(
                        header="Green Lines = Moving Together",
                        content=rx.text(
                            "Green connections show positive correlations—when one feature rises, so does the other. "
                            "Thicker lines mean stronger relationships. A thick green line between S&P 500 and NASDAQ "
                            "(r = 0.99) visually confirms they're nearly identical.",
                            size="3",
                            line_height="1.7",
                            color="var(--gray-12)"
                        )
                    ),
                    rx.accordion.item(
                        header="Red Lines = Opposing Forces",
                        content=rx.text(
                            "Red connections indicate inverse relationships. When real interest rates rise, gold prices tend to fall "
                            "(opportunity cost). This economic principle appears as red lines between interest rate features and gold.",
                            size="3",
                            line_height="1.7",
                            color="var(--gray-12)"
                        )
                    ),
                    collapsible=True,
                    variant="soft",
                    width="100%",
                    margin_bottom="1.5em"
                ),
                
                rx.box(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("lightbulb", size=28, color=rx.color("amber", 10)),
                            rx.heading("What This Tells Us", size="5", weight="bold", color=rx.color("amber", 11)),
                            spacing="3",
                            align="center",
                            margin_bottom="0.75em"
                        ),
                        rx.text(
                            "By examining this network, we identified ",
                            rx.text.strong("4 features to remove", color=rx.color("red", 10), size="4"),
                            " due to redundancy: ",
                            rx.text.strong("NASDAQ ", color=rx.color("blue", 10)),
                            "(too similar to S&P 500), ",
                            rx.text.strong("M2 Supply ", color=rx.color("blue", 10)),
                            "(duplicates CPI information), ",
                            rx.text.strong("GPRT ", color=rx.color("blue", 10)),
                            "(subset of GPR coverage), and ",
                            rx.text.strong("Gold Futures ", color=rx.color("blue", 10)),
                            "(nearly identical to Gold Spot). "
                            "The network also reveals ",
                            rx.text.strong("VIX as an isolated node", color=rx.color("green", 10)),
                            "—it measures market fear independently of other indicators, "
                            "making it worth keeping despite low gold correlation.",
                            size="3",
                            color="var(--gray-12)",
                            line_height="1.8"
                        ),
                        spacing="3",
                        align="start"
                    ),
                    padding="1.5em",
                    background=rx.color("amber", 2),
                    border=f"2px solid {rx.color('amber', 7)}",
                    border_radius="var(--radius-4)",
                    margin_top="1em",
                    box_shadow="0 2px 8px rgba(0, 0, 0, 0.08)"
                ),
                
                spacing="3",
                align="start"
            ),
            padding="1.5em",
            background=rx.color("gray", 1),
            border="1px solid",
            border_color=rx.color("gray", 5),
            border_radius="var(--radius-4)",
            margin_bottom="1.5em"
        ),
        
        spacing="3",
        align="start",
        width="100%",
        margin_bottom="1em"
    )


def feature_selection_section() -> rx.Component:
    """Feature selection and multicollinearity section."""
    return rx.vstack(
        rx.heading("Feature Selection: From 17 to 13", size="7", weight="bold", margin_bottom="1em"),
        
        rx.text(
            "While having many features sounds good, highly correlated features (multicollinearity) "
            "can confuse machine learning models and reduce prediction accuracy. We identified and removed "
            "4 redundant features, keeping only the most informative ones.",
            size="4",
            color="var(--gray-12)",
            line_height="1.7",
            margin_bottom="1.5em"
        ),
        
        rx.box(
            rx.vstack(
                rx.heading("Multicollinearity Detection", size="5", weight="bold", margin_bottom="1em"),
                
                rx.text(
                    "Features with correlation > 0.90 are considered highly redundant. "
                    "We used correlation analysis and domain knowledge to decide which features to keep:",
                    size="3",
                    color="var(--gray-12)",
                    margin_bottom="1em"
                ),
                
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.table.column_header_cell("Feature Pair"),
                            rx.table.column_header_cell("Correlation"),
                            rx.table.column_header_cell("Decision"),
                            rx.table.column_header_cell("Reason"),
                        )
                    ),
                    rx.table.body(
                        rx.table.row(
                            rx.table.cell("Gold Spot ↔ Gold Futures ↔ Gold ETF"),
                            rx.table.cell(rx.badge("0.98+", color_scheme="red", size="2")),
                            rx.table.cell(rx.text("Keep ", rx.text.strong("Gold Spot"), style={"color": "green"})),
                            rx.table.cell("Ground truth physical price, monthly aligned, less speculative")
                        ),
                        rx.table.row(
                            rx.table.cell("S&P 500 ↔ NASDAQ"),
                            rx.table.cell(rx.badge("0.99", color_scheme="red", size="2")),
                            rx.table.cell(rx.text("Keep ", rx.text.strong("S&P 500"), style={"color": "green"})),
                            rx.table.cell("Broader market representation, more stable than tech-heavy NASDAQ")
                        ),
                        rx.table.row(
                            rx.table.cell("CPI ↔ M2 Supply"),
                            rx.table.cell(rx.badge("0.96", color_scheme="red", size="2")),
                            rx.table.cell(rx.text("Keep ", rx.text.strong("CPI"), style={"color": "green"})),
                            rx.table.cell("Direct inflation indicator vs indirect money supply measure")
                        ),
                        rx.table.row(
                            rx.table.cell("GPR ↔ GPRT"),
                            rx.table.cell(rx.badge("0.91", color_scheme="red", size="2")),
                            rx.table.cell(rx.text("Keep ", rx.text.strong("GPR"), style={"color": "green"})),
                            rx.table.cell("Broader geopolitical coverage (wars, instability) vs terrorism-only")
                        ),
                    ),
                    variant="surface",
                    size="3"
                ),
                
                rx.box(
                    rx.vstack(
                        rx.heading("Detailed Rationale", size="4", weight="bold", margin_bottom="1em"),
                        
                        rx.accordion.root(
                            rx.accordion.item(
                                header="Gold Spot vs Gold Futures vs Gold ETF",
                                content=rx.box(
                                    rx.unordered_list(
                                        rx.list_item(rx.text.strong("Gold Spot: "), "Represents actual physical price of gold in global markets"),
                                        rx.list_item(rx.text.strong("Gold Futures: "), "Subject to speculative trading behavior and contract rollover costs"),
                                        rx.list_item(rx.text.strong("Gold ETF: "), "Includes fund management fees and tracking errors"),
                                        rx.list_item(rx.text.strong("Selected: "), "Gold Spot aligns with monthly macroeconomic indicators and reflects true market value"),
                                        spacing="2",
                                        padding_left="1.5em"
                                    ),
                                    width="100%"
                                )
                            ),
                            rx.accordion.item(
                                header="S&P 500 vs NASDAQ",
                                content=rx.box(
                                    rx.unordered_list(
                                        rx.list_item(rx.text.strong("S&P 500: "), "Represents overall U.S. economy across 500 companies"),
                                        rx.list_item(rx.text.strong("NASDAQ: "), "Tech-heavy index, more volatile and sector-concentrated"),
                                        rx.list_item(rx.text.strong("Selected: "), "S&P 500 provides broader, more stable market signal"),
                                        spacing="2",
                                        padding_left="1.5em"
                                    ),
                                    width="100%"
                                )
                            ),
                            rx.accordion.item(
                                header="CPI vs M2 Money Supply",
                                content=rx.box(
                                    rx.unordered_list(
                                        rx.list_item(rx.text.strong("CPI: "), "Consumer Price Index directly measures inflation"),
                                        rx.list_item(rx.text.strong("M2: "), "Money supply reflects liquidity, indirectly affects inflation expectations"),
                                        rx.list_item(rx.text.strong("Selected: "), "CPI is the direct, primary inflation indicator"),
                                        spacing="2",
                                        padding_left="1.5em"
                                    ),
                                    width="100%"
                                )
                            ),
                            rx.accordion.item(
                                header="GPR vs GPRT",
                                content=rx.box(
                                    rx.unordered_list(
                                        rx.list_item(rx.text.strong("GPR: "), "Captures general geopolitical tensions (wars, political instability)"),
                                        rx.list_item(rx.text.strong("GPRT: "), "Focuses only on terrorism-related geopolitical risks"),
                                        rx.list_item(rx.text.strong("Selected: "), "GPR provides broader coverage of global uncertainty"),
                                        spacing="2",
                                        padding_left="1.5em"
                                    ),
                                    width="100%"
                                )
                            ),
                            collapsible=True,
                            variant="soft",
                            width="100%"
                        ),
                        
                        spacing="3",
                        align="start",
                        width="100%"
                    ),
                    padding="1em",
                    background=rx.color("blue", 1),
                    border="1px solid",
                    border_color=rx.color("blue", 5),
                    border_radius="var(--radius-3)",
                    margin_top="1em",
                    width="100%"
                ),
                spacing="3",
                align="start"
            ),
            size="3",
            padding="1.5em",
            background=rx.color("amber", 2),
            border_left=f"4px solid {rx.color('amber', 9)}",
            border_radius="var(--radius-3)",
            margin_bottom="1.5em"
        ),
        
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.icon("circle-check", size=28, color=rx.color("green", 10)),
                    rx.heading("Final 13 Features Selected", size="5", weight="bold", color=rx.color("green", 11)),
                    spacing="3",
                    align="center",
                    margin_bottom="1em"
                ),
                
                rx.grid(
                    # Target Variable
                    rx.box(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("target", size=20, color=rx.color("amber", 10)),
                                rx.heading("Target Variable", size="4", weight="bold", color=rx.color("amber", 11)),
                                spacing="2",
                                align="center"
                            ),
                            rx.badge("1 feature", color_scheme="amber", size="1"),
                            rx.divider(margin_y="0.75em"),
                            rx.vstack(
                                rx.box(
                                    rx.hstack(
                                        rx.icon("circle-dot", size=16, color=rx.color("amber", 9)),
                                        rx.text("Gold_Spot", size="3", weight="medium"),
                                        spacing="2",
                                        align="center"
                                    ),
                                    padding="0.5em",
                                    background=rx.color("amber", 3),
                                    border_radius="var(--radius-2)",
                                    width="100%"
                                ),
                                spacing="2",
                                align="start",
                                width="100%"
                            ),
                            spacing="3",
                            align="start"
                        ),
                        padding="1.25em",
                        background=rx.color("amber", 1),
                        border="2px solid",
                        border_color=rx.color("amber", 6),
                        border_radius="var(--radius-4)",
                        box_shadow="0 2px 8px rgba(0, 0, 0, 0.05)",
                        _hover={
                            "border_color": rx.color("amber", 7),
                            "box_shadow": "0 4px 12px rgba(0, 0, 0, 0.1)"
                        },
                        transition="all 0.2s ease"
                    ),
                    
                    # Precious Metals
                    rx.box(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("gem", size=20, color=rx.color("purple", 10)),
                                rx.heading("Precious Metals", size="4", weight="bold", color=rx.color("purple", 11)),
                                spacing="2",
                                align="center"
                            ),
                            rx.badge("1 feature", color_scheme="purple", size="1"),
                            rx.divider(margin_y="0.75em"),
                            rx.vstack(
                                rx.box(
                                    rx.hstack(
                                        rx.icon("circle-dot", size=16, color=rx.color("purple", 9)),
                                        rx.text("Silver_Futures", size="3", weight="medium"),
                                        spacing="2",
                                        align="center"
                                    ),
                                    padding="0.5em",
                                    background=rx.color("purple", 3),
                                    border_radius="var(--radius-2)",
                                    width="100%"
                                ),
                                spacing="2",
                                align="start",
                                width="100%"
                            ),
                            spacing="3",
                            align="start"
                        ),
                        padding="1.25em",
                        background=rx.color("purple", 1),
                        border="2px solid",
                        border_color=rx.color("purple", 6),
                        border_radius="var(--radius-4)",
                        box_shadow="0 2px 8px rgba(0, 0, 0, 0.05)",
                        _hover={
                            "border_color": rx.color("purple", 7),
                            "box_shadow": "0 4px 12px rgba(0, 0, 0, 0.1)"
                        },
                        transition="all 0.2s ease"
                    ),
                    
                    # Financial Markets
                    rx.box(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("trending-up", size=20, color=rx.color("blue", 10)),
                                rx.heading("Financial Markets", size="4", weight="bold", color=rx.color("blue", 11)),
                                spacing="2",
                                align="center"
                            ),
                            rx.badge("4 features", color_scheme="blue", size="1"),
                            rx.divider(margin_y="0.75em"),
                            rx.vstack(
                                rx.box(
                                    rx.hstack(
                                        rx.icon("circle-dot", size=16, color=rx.color("blue", 9)),
                                        rx.text("S&P_500", size="3", weight="medium"),
                                        spacing="2",
                                        align="center"
                                    ),
                                    padding="0.5em",
                                    background=rx.color("blue", 3),
                                    border_radius="var(--radius-2)",
                                    width="100%"
                                ),
                                rx.box(
                                    rx.hstack(
                                        rx.icon("circle-dot", size=16, color=rx.color("blue", 9)),
                                        rx.text("^VIX", size="3", weight="medium"),
                                        spacing="2",
                                        align="center"
                                    ),
                                    padding="0.5em",
                                    background=rx.color("blue", 3),
                                    border_radius="var(--radius-2)",
                                    width="100%"
                                ),
                                rx.box(
                                    rx.hstack(
                                        rx.icon("circle-dot", size=16, color=rx.color("blue", 9)),
                                        rx.text("Crude_Oil", size="3", weight="medium"),
                                        spacing="2",
                                        align="center"
                                    ),
                                    padding="0.5em",
                                    background=rx.color("blue", 3),
                                    border_radius="var(--radius-2)",
                                    width="100%"
                                ),
                                rx.box(
                                    rx.hstack(
                                        rx.icon("circle-dot", size=16, color=rx.color("blue", 9)),
                                        rx.text("USD_Index", size="3", weight="medium"),
                                        spacing="2",
                                        align="center"
                                    ),
                                    padding="0.5em",
                                    background=rx.color("blue", 3),
                                    border_radius="var(--radius-2)",
                                    width="100%"
                                ),
                                spacing="2",
                                align="start",
                                width="100%"
                            ),
                            spacing="3",
                            align="start"
                        ),
                        padding="1.25em",
                        background=rx.color("blue", 1),
                        border="2px solid",
                        border_color=rx.color("blue", 6),
                        border_radius="var(--radius-4)",
                        box_shadow="0 2px 8px rgba(0, 0, 0, 0.05)",
                        _hover={
                            "border_color": rx.color("blue", 7),
                            "box_shadow": "0 4px 12px rgba(0, 0, 0, 0.1)"
                        },
                        transition="all 0.2s ease"
                    ),
                    
                    # Macroeconomic
                    rx.box(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("bar-chart-2", size=20, color=rx.color("green", 10)),
                                rx.heading("Macroeconomic", size="4", weight="bold", color=rx.color("green", 11)),
                                spacing="2",
                                align="center"
                            ),
                            rx.badge("5 features", color_scheme="green", size="1"),
                            rx.divider(margin_y="0.75em"),
                            rx.vstack(
                                rx.box(
                                    rx.hstack(
                                        rx.icon("circle-dot", size=16, color=rx.color("green", 9)),
                                        rx.text("CPI", size="3", weight="medium"),
                                        spacing="2",
                                        align="center"
                                    ),
                                    padding="0.5em",
                                    background=rx.color("green", 3),
                                    border_radius="var(--radius-2)",
                                    width="100%"
                                ),
                                rx.box(
                                    rx.hstack(
                                        rx.icon("circle-dot", size=16, color=rx.color("green", 9)),
                                        rx.text("Unemployment", size="3", weight="medium"),
                                        spacing="2",
                                        align="center"
                                    ),
                                    padding="0.5em",
                                    background=rx.color("green", 3),
                                    border_radius="var(--radius-2)",
                                    width="100%"
                                ),
                                rx.box(
                                    rx.hstack(
                                        rx.icon("circle-dot", size=16, color=rx.color("green", 9)),
                                        rx.text("Treasury_Yield_10Y", size="3", weight="medium"),
                                        spacing="2",
                                        align="center"
                                    ),
                                    padding="0.5em",
                                    background=rx.color("green", 3),
                                    border_radius="var(--radius-2)",
                                    width="100%"
                                ),
                                rx.box(
                                    rx.hstack(
                                        rx.icon("circle-dot", size=16, color=rx.color("green", 9)),
                                        rx.text("Real_Interest_Rate", size="3", weight="medium"),
                                        spacing="2",
                                        align="center"
                                    ),
                                    padding="0.5em",
                                    background=rx.color("green", 3),
                                    border_radius="var(--radius-2)",
                                    width="100%"
                                ),
                                rx.box(
                                    rx.hstack(
                                        rx.icon("circle-dot", size=16, color=rx.color("green", 9)),
                                        rx.text("Fed_Funds_Rate", size="3", weight="medium"),
                                        spacing="2",
                                        align="center"
                                    ),
                                    padding="0.5em",
                                    background=rx.color("green", 3),
                                    border_radius="var(--radius-2)",
                                    width="100%"
                                ),
                                spacing="2",
                                align="start",
                                width="100%"
                            ),
                            spacing="3",
                            align="start"
                        ),
                        padding="1.25em",
                        background=rx.color("green", 1),
                        border="2px solid",
                        border_color=rx.color("green", 6),
                        border_radius="var(--radius-4)",
                        box_shadow="0 2px 8px rgba(0, 0, 0, 0.05)",
                        _hover={
                            "border_color": rx.color("green", 7),
                            "box_shadow": "0 4px 12px rgba(0, 0, 0, 0.1)"
                        },
                        transition="all 0.2s ease"
                    ),
                    
                    # Geopolitical
                    rx.box(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("triangle-alert", size=20, color=rx.color("red", 10)),
                                rx.heading("Geopolitical Risk", size="4", weight="bold", color=rx.color("red", 11)),
                                spacing="2",
                                align="center"
                            ),
                            rx.badge("2 features", color_scheme="red", size="1"),
                            rx.divider(margin_y="0.75em"),
                            rx.vstack(
                                rx.box(
                                    rx.hstack(
                                        rx.icon("circle-dot", size=16, color=rx.color("red", 9)),
                                        rx.text("GPR", size="3", weight="medium"),
                                        spacing="2",
                                        align="center"
                                    ),
                                    padding="0.5em",
                                    background=rx.color("red", 3),
                                    border_radius="var(--radius-2)",
                                    width="100%"
                                ),
                                rx.box(
                                    rx.hstack(
                                        rx.icon("circle-dot", size=16, color=rx.color("red", 9)),
                                        rx.text("GPRA", size="3", weight="medium"),
                                        spacing="2",
                                        align="center"
                                    ),
                                    padding="0.5em",
                                    background=rx.color("red", 3),
                                    border_radius="var(--radius-2)",
                                    width="100%"
                                ),
                                spacing="2",
                                align="start",
                                width="100%"
                            ),
                            spacing="3",
                            align="start"
                        ),
                        padding="1.25em",
                        background=rx.color("red", 1),
                        border="2px solid",
                        border_color=rx.color("red", 6),
                        border_radius="var(--radius-4)",
                        box_shadow="0 2px 8px rgba(0, 0, 0, 0.05)",
                        _hover={
                            "border_color": rx.color("red", 7),
                            "box_shadow": "0 4px 12px rgba(0, 0, 0, 0.1)"
                        },
                        transition="all 0.2s ease"
                    ),
                    
                    columns="3",
                    spacing="4",
                    width="100%"
                ),
                
                spacing="3",
                align="start",
                width="100%"
            ),
            padding="2em",
            background=rx.color("gray", 1),
            border="1px solid",
            border_color=rx.color("gray", 5),
            border_radius="var(--radius-4)"
        ),
        
        spacing="3",
        align="start",
        width="100%",
        margin_bottom="2em"
    )


def key_insights_section() -> rx.Component:
    """Key insights and takeaways section."""
    return rx.vstack(
        rx.heading("Key Insights & Takeaways", size="7", weight="bold", margin_bottom="1em"),
        
        rx.accordion.root(
            rx.accordion.item(
                header="1. Inflation is the Dominant Driver",
                content=rx.text(
                    "CPI shows the strongest correlation (0.87) with gold prices, confirming gold's historical role "
                    "as an inflation hedge. When purchasing power declines, investors flock to gold to preserve wealth. "
                    "This relationship has remained consistent across different economic regimes.",
                    size="3",
                    line_height="1.7"
                )
            ),
            rx.accordion.item(
                header="2. Interest Rates Create Opportunity Cost",
                content=rx.text(
                    "Real interest rates show a negative correlation (-0.26) with gold. When rates are high, "
                    "interest-bearing assets like bonds become more attractive, reducing demand for non-yielding gold. "
                    "Negative real rates (inflation > interest rate) create favorable conditions for gold.",
                    size="3",
                    line_height="1.7"
                )
            ),
            rx.accordion.item(
                header="3. Gold-Equity Correlation Challenges Conventional Wisdom",
                content=rx.text(
                    "Surprisingly, gold and S&P 500 show strong positive correlation (0.82), contradicting the "
                    "'safe haven' narrative. This suggests both assets benefit from liquidity injections and monetary easing. "
                    "Gold may not be the portfolio diversifier many believe it to be in modern markets.",
                    size="3",
                    line_height="1.7"
                )
            ),
            rx.accordion.item(
                header="4. VIX Shows Near-Zero Correlation – A Puzzle",
                content=rx.text(
                    "The VIX (fear index) has essentially zero correlation (r≈0.00) with gold prices. This is unexpected, "
                    "as conventional wisdom suggests gold should rally during high volatility periods. This finding suggests "
                    "market fear and gold demand are driven by different mechanisms than commonly assumed.",
                    size="3",
                    line_height="1.7"
                )
            ),
            rx.accordion.item(
                header="5. Multicollinearity Reduction Improved Model Quality",
                content=rx.text(
                    "Removing 4 highly correlated features (NASDAQ, M2, GPRT, Gold_Futures) reduced redundancy "
                    "while preserving 98% of information. This will help our machine learning models train faster, "
                    "avoid overfitting, and produce more interpretable predictions.",
                    size="3",
                    line_height="1.7"
                )
            ),
            collapsible=True,
            variant="soft",
            width="100%"
        ),
        
        spacing="3",
        align="start",
        width="100%",
        margin_bottom="2em"
    )


# ======================================================================
# MAIN PAGE FUNCTION
# ======================================================================

def eda_page() -> rx.Component:
    """Chapter 2: Exploratory Data Analysis page."""
    
    return page_layout(
        rx.flex(
            rx.vstack(
                chapter_progress(current=2),
                
                rx.vstack(
                    rx.heading(
                        "Chapter 2: Exploratory Data Analysis",
                        size="8",
                        weight="bold",
                        color_scheme="amber",
                        align="center"
                    ),
                    rx.heading(
                        "Understanding the Gold Price Landscape",
                        size="6",
                        weight="bold",
                        color="var(--gray-10)",
                        align="center"
                    ),
                    spacing="1",
                    margin_bottom="1.5em"
                ),
                
                executive_summary(),
                section_divider(),
                
                data_collection_journey(),
                section_divider(),
                
                dataset_overview(),
                section_divider(),
                
                gold_spot_currency_analysis(),
                section_divider(),
                
                target_variable_section(),
                section_divider(),
                
                distribution_analysis_section(),
                section_divider(),
                
                correlation_analysis_section(),
                
                feature_selection_section(),
                section_divider(),
                
                key_insights_section(),
                section_divider(),
                
                rx.box(
                    rx.vstack(
                        rx.heading(
                            "Ready for Modeling",
                            size="6",
                            weight="bold",
                            color_scheme="amber"
                        ),
                        rx.text(
                            "With 13 carefully selected features and deep understanding of the data, "
                            "we're ready to build predictive models. In the next chapter, we'll train "
                            "and compare multiple machine learning architectures to forecast gold prices.",
                            size="4",
                            color="var(--gray-12)",
                            line_height="1.7",
                            text_align="justify"
                        ),
                        spacing="4",
                        align="start"
                    ),
                    padding="1.5em",
                    background_color=rx.color("amber", 2),
                    border_left=f"4px solid {rx.color('amber', 9)}",
                    border_radius="var(--radius-3)",
                    margin_y="1.5em"
                ),
                
                rx.flex(
                    rx.link(
                        rx.button(
                            "Next: Chapter 3 - Modeling ➔",
                            size="3",
                            color_scheme="amber",
                            variant="solid"
                        ),
                        href="/modeling",
                    ),
                    justify="center",
                    width="100%",
                    padding_top="1.5em"
                ),
                
                spacing="5",
                align="start",
                width="100%"
            ),
            
            max_width="900px",
            padding_x="2em",
            padding_y="2em",
            margin_x="auto",
            width="100%"
        )
    )
