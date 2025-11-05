"""Chapter 1: Data Collection - The Foundation of Gold Price Prediction"""

import reflex as rx
from goldsight.components import page_layout, chapter_progress

# ======================================================================
# HELPER COMPONENTS
# ======================================================================

def section_divider() -> rx.Component:
    """A simple visual divider for sections."""
    return rx.divider(margin_y="1.5em")

def next_chapter_navigation(next_title: str, next_route: str) -> rx.Component:
    """A large button to guide the user to the next chapter."""
    return rx.flex(
        rx.link(
            rx.button(
                f"Next: {next_title} ➔",
                size="3",
                color_scheme="amber",
                variant="solid"
            ),
            href=next_route,
        ),
        justify="center",
        width="100%",
        padding_top="1.5em"
    )

# ======================================================================
# PAGE-SPECIFIC COMPONENTS
# ======================================================================

def step_header(step_number: int, title: str, description: str) -> rx.Component:
    """Display step header with number, title and description."""
    return rx.vstack(
        rx.hstack(
            rx.box(
                rx.text(
                    str(step_number),
                    size="4",
                    weight="bold",
                    color="white"
                ),
                width="40px",
                height="40px",
                border_radius="50%",
                background_color=rx.color("amber", 9),
                display="flex",
                align_items="center",
                justify_content="center",
                flex_shrink="0"
            ),
            rx.heading(title, size="6", weight="bold"),
            spacing="3",
            align="center"
        ),
        rx.text(
            description,
            size="4",
            color="var(--gray-11)",
            line_height="1.7"
        ),
        spacing="3",
        align="start",
        width="100%",
        margin_bottom="1em"
    )


# def code_snippet(code: str, language: str = "python") -> rx.Component:
#     """Display syntax highlighted code block."""
#     return rx.box(
#         rx.code_block(
#             code,
#             language=language,
#             show_line_numbers=True,
#             theme="dark",
#             can_copy=True,
#             copy_button_margin="0.5em"
#         ),
#         width="100%",
#         overflow_x="auto",
#         border_radius="var(--radius-3)",
#         border=f"1px solid {rx.color('gray', 6)}",
#         background_color="#1e1e1e",
#         margin_bottom="2em"
#     )


def feature_item_with_dialog(
    feature_name: str,
    feature_ticker: str | None,
    description: str
) -> rx.Component:
    """
    Displays a feature item that, when clicked, opens a dialog
    with a detailed explanation.
    """
    trigger_text = (
        f"{feature_name} ({feature_ticker})" if feature_ticker else feature_name
    )
    
    return rx.dialog.root(
        rx.dialog.trigger(
            rx.list_item(
                rx.text.strong(trigger_text),
                color="var(--blue-11)",
                text_decoration="underline",
                # text_decoration_style="dotted",
                cursor="pointer",
                _hover={"text_decoration": "solid"},
                style={"font_size": "0.9em"}
            )
        ),
        rx.dialog.content(
            rx.dialog.title(feature_name),
            rx.dialog.description(
                rx.vstack(
                    rx.text(description, text_align="left", size="3"),
                    rx.cond(
                        feature_ticker,
                        rx.text("Ticker/ID: ", rx.code(feature_ticker), size="2")
                    ),
                    spacing="3",
                    align="start"
                )
            ),
            rx.dialog.close(
                rx.button("Close", variant="soft", color_scheme="gray", margin_top="1em")
            ),
            style={"max_width": "500px"}
        ),
        modal=True  # Prevent scroll jump
    )


def data_source_card(
    source_name: str,
    description: str,
    frequency: str,
    icon: str = "database",
    href: str = ""
) -> rx.Component:
    """Card displaying information about a data source."""
    return rx.link(
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.icon(icon, size=24, color_scheme="amber"),
                    rx.heading(source_name, size="6", weight="bold"),
                    spacing="2",
                    align="center"
                ),
                rx.text(
                    description,
                    size="3",
                    color="var(--gray-11)"
                ),
                rx.badge(
                    f"Frequency: {frequency}",
                    color_scheme="blue",
                    size="2"
                ),
                spacing="4",
                align="start",
                width="100%"
            ),
            padding="1.5em",
            border="1px solid",
            border_color=rx.color("gray", 5),
            border_radius="var(--radius-4)",
            background_color=rx.color("gray", 1),
            width="100%",
            _hover={
                "border_color": rx.color("amber", 6),
                "background_color": rx.color("amber", 1),
                "transform": "translateY(-2px)",
                "box_shadow": "0 4px 12px rgba(0, 0, 0, 0.1)"
            },
            transition="all 0.2s ease",
            cursor="pointer"
        ),
        href=href,
        target='_blank'
    )

# ======================================================================
# MAIN PAGE FUNCTION
# ======================================================================

def data_collection_page() -> rx.Component:
    """Chapter 1: Data Collection - Storytelling layout."""
    
    # Code snippets from actual notebook
    code_yfinance = """import yfinance as yf

tickers = {
    'GC=F': 'Gold_Futures',
    'SI=F': 'Silver_Futures',
    'CL=F': 'Crude_Oil',
    '^GSPC': 'SP_500',
    '^IXIC': 'NASDAQ',
    'GLD': 'Gold_ETF'
}

# Download 'Close' price data
raw_data = yf.download(list(tickers.keys()), start='2000-01-01', auto_adjust=False)

# Extract Close prices
close_data = raw_data['Close'].copy()

# Rename columns to descriptive names
close_data = close_data.rename(columns=tickers)
"""

    code_fred = """from fredapi import Fred
import pandas as pd

fred = Fred(api_key='YOUR_API_KEY')
start_date = '2000-01-01'

# Download each macroeconomic indicator as a DataFrame
indicators = {
    'CPI': 'CPIAUCSL',
    'Fed_Funds_Rate': 'FEDFUNDS',
    'M2_Supply': 'M2SL',
    'Unemployment': 'UNRATE',
    'Treasury_Yield_10Y': 'GS10',
    'Real_Interest_Rate': 'DFII10',
    'USD_Index': 'DTWEXBGS'
}

macro_monthly = pd.DataFrame()
real_interest_rate = pd.DataFrame()
usd_index = pd.DataFrame()

for name, code in indicators.items():
    series = fred.get_series(code, observation_start=start_date)
    series = series.to_frame(name=name)
    series.index.name = 'Date'
    if name == 'USD_Index':
        usd_index = series
    elif name == 'Real_Interest_Rate':
        real_interest_rate = series
    else:
        if macro_monthly.empty:
            macro_monthly = series
        else:
            macro_monthly = macro_monthly.join(series, how='outer')
"""

    code_vix = """import yfinance as yf 

vix = yf.download("^VIX", start="2000-01-01")
close_vix = vix["Close"]
"""
    
    return page_layout(
        rx.flex(
            rx.vstack(
                chapter_progress(current=1),
                
                rx.vstack(
                    rx.heading(
                        "Chapter 1: The Data",
                        size="8",
                        weight="bold",
                        color_scheme="amber",
                        align="center"
                    ),
                    rx.heading(
                        "Building the Foundation",
                        size="6",
                        weight="bold",
                        color="var(--gray-10)",
                        align="center"
                    ),
                    spacing="1",
                    margin_bottom="1.5em"
                ),
                
                rx.text(
                    "Every great machine learning project begins with data. For gold price prediction, "
                    "we needed more than just historical gold prices—we needed a comprehensive view of "
                    "the economic landscape. We collected 17 features spanning 20+ years (2000-2025) "
                    "from four major categories: Precious Metals, Financial Markets, Macroeconomics, "
                    "and Geopolitical Risk. This rich dataset forms the foundation of our analysis.",
                    size="4",
                    color="var(--gray-12)",
                    text_align="left",
                    line_height="1.7",
                    margin_bottom="1.5em"
                ),
                
                section_divider(),
                
                rx.heading(
                    "Feature Categories (17 Total)",
                    size="7",
                    weight="bold",
                    margin_bottom="1em"
                ),
                
                rx.grid(
                    # Category 1: Precious Metals
                    rx.box(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("gem", size=20, color=rx.color("amber", 9)),
                                rx.heading("Precious Metals", size="5", weight="bold"),
                                spacing="2"
                            ),
                            rx.badge("3 features", color_scheme="amber", size="1"),
                            rx.unordered_list(
                                feature_item_with_dialog(
                                    "Gold Spot", None,
                                    "This is our Target Variable. It represents the price of physical gold (per ounce) in the global market. Our goal is to predict this value."
                                ),
                                feature_item_with_dialog(
                                    "Gold Futures", "GC=F",
                                    "Futures contracts reflect market expectations of where gold prices will be at a future date. It's a key indicator of market sentiment."
                                ),
                                feature_item_with_dialog(
                                    "Silver Futures", "SI=F",
                                    "Silver is a precious metal highly correlated with gold. It often moves in the same direction, representing co-movement in the metals market."
                                ),
                                spacing="1"
                            ),
                            spacing="3",
                            align="start"
                        ),
                        padding="1.25em",
                        border="1px solid",
                        border_color=rx.color("amber", 6),
                        border_radius="var(--radius-3)",
                        background=rx.color("amber", 2),
                    ),
                    
                    # Category 2: Financial Markets
                    rx.box(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("trending-up", size=20, color=rx.color("blue", 9)),
                                rx.heading("Financial Markets", size="5", weight="bold"),
                                spacing="2"
                            ),
                            rx.badge("5 features", color_scheme="blue", size="1"),
                            rx.unordered_list(
                                feature_item_with_dialog(
                                    "S&P 500 & NASDAQ", "^GSPC, ^IXIC",
                                    "Major U.S. stock indices. They are often inversely correlated with gold. When stocks are up ('risk-on'), investors may sell gold (a 'safe-haven' asset)."
                                ),
                                feature_item_with_dialog(
                                    "Crude Oil", "CL=F",
                                    "Oil prices heavily influence inflation (which gold is a hedge against) and the overall cost of industrial production."
                                ),
                                feature_item_with_dialog(
                                    "VIX Index", "^VIX",
                                    "The 'Fear Index'. It measures expected market volatility. When VIX is high (high fear), demand for gold as a safe-haven asset typically increases."
                                ),
                                feature_item_with_dialog(
                                    "Gold ETF", "GLD",
                                    "SPDR Gold Shares (GLD) is an Exchange-Traded Fund. Its holdings reflect direct financial investment demand for gold from retail and institutional investors."
                                ),
                                spacing="1"
                            ),
                            spacing="3",
                            align="start"
                        ),
                        padding="1.25em",
                        border="1px solid",
                        border_color=rx.color("blue", 6),
                        border_radius="var(--radius-3)",
                        background=rx.color("blue", 2),
                    ),
                    
                    # Category 3: Macroeconomic
                    rx.box(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("bar-chart-2", size=20, color=rx.color("green", 9)),
                                rx.heading("Macroeconomic", size="5", weight="bold"),
                                spacing="2"
                            ),
                            rx.badge("6 features", color_scheme="green", size="1"),
                            rx.unordered_list(
                                feature_item_with_dialog(
                                    "CPI (Inflation)", "CPIAUCSL",
                                    "Consumer Price Index. This is a key measure of inflation. Gold is traditionally seen as a hedge against inflation, so as CPI rises, gold demand often follows."
                                ),
                                feature_item_with_dialog(
                                    "Fed Funds Rate", "FEDFUNDS",
                                    "The benchmark interest rate. Higher rates make interest-bearing assets (like bonds) more attractive, reducing the appeal of gold (which pays no interest)."
                                ),
                                feature_item_with_dialog(
                                    "10Y Treasury & Real Rate", "GS10, DFII10",
                                    "The real interest rate (Treasury yield minus inflation). This is a critical driver. When real rates are low or negative, the 'opportunity cost' of holding gold is low, making it more attractive."
                                ),
                                feature_item_with_dialog("USD Index", "DTWEXBGS", "Measures the strength of the U.S. Dollar. Since gold is priced in USD, a stronger dollar makes gold more expensive for foreign buyers, often lowering demand and price (and vice-versa)."),
                                feature_item_with_dialog("M2 Money Supply", "M2SL", "Represents the total amount of money in the economy. A rapid increase in money supply can lead to inflation fears, boosting gold's appeal."),
                                feature_item_with_dialog("Unemployment Rate", "UNRATE", "A key indicator of economic health. High unemployment can signal economic distress, increasing demand for gold as a safe-haven asset."),
                                spacing="1"
                            ),
                            spacing="3",
                            align="start"
                        ),
                        padding="1.25em",
                        border="1px solid",
                        border_color=rx.color("green", 6),
                        border_radius="var(--radius-3)",
                        background=rx.color("green", 2),
                    ),
                    
                    # Category 4: Geopolitical
                    rx.box(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("triangle-alert", size=20, color=rx.color("red", 9)),
                                rx.heading("Geopolitical Risk", size="5", weight="bold"),
                                spacing="2"
                            ),
                            rx.badge("3 features", color_scheme="red", size="1"),
                            rx.unordered_list(
                                feature_item_with_dialog(
                                    "GPR (Risk Index)", "GPR",
                                    "The Geopolitical Risk Index (GPR) measures tensions from news reports. High geopolitical risk (wars, conflicts) drives investors to safe-haven assets like gold."
                                ),
                                feature_item_with_dialog("GPR Acts (GPRA)", "GPRA", "A subset of the GPR index that measures only concrete geopolitical 'acts' (e.g., a new conflict starting)."),
                                feature_item_with_dialog("GPR Threats (GPRT)", "GPRT", "A subset of the GPR index that measures geopolitical 'threats' (e.g., new war threats)."),
                                spacing="1"
                            ),
                            spacing="3",
                            align="start"
                        ),
                        padding="1.25em",
                        border="1px solid",
                        border_color=rx.color("red", 6),
                        border_radius="var(--radius-3)",
                        background=rx.color("red", 2),
                    ),
                    
                    columns="2",
                    spacing="3",
                    width="100%",
                    margin_bottom="1.5em"
                ),
                
                # Why these categories - Full paragraph
                rx.box(
                    rx.vstack(
                        rx.heading(
                            "💡 Why These Categories?",
                            size="6",
                            weight="bold",
                            color_scheme="blue",
                            margin_bottom="0.5em"
                        ),
                        rx.text(
                            "Gold prices don't exist in isolation—they are shaped by a complex interplay of economic, financial, and geopolitical forces. "
                            "Our feature selection is grounded in economic theory and empirical research. ",
                            rx.text.strong("Inflation "),
                            "(measured through CPI and Real Interest Rates) directly affects gold's role as a store of value. When inflation rises, investors flock to gold to preserve purchasing power. ",
                            rx.text.strong("Market Sentiment "),
                            "(captured by stock indices, VIX, and commodity prices) reflects investor risk appetite—during 'risk-on' periods, capital flows to equities; during 'risk-off' periods, it shifts to safe havens like gold. ",
                            rx.text.strong("Monetary Policy "),
                            "(Federal Funds Rate, M2 Money Supply, and Treasury Yields) influences the opportunity cost of holding non-yielding assets like gold. Finally, ",
                            rx.text.strong("Geopolitical Risk "),
                            "(GPR indices) measures global uncertainty and conflict, which historically drives demand for gold as a crisis hedge. "
                            "These 17 features are not arbitrary—they represent the fundamental drivers that economics research has identified as key determinants of gold prices over the past two decades.",
                            size="4",
                            line_height="1.8",
                            text_align="justify",
                            color="var(--gray-12)"
                        ),
                        spacing="3",
                        align="start"
                    ),
                    padding="1.5em",
                    background=rx.color("blue", 2),
                    border_left=f"4px solid {rx.color('blue', 9)}",
                    border_radius="var(--radius-3)",
                    margin_y="1.5em"
                ),
                
                section_divider(),
                
                rx.heading(
                    "Our Data Sources",
                    size="7",
                    weight="bold",
                    margin_bottom="1em"
                ),
                rx.grid(
                    data_source_card(
                        "Yahoo Finance (yfinance)",
                        "Real-time and historical market data for commodities, indices, and ETFs.",
                        "Daily", "trending-up","https://finance.yahoo.com/"
                    ),
                    data_source_card(
                        "FRED API",
                        "Federal Reserve Economic Data - comprehensive macroeconomic indicators.",
                        "Monthly", "bar-chart", "https://fred.stlouisfed.org/"
                    ),
                    data_source_card(
                        "World Gold Council (WGC)",
                        "Official source for our target variable: the daily Gold Spot Price.",
                        "Monthly", "circle-dollar-sign", "https://www.gold.org/"
                    ),
                    data_source_card(
                        "GPR Database",
                        "Measures global geopolitical tensions by Caldara & Iacoviello (2022).",
                        "Monthly", "triangle-alert", "https://www.matteoiacoviello.com/gpr.htm"
                    ),
                    columns="2",
                    spacing="3",
                    width="100%",
                    margin_bottom="1.5em"
                ),
                
                section_divider(),
                
                rx.heading(
                    "Data Collection Process",
                    size="7",
                    weight="bold",
                    margin_bottom="1.5em"
                ),
                
                step_header(
                    1, 
                    "Fetch Market Data (Daily)",
                    "Using the yfinance API, we downloaded daily Close prices for all market-driven features including gold, silver, oil, and major stock indices."
                ),
                rx.code_block(
                    code_yfinance,
                    language="python",
                    show_line_numbers=True,
                    theme="dark",
                    can_copy=True,
                    copy_button_margin="0.5em"
                ),

                # code_snippet(code_yfinance),
                
                step_header(
                    2, 
                    "Fetch Macroeconomic Data (Monthly)",
                    "Using the FRED API, we retrieved all monthly macroeconomic indicators from the Federal Reserve Economic Data database, including inflation, interest rates, and money supply."
                ),

                rx.code_block(
                    code_fred,
                    language="python",
                    show_line_numbers=True,
                    theme="dark",
                    can_copy=True,
                    copy_button_margin="0.5em"
                ),
                # code_snippet(code_fred),
                
                step_header(
                    3, 
                    "Fetch VIX Data (Daily)",
                    "The VIX (Volatility Index) measures market fear and uncertainty. We downloaded daily VIX data using yfinance to capture market sentiment."
                ),

                rx.code_block(
                    code_vix,
                    language="python",
                    show_line_numbers=True,
                    theme="dark",
                    can_copy=True,
                    copy_button_margin="0.5em"
                ),
                # code_snippet(code_vix),
                
                section_divider(),

                rx.box(
                    rx.vstack(
                        rx.heading(
                            "The Foundation is Set",
                            size="6",
                            weight="bold",
                            color_scheme="amber"
                        ),
                        rx.text(
                            "With our raw data collected, we have the building blocks for our models. "
                            "However, this data is messy: it has different frequencies (Daily vs. Monthly) "
                            "and missing values. In the next chapter, we will clean, process, and "
                            "explore this data to uncover its hidden stories.",
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
                
                next_chapter_navigation(
                    "Chapter 2: The Exploration (EDA)",
                    "/eda"
                ),
                
                spacing="5",
                align="start",
                width="100%"
            ),
            
            # Blog-style layout constraints
            max_width="900px",
            padding_x="2em",
            padding_y="2em",
            margin_x="auto",
            width="100%"
        )
    )