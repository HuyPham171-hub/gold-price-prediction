"""Chapter 3: Modeling & Evaluation - From Statistics to Deep Learning"""

import reflex as rx
from goldsight.components import page_layout, chapter_progress

# ======================================================================
# HELPER COMPONENTS
# ======================================================================

def section_divider() -> rx.Component:
    """Visual divider between sections."""
    return rx.divider(margin_y="1.5em")


def model_badge(rank: int, model_name: str) -> rx.Component:
    """Badge showing model ranking."""
    colors = {1: "amber", 2: "gray", 3: "orange"}
    icons = {1: "trophy", 2: "medal", 3: "award"}
    return rx.hstack(
        rx.icon(icons.get(rank, "circle"), size=16, color=rx.color(colors.get(rank, "gray"), 9)),
        rx.text(model_name, size="2", weight="bold"),
        spacing="1",
        align="center"
    )


def metric_card(label: str, value: str, color_scheme: str = "blue", description: str = "") -> rx.Component:
    """Display a single metric card."""
    return rx.box(
        rx.vstack(
            rx.text(label, size="2", color="var(--gray-12)", weight="medium"),
            rx.heading(value, size="7", weight="bold", color=rx.color(color_scheme, 10)),
            rx.cond(
                description != "",
                rx.text(description, size="1", color="var(--gray-10)"),
                rx.fragment()
            ),
            spacing="1",
            align="center"
        ),
        padding="1.25em",
        border="1px solid",
        border_color=rx.color("gray", 5),
        border_radius="var(--radius-3)",
        background=rx.color(color_scheme, 1),
        width="100%",
        _hover={
            "border_color": rx.color(color_scheme, 6),
            "transform": "translateY(-2px)",
            "box_shadow": "0 4px 12px rgba(0, 0, 0, 0.1)"
        },
        transition="all 0.2s ease"
    )


def comparison_table_section(title: str, description: str, data: list, highlight_best: bool = True) -> rx.Component:
    """Reusable comparison table with metrics."""
    
    # Find best model (highest R²)
    best_idx = 0
    if highlight_best and len(data) > 0:
        best_r2 = max([float(row[1].replace("−", "-")) for row in data])  # Handle negative sign
        best_idx = next(i for i, row in enumerate(data) if float(row[1].replace("−", "-")) == best_r2)
    
    table_rows = []
    for idx, row in enumerate(data):
        is_best = idx == best_idx and highlight_best
        row_style = {
            "background": rx.color("green", 2) if is_best else "transparent",
            "font_weight": "bold" if is_best else "normal"
        }
        
        table_rows.append(
            rx.table.row(
                rx.table.cell(
                    rx.hstack(
                        rx.text(row[0]),
                        rx.cond(
                            is_best,
                            rx.icon("trophy", size=16, color=rx.color("amber", 9)),
                            rx.fragment()
                        ),
                        spacing="2",
                        align="center"
                    )
                ),
                rx.table.cell(rx.badge(row[1], color_scheme="green" if float(row[1].replace("−", "-")) > 0.9 else "gray", size="2")),
                rx.table.cell(row[2]),
                rx.table.cell(row[3]),
                rx.table.cell(row[4]),
                style=row_style
            )
        )
    
    return rx.vstack(
        rx.heading(title, size="6", weight="bold", margin_bottom="0.5em"),
        rx.text(description, size="3", color="var(--gray-12)", margin_bottom="1em", line_height="1.7"),
        
        rx.table.root(
            rx.table.header(
                rx.table.row(
                    rx.table.column_header_cell("Model"),
                    rx.table.column_header_cell("R²"),
                    rx.table.column_header_cell("RMSE"),
                    rx.table.column_header_cell("MAE"),
                    rx.table.column_header_cell("Notes"),
                )
            ),
            rx.table.body(*table_rows),
            variant="surface",
            size="3",
            width="100%"
        ),
        
        spacing="3",
        align="start",
        width="100%",
        margin_bottom="2em"
    )


def insight_box(icon: str, title: str, content: str, color_scheme: str = "blue") -> rx.Component:
    """Insight highlight box."""
    return rx.box(
        rx.hstack(
            rx.icon(icon, size=48, color=rx.color(color_scheme, 9)),
            rx.vstack(
                rx.heading(title, size="4", weight="bold"),
                rx.text(content, size="3", color="var(--gray-12)", line_height="1.7"),
                spacing="2",
                align="start"
            ),
            spacing="3",
            align="start"
        ),
        padding="1.25em",
        background=rx.color(color_scheme, 2),
        border_left=f"4px solid {rx.color(color_scheme, 9)}",
        border_radius="var(--radius-3)",
        margin_bottom="1em"
    )


# ======================================================================
# MAIN PAGE SECTIONS
# ======================================================================

def executive_summary() -> rx.Component:
    """Executive summary with key findings."""
    return rx.vstack(
        rx.heading("Overview", size="6", weight="bold", color_scheme="purple"),
        
        rx.box(
            rx.vstack(
                rx.text(
                    "In this chapter, we systematically test 11 different models across 3 algorithm families to find "
                    "the optimal approach for gold price prediction. Rather than jumping to complex solutions, we start "
                    "with simple baselines and progressively add complexity, understanding what each model contributes.",
                    size="3",
                    color="var(--gray-12)",
                    line_height="1.7",
                    margin_bottom="1em"
                ),
                
                rx.heading("What We'll Explore", size="5", weight="bold", margin_bottom="0.5em"),
                rx.unordered_list(
                    rx.list_item(
                        rx.text.strong("Univariate vs Multivariate: "),
                        "Does each feature work alone, or do they need to work together?"
                    ),
                    rx.list_item(
                        rx.text.strong("Linear vs Non-linear: "),
                        "Are relationships straight lines, or do we need curves and interactions?"
                    ),
                    rx.list_item(
                        rx.text.strong("Time Series vs Feature-based: "),
                        "Should we focus on temporal patterns or cross-variable relationships?"
                    ),
                    rx.list_item(
                        rx.text.strong("Simple vs Complex: "),
                        "When does added complexity actually improve predictions?"
                    ),
                    spacing="2",
                    padding_left="1.5em",
                    margin_bottom="1em"
                ),
                
                rx.callout(
                    rx.text(
                        rx.text.strong("The journey ahead"),
                        ": We'll start with simple linear regression to establish a baseline, "
                        "then explore traditional machine learning methods to handle non-linearity, and finally test "
                        "deep learning architectures to capture temporal dependencies. Each step reveals what matters "
                        "most for gold price prediction.",
                        size="2",
                        line_height="1.6"
                    ),
                    icon="info",
                    color_scheme="blue",
                    variant="soft"
                ),
                
                spacing="3",
                align="start"
            ),
            padding="1.5em",
            background=rx.color("purple", 2),
            border_left=f"4px solid {rx.color('purple', 9)}",
            border_radius="var(--radius-3)"
        ),
        
        spacing="3",
        align="start",
        width="100%",
        margin_bottom="2em"
    )


def modeling_philosophy() -> rx.Component:
    """Explain modeling approach."""
    return rx.vstack(
        rx.heading("Modeling Philosophy: Start Simple, Add Complexity", size="7", weight="bold", margin_bottom="1em"),
        
        rx.text(
            "Rather than jumping straight to complex solutions, we follow a systematic approach: "
            "start with simple baselines, understand their limitations, then progressively add complexity. "
            "This ensures we understand what each model contributes and avoid unnecessary sophistication.",
            size="4",
            color="var(--gray-12)",
            line_height="1.7",
            margin_bottom="1.5em"
        ),
        
        rx.grid(
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.icon("activity", size=32, color=rx.color("blue", 9)),
                        rx.heading("1. Baseline", size="5", weight="bold"),
                        spacing="2",
                        align="center"
                    ),
                    rx.text(
                        "Linear Models & Time Series",
                        size="3",
                        color="var(--gray-12)"
                    ),
                    rx.text(
                        "Establish minimum acceptable performance. If simple works, why complicate?",
                        size="2",
                        color="var(--gray-11)",
                        line_height="1.6"
                    ),
                    spacing="2",
                    align="start"
                ),
                padding="1.5em",
                border="1px solid",
                border_color=rx.color("blue", 5),
                border_radius="var(--radius-4)",
                background=rx.color("blue", 1)
            ),
            
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.icon("cpu", size=32, color=rx.color("amber", 9)),
                        rx.heading("2. Traditional ML", size="5", weight="bold"),
                        spacing="2",
                        align="center"
                    ),
                    rx.text(
                        "Non-linear & Ensemble Methods",
                        size="3",
                        color="var(--gray-12)"
                    ),
                    rx.text(
                        "Handle non-linearity and feature interactions. Test if advanced methods outperform linear baselines.",
                        size="2",
                        color="var(--gray-11)",
                        line_height="1.6"
                    ),
                    spacing="2",
                    align="start"
                ),
                padding="1.5em",
                border="1px solid",
                border_color=rx.color("amber", 5),
                border_radius="var(--radius-4)",
                background=rx.color("amber", 1)
            ),
            
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.icon("zap", size=32, color=rx.color("purple", 9)),
                        rx.heading("3. Deep Learning", size="5", weight="bold"),
                        spacing="2",
                        align="center"
                    ),
                    rx.text(
                        "Neural Networks & Sequence Models",
                        size="3",
                        color="var(--gray-12)"
                    ),
                    rx.text(
                        "Capture temporal dependencies and complex patterns. Maximum predictive power.",
                        size="2",
                        color="var(--gray-11)",
                        line_height="1.6"
                    ),
                    spacing="2",
                    align="start"
                ),
                padding="1.5em",
                border="1px solid",
                border_color=rx.color("purple", 5),
                border_radius="var(--radius-4)",
                background=rx.color("purple", 1)
            ),
            
            columns="3",
            spacing="3",
            width="100%"
        ),
        
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.icon("chart-column-stacked", size=32, color=rx.color("green", 9)),
                    rx.heading("Evaluation Criteria", size="5", weight="bold", margin_bottom="0.5em")
                    ),
                rx.grid(
                    rx.vstack(
                        rx.text.strong("R² (Coefficient of Determination)"),
                        rx.text("How much of the gold price's movement can our model explain? (Scale: 0 to 1. Higher is better).", size="2", color="var(--gray-12)"),
                        align="start"
                    ),
                    rx.vstack(
                        rx.text.strong("RMSE (Root Mean Squared Error)"),
                        rx.text("Average prediction error in dollars. Lower is better.", size="2", color="var(--gray-12)"),
                        align="start"
                    ),
                    rx.vstack(
                        rx.text.strong("MAE (Mean Absolute Error)"),
                        rx.text("What is our average error, in dollars? (e.g., 'On average, the model is off by $35').", size="2", color="var(--gray-12)"),
                        align="start"
                    ),
                    columns="3",
                    spacing="3",
                    width="100%"
                ),
                spacing="2",
                align="start"
            ),
            padding="1.5em",
            background=rx.color("gray", 1),
            border="1px solid",
            border_color=rx.color("gray", 5),
            border_radius="var(--radius-3)",
            margin_top="1.5em"
        ),
        
        spacing="3",
        align="start",
        width="100%",
        margin_bottom="2em"
    )


def univariate_regression_detail() -> rx.Component:
    """Detailed univariate regression results."""
    univariate_results = [
        ["CPI", "0.720", "$266.74", "$210.61", "Strongest single predictor"],
        ["S&P_500", "0.619", "$311.12", "$240.87", "Stock market correlation"],
        ["Silver_Futures", "0.526", "$346.97", "$274.55", "Precious metal co-movement"],
        ["USD_Index", "0.361", "$402.80", "$326.55", "Currency strength impact"],
        ["GPR", "0.193", "$452.57", "$368.78", "Geopolitical risk factor"],
        ["GPRA", "0.083", "$482.24", "$382.14", "Action-based risk"],
        ["Real_Interest_Rate", "0.079", "$483.29", "$352.45", "Moderate predictive power"],
        ["Treasury_Yield_10Y", "0.053", "$490.13", "$374.42", "Weak linear relationship"],
        ["VIX", "-0.020", "$508.61", "$403.34", "Near-zero linear fit"],
        ["Unemployment", "-0.002", "$504.21", "$400.82", "Near-zero linear fit"],
        ["Crude_Oil", "0.001", "$503.37", "$391.79", "Near-zero linear fit"],
        ["Fed_Funds_Rate", "-0.043", "$514.37", "$400.19", "Weak negative fit"]
    ]
    
    return rx.vstack(
        rx.heading("Univariate Linear Regression: Testing Each Feature", size="6", weight="bold", margin_bottom="1em"),
        
        rx.text(
            "Before building multivariate models, we tested each of the 13 features individually to understand "
            "their standalone predictive power. This reveals which features have strong linear relationships with gold prices.",
            size="4",
            color="var(--gray-12)",
            line_height="1.7",
            margin_bottom="1.5em"
        ),
        
        rx.tabs.root(
            rx.tabs.list(
                rx.tabs.trigger("Summary Table", value="table"),
                rx.tabs.trigger("Top 3 Features", value="top3"),
                rx.tabs.trigger("Why Others Failed", value="failed"),
            ),
            
            rx.tabs.content(
                rx.vstack(
                    rx.text(
                        "Results sorted by R² (highest to lowest). Negative R² means the model performs worse than simply predicting the mean.",
                        size="3",
                        color="var(--gray-12)",
                        margin_bottom="1em"
                    ),
                    
                    rx.table.root(
                        rx.table.header(
                            rx.table.row(
                                rx.table.column_header_cell("Feature"),
                                rx.table.column_header_cell("R²"),
                                rx.table.column_header_cell("RMSE"),
                                rx.table.column_header_cell("MAE"),
                                rx.table.column_header_cell("Interpretation"),
                            )
                        ),
                        rx.table.body(
                            *[
                                rx.table.row(
                                    rx.table.cell(row[0]),
                                    rx.table.cell(rx.badge(
                                        row[1], 
                                        color_scheme="green" if float(row[1].replace("−", "-")) > 0.5 else ("blue" if float(row[1].replace("−", "-")) > 0 else "red"), 
                                        size="2"
                                    )),
                                    rx.table.cell(row[2]),
                                    rx.table.cell(row[3]),
                                    rx.table.cell(row[4]),
                                    style={
                                        "background": rx.color("green", 2) if i < 3 else "transparent",
                                        "font_weight": "bold" if i < 3 else "normal"
                                    }
                                )
                                for i, row in enumerate(univariate_results)
                            ]
                        ),
                        variant="surface",
                        size="3",
                        width="100%"
                    ),
                    
                    spacing="3",
                    align="start",
                    width="100%"
                ),
                value="table"
            ),
            
            rx.tabs.content(
                rx.vstack(
                    rx.grid(
                        rx.box(
                            rx.vstack(
                                rx.hstack(
                                    rx.icon("trophy", size=24, color=rx.color("amber", 9)),
                                    rx.heading("1. CPI (Inflation)", size="4", weight="bold"),
                                    spacing="2",
                                    align="center"
                                ),
                                rx.heading("R² = 0.720", size="6", weight="bold", color=rx.color("green", 10)),
                                rx.text("RMSE: $266.74 | MAE: $210.61", size="3", color="var(--gray-11)"),
                                rx.divider(margin_y="0.75em"),
                                rx.text(
                                    "Consumer Price Index explains 72% of gold price variance. "
                                    "When inflation rises, gold prices follow as investors seek inflation hedge. "
                                    "This is the single most predictive feature.",
                                    size="2",
                                    color="var(--gray-12)",
                                    line_height="1.6"
                                ),
                                rx.text.strong(
                                    "Formula: Gold = 13.41 x CPI - 1876.60",
                                    size="2",
                                    color=rx.color("amber", 10)
                                ),
                                spacing="2",
                                align="start"
                            ),
                            padding="1.5em",
                            background=rx.color("amber", 1),
                            border="2px solid",
                            border_color=rx.color("amber", 6),
                            border_radius="var(--radius-4)"
                        ),
                        
                        rx.box(
                            rx.vstack(
                                rx.hstack(
                                    rx.icon("trending-up", size=24, color=rx.color("blue", 9)),
                                    rx.heading("2. S&P 500", size="4", weight="bold"),
                                    spacing="2",
                                    align="center"
                                ),
                                rx.heading("R² = 0.619", size="6", weight="bold", color=rx.color("green", 10)),
                                rx.text("RMSE: $311.12 | MAE: $240.87", size="3", color="var(--gray-11)"),
                                rx.divider(margin_y="0.75em"),
                                rx.text(
                                    "Stock market index explains 62% of variance. "
                                    "Surprising positive correlation: both rise in liquidity-driven markets. "
                                    "Challenges 'gold vs stocks' narrative.",
                                    size="2",
                                    color="var(--gray-12)",
                                    line_height="1.6"
                                ),
                                rx.text.strong(
                                    "Formula: Gold = 0.30 x S&P500 + 686.66",
                                    size="2",
                                    color=rx.color("blue", 10)
                                ),
                                spacing="2",
                                align="start"
                            ),
                            padding="1.5em",
                            background=rx.color("blue", 1),
                            border="2px solid",
                            border_color=rx.color("blue", 6),
                            border_radius="var(--radius-4)"
                        ),
                        
                        rx.box(
                            rx.vstack(
                                rx.hstack(
                                    rx.icon("gem", size=24, color=rx.color("purple", 9)),
                                    rx.heading("3. Silver Futures", size="4", weight="bold"),
                                    spacing="2",
                                    align="center"
                                ),
                                rx.heading("R² = 0.526", size="6", weight="bold", color=rx.color("green", 10)),
                                rx.text("RMSE: $346.97 | MAE: $274.55", size="3", color="var(--gray-11)"),
                                rx.divider(margin_y="0.75em"),
                                rx.text(
                                    "Precious metals move together. "
                                    "Silver and gold share similar drivers (inflation hedge, safe haven). "
                                    "53% of gold variance explained by silver alone.",
                                    size="2",
                                    color="var(--gray-12)",
                                    line_height="1.6"
                                ),
                                rx.text.strong(
                                    "Formula: Gold = 50.17 x Silver + 382.50",
                                    size="2",
                                    color=rx.color("purple", 10)
                                ),
                                spacing="2",
                                align="start"
                            ),
                            padding="1.5em",
                            background=rx.color("purple", 1),
                            border="2px solid",
                            border_color=rx.color("purple", 6),
                            border_radius="var(--radius-4)"
                        ),
                        
                        columns="3",
                        spacing="3",
                        width="100%"
                    ),
                    
                    spacing="3",
                    align="start",
                    width="100%"
                ),
                value="top3"
            ),
            
            rx.tabs.content(
                rx.vstack(
                    rx.box(
                        rx.vstack(
                            rx.heading("Classification by Predictive Power", size="5", weight="bold", margin_bottom="1em"),
                            
                            rx.text(
                                "Not all features work well alone. Low R² doesn't mean irrelevant — it means the relationship "
                                "is non-linear, lagged, or requires interaction with other variables.",
                                size="3",
                                color="var(--gray-12)",
                                line_height="1.7",
                                margin_bottom="1em"
                            ),
                            
                            # Weak/Failed features (6 features)
                            rx.heading("Weak/Insignificant (R² < 0.08)", size="4", weight="bold", margin_bottom="0.75em", color=rx.color("red", 10)),
                            rx.grid(
                                rx.vstack(
                                    rx.text.strong("VIX (R² = -0.020)", color=rx.color("red", 10)),
                                    rx.text("High p-value (>> 0.05). Affects gold through time lags/threshold effects.", size="2", color="var(--gray-12)"),
                                    align="start", spacing="1"
                                ),
                                rx.vstack(
                                    rx.text.strong("Crude Oil (R² = 0.001)", color=rx.color("red", 10)),
                                    rx.text("Supply shocks create noise. Works better in multivariate context.", size="2", color="var(--gray-12)"),
                                    align="start", spacing="1"
                                ),
                                rx.vstack(
                                    rx.text.strong("Unemployment (R² = -0.002)", color=rx.color("red", 10)),
                                    rx.text("Indirect effect through Fed policy. Non-linear relationship.", size="2", color="var(--gray-12)"),
                                    align="start", spacing="1"
                                ),
                                rx.vstack(
                                    rx.text.strong("Fed Funds (R² = -0.043)", color=rx.color("red", 10)),
                                    rx.text("Multiple channels with lags. Needs multivariate context.", size="2", color="var(--gray-12)"),
                                    align="start", spacing="1"
                                ),
                                rx.vstack(
                                    rx.text.strong("Treasury Yield (R² = 0.053)", color=rx.color("red", 10)),
                                    rx.text("Regime-dependent. Flight-to-quality vs inflation effects.", size="2", color="var(--gray-12)"),
                                    align="start", spacing="1"
                                ),
                                rx.vstack(
                                    rx.text.strong("Real Interest (R² = 0.079)", color=rx.color("red", 10)),
                                    rx.text("7.9% power. Regime changes (QE vs rate hikes) complicate linear fit.", size="2", color="var(--gray-12)"),
                                    align="start", spacing="1"
                                ),
                                columns="2",
                                spacing="2",
                                width="100%",
                                margin_bottom="1.5em"
                            ),
                            
                            # Moderate predictors (3 features)
                            rx.heading("Moderate Predictors (R² = 0.08–0.36)", size="4", weight="bold", margin_bottom="0.75em", color=rx.color("blue", 10)),
                            rx.grid(
                                rx.vstack(
                                    rx.text.strong("USD Index (R² = 0.361)", color=rx.color("blue", 10)),
                                    rx.text("36% power, significant p-value. Inverse USD-gold relationship, but regime-dependent.", size="2", color="var(--gray-12)"),
                                    align="start", spacing="1"
                                ),
                                rx.vstack(
                                    rx.text.strong("GPR (R² = 0.193)", color=rx.color("blue", 10)),
                                    rx.text("19% power. Safe-haven response to geopolitical events (episodic, not continuous).", size="2", color="var(--gray-12)"),
                                    align="start", spacing="1"
                                ),
                                rx.vstack(
                                    rx.text.strong("GPRA (R² = 0.083)", color=rx.color("blue", 10)),
                                    rx.text("8% power. Action-based risk component, event-driven spikes.", size="2", color="var(--gray-12)"),
                                    align="start", spacing="1"
                                ),
                                columns="3",
                                spacing="2",
                                width="100%"
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
                    
                    rx.box(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("lightbulb", size=24, color=rx.color("amber", 9)),
                                rx.heading("Solution: Multivariate Models", size="4", weight="bold"),
                                spacing="2",
                                align="center",
                                margin_bottom="0.5em"
                            ),
                            rx.text(
                                "These 'weak' features become valuable in multivariate models through interactions. "
                                "Example: Real Interest Rate (R² = 0.079 alone) + CPI + Fed Funds jointly capture "
                                "the real cost of holding gold vs interest-bearing assets.",
                                size="3",
                                color="var(--gray-12)",
                                line_height="1.7"
                            ),
                            spacing="2",
                            align="start"
                        ),
                        padding="1.25em",
                        background=rx.color("amber", 2),
                        border_left=f"4px solid {rx.color('amber', 9)}",
                        border_radius="var(--radius-3)",
                        margin_top="1em"
                    ),
                    
                    spacing="3",
                    align="start",
                    width="100%"
                ),
                value="failed"
            ),
            
            default_value="table",
            width="100%"
        ),
        
        spacing="3",
        align="start",
        width="100%",
        margin_bottom="2em"
    )


def multivariate_regression_detail() -> rx.Component:
    """OLS regression with statistical details."""
    return rx.vstack(
        rx.heading("Multivariate Linear Regression: Combining All Features", size="6", weight="bold", margin_bottom="1em"),
        
        rx.text(
            "Now we use all 13 features simultaneously. This allows the model to capture interactions between variables "
            "(e.g., inflation + interest rates together affecting gold). Results: R² = 0.947, RMSE = $115.88, MAE = $77.06.",
            size="4",
            color="var(--gray-12)",
            line_height="1.7",
            margin_bottom="1.5em"
        ),
        
        rx.grid(
            metric_card("R²", "0.947", "green", "95% variance explained"),
            metric_card("Adj R²", "0.928", "blue", "Adjusted for features"),
            metric_card("RMSE", "$115.88", "purple", "Typical error"),
            metric_card("MAE", "$77.06", "amber", "Average deviation"),
            columns="4",
            spacing="3",
            width="100%",
            margin_y="1em"
        ),
        
        rx.tabs.root(
            rx.tabs.list(
                rx.tabs.trigger("Coefficients & Significance", value="coef"),
                rx.tabs.trigger("Model Diagnostics", value="diag"),
                rx.tabs.trigger("VIF Analysis", value="vif"),
            ),
            
            rx.tabs.content(
                rx.vstack(
                    rx.text(
                        "OLS (Ordinary Least Squares) regression output showing coefficient estimates, standard errors, and statistical significance (p-values). "
                        "Features with p < 0.05 are statistically significant.",
                        size="3",
                        color="var(--gray-12)",
                        margin_bottom="1em",
                        line_height="1.6"
                    ),
                    
                    rx.table.root(
                        rx.table.header(
                            rx.table.row(
                                rx.table.column_header_cell("Feature"),
                                rx.table.column_header_cell("Coefficient (β)"),
                                rx.table.column_header_cell("p-value"),
                                rx.table.column_header_cell("95% CI"),
                                rx.table.column_header_cell("Significance"),
                            )
                        ),
                        rx.table.body(
                            rx.table.row(
                                rx.table.cell("Intercept"),
                                rx.table.cell("-1009.58"),
                                rx.table.cell(rx.badge("0.000", color_scheme="green", size="2")),
                                rx.table.cell("[-1554, -465]"),
                                rx.table.cell("Highly significant"),
                                style={"background": rx.color("green", 2), "font_weight": "bold"}
                            ),
                            rx.table.row(
                                rx.table.cell(rx.text.strong("Silver_Futures")),
                                rx.table.cell(rx.text.strong("+25.49")),
                                rx.table.cell(rx.badge("0.000", color_scheme="green", size="2")),
                                rx.table.cell("[21.35, 29.64]"),
                                rx.table.cell("Very strong"),
                                style={"background": rx.color("green", 2), "font_weight": "bold"}
                            ),
                            rx.table.row(
                                rx.table.cell(rx.text.strong("Unemployment")),
                                rx.table.cell(rx.text.strong("+32.04")),
                                rx.table.cell(rx.badge("0.000", color_scheme="green", size="2")),
                                rx.table.cell("[19.63, 44.45]"),
                                rx.table.cell("Positive"),
                                style={"background": rx.color("green", 2), "font_weight": "bold"}
                            ),
                            rx.table.row(
                                rx.table.cell(rx.text.strong("CPI")),
                                rx.table.cell(rx.text.strong("+10.20")),
                                rx.table.cell(rx.badge("0.000", color_scheme="green", size="2")),
                                rx.table.cell("[7.46, 12.95]"),
                                rx.table.cell("Inflation hedge"),
                                style={"background": rx.color("green", 2), "font_weight": "bold"}
                            ),
                            rx.table.row(
                                rx.table.cell(rx.text.strong("S&P_500")),
                                rx.table.cell(rx.text.strong("+0.103")),
                                rx.table.cell(rx.badge("0.000", color_scheme="green", size="2")),
                                rx.table.cell("[0.050, 0.156]"),
                                rx.table.cell("Market linkage"),
                                style={"background": rx.color("green", 2), "font_weight": "bold"}
                            ),
                            rx.table.row(
                                rx.table.cell(rx.text.strong("USD_Index")),
                                rx.table.cell(rx.text.strong("-7.84")),
                                rx.table.cell(rx.badge("0.010", color_scheme="green", size="2")),
                                rx.table.cell("[-13.76, -1.91]"),
                                rx.table.cell("Currency inverse"),
                                style={"background": rx.color("green", 2), "font_weight": "bold"}
                            ),
                            rx.table.row(
                                rx.table.cell(rx.text.strong("Crude_Oil")),
                                rx.table.cell(rx.text.strong("-2.20")),
                                rx.table.cell(rx.badge("0.022", color_scheme="green", size="2")),
                                rx.table.cell("[-4.08, -0.33]"),
                                rx.table.cell("Negative (multicollinearity)"),
                                style={"background": rx.color("green", 2), "font_weight": "bold"}
                            ),
                            rx.table.row(
                                rx.table.cell("VIX"),
                                rx.table.cell("+1.51"),
                                rx.table.cell(rx.badge("0.239", color_scheme="gray", size="2")),
                                rx.table.cell("[-1.01, 4.02]"),
                                rx.table.cell("Not significant"),
                            ),
                            rx.table.row(
                                rx.table.cell("Treasury_Yield_10Y"),
                                rx.table.cell("-52.38"),
                                rx.table.cell(rx.badge("0.171", color_scheme="gray", size="2")),
                                rx.table.cell("[-127.63, 22.87]"),
                                rx.table.cell("Not significant"),
                            ),
                            rx.table.row(
                                rx.table.cell("Real_Interest_Rate"),
                                rx.table.cell("+24.05"),
                                rx.table.cell(rx.badge("0.525", color_scheme="gray", size="2")),
                                rx.table.cell("[-50.55, 98.66]"),
                                rx.table.cell("Not significant"),
                            ),
                            rx.table.row(
                                rx.table.cell("Fed_Funds_Rate"),
                                rx.table.cell("+5.12"),
                                rx.table.cell(rx.badge("0.609", color_scheme="gray", size="2")),
                                rx.table.cell("[-14.62, 24.87]"),
                                rx.table.cell("Not significant"),
                            ),
                            rx.table.row(
                                rx.table.cell("GPR"),
                                rx.table.cell("+0.22"),
                                rx.table.cell(rx.badge("0.682", color_scheme="gray", size="2")),
                                rx.table.cell("[-0.86, 1.31]"),
                                rx.table.cell("Not significant"),
                            ),
                            rx.table.row(
                                rx.table.cell("GPRA"),
                                rx.table.cell("+0.08"),
                                rx.table.cell(rx.badge("0.867", color_scheme="gray", size="2")),
                                rx.table.cell("[-0.84, 0.99]"),
                                rx.table.cell("Not significant"),
                            ),
                        ),
                        variant="surface",
                        size="3",
                        width="100%"
                    ),
                    
                    rx.box(
                        rx.vstack(
                            rx.heading("Key Findings", size="4", weight="bold", margin_bottom="0.5em"),
                            rx.unordered_list(
                                rx.list_item(
                                    rx.text.strong("6 significant features (p < 0.05): "),
                                    "Silver, Unemployment, CPI, S&P500, USD Index, Crude Oil"
                                ),
                                rx.list_item(
                                    rx.text.strong("Unemployment coefficient (+32.04): "),
                                    "When unemployment increase -> gold increase (safe haven during economic stress)"
                                ),
                                rx.list_item(
                                    rx.text.strong("Crude Oil negative (-2.20): "),
                                    "Counterintuitive, likely due to multicollinearity with CPI"
                                ),
                                rx.list_item(
                                    rx.text.strong("7 non-significant features: "),
                                    "VIX, interest rates, geopolitical indices (redundant in multivariate context)"
                                ),
                                spacing="2",
                                padding_left="1.5em"
                            ),
                            spacing="2",
                            align="start"
                        ),
                        padding="1.25em",
                        background=rx.color("blue", 1),
                        border_left=f"4px solid {rx.color('blue', 9)}",
                        border_radius="var(--radius-3)",
                        margin_top="1em"
                    ),
                    
                    spacing="3",
                    align="start",
                    width="100%"
                ),
                value="coef"
            ),
            
            rx.tabs.content(
                rx.vstack(
                    rx.heading("Model Diagnostics & Assumptions", size="5", weight="bold", margin_bottom="1em"),
                    
                    rx.grid(
                        rx.box(
                            rx.vstack(
                                rx.text.strong("Overall Model Fit"),
                                rx.divider(margin_y="0.5em"),
                                rx.hstack(
                                    rx.text("F-statistic:", size="2", color="var(--gray-11)"),
                                    rx.heading("312.9", size="5", color=rx.color("green", 10)),
                                    spacing="2",
                                    align="center"
                                ),
                                rx.text("Prob (F) = 1.89e-110 ~ 0.000", size="2", color="var(--gray-12)"),
                                rx.text("Model is highly significant", size="2", color=rx.color("green", 10), weight="bold"),
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
                                rx.text.strong("Durbin-Watson"),
                                rx.divider(margin_y="0.5em"),
                                rx.hstack(
                                    rx.text("DW statistic:", size="2", color="var(--gray-11)"),
                                    rx.heading("2.221", size="5", color=rx.color("blue", 10)),
                                    spacing="2",
                                    align="center"
                                ),
                                rx.text("Target: near 2.0", size="2", color="var(--gray-12)"),
                                rx.text("No autocorrelation", size="2", color=rx.color("green", 10), weight="bold"),
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
                                rx.text.strong("Omnibus Test"),
                                rx.divider(margin_y="0.5em"),
                                rx.hstack(
                                    rx.text("Prob:", size="2", color="var(--gray-11)"),
                                    rx.heading("0.000", size="5", color=rx.color("red", 10)),
                                    spacing="2",
                                    align="center"
                                ),
                                rx.text("Skew: 0.928 | Kurtosis: 10.38", size="2", color="var(--gray-12)"),
                                rx.text("Residuals not normal", size="2", color=rx.color("red", 10), weight="bold"),
                                spacing="2",
                                align="start"
                            ),
                            padding="1.25em",
                            background=rx.color("red", 1),
                            border="1px solid",
                            border_color=rx.color("red", 5),
                            border_radius="var(--radius-3)"
                        ),
                        
                        rx.box(
                            rx.vstack(
                                rx.text.strong("Condition Number"),
                                rx.divider(margin_y="0.5em"),
                                rx.hstack(
                                    rx.text("Cond No.:", size="2", color="var(--gray-11)"),
                                    rx.heading("9.90e+04", size="5", color=rx.color("orange", 10)),
                                    spacing="2",
                                    align="center"
                                ),
                                rx.text("Threshold: > 30 indicates multicollinearity", size="2", color="var(--gray-12)"),
                                rx.text("Moderate multicollinearity", size="2", color=rx.color("orange", 10), weight="bold"),
                                spacing="2",
                                align="start"
                            ),
                            padding="1.25em",
                            background=rx.color("orange", 1),
                            border="1px solid",
                            border_color=rx.color("orange", 5),
                            border_radius="var(--radius-3)"
                        ),
                        
                        columns="2",
                        spacing="3",
                        width="100%"
                    ),
                    
                    rx.box(
                        rx.vstack(
                            rx.heading("Interpretation", size="4", weight="bold", margin_bottom="0.5em"),
                            rx.unordered_list(
                                rx.list_item(
                                    rx.text.strong("Strong overall fit: "),
                                    "F-statistic = 312.9 confirms the model explains variance significantly better than the null model"
                                ),
                                rx.list_item(
                                    rx.text.strong("No autocorrelation: "),
                                    "Durbin-Watson = 2.221 ~ 2.0 means residuals are independent (good for regression assumptions)"
                                ),
                                rx.list_item(
                                    rx.text.strong("Non-normal residuals: "),
                                    "Skew = 0.928, Kurtosis = 10.38 indicate heavy-tailed distribution. This affects t-test/F-test reliability."
                                ),
                                rx.list_item(
                                    rx.text.strong("Multicollinearity present: "),
                                    "Condition number = 99,000 suggests some features are highly correlated (e.g., CPI <-> M2, S&P <-> NASDAQ removed earlier)"
                                ),
                                rx.list_item(
                                    rx.text.strong("Solution: "),
                                    "Use Ridge regression or remove non-significant features to reduce multicollinearity"
                                ),
                                spacing="2",
                                padding_left="1.5em"
                            ),
                            spacing="2",
                            align="start"
                        ),
                        padding="1.25em",
                        background=rx.color("gray", 1),
                        border="1px solid",
                        border_color=rx.color("gray", 5),
                        border_radius="var(--radius-3)",
                        margin_top="1em"
                    ),
                    
                    spacing="3",
                    align="start",
                    width="100%"
                ),
                value="diag"
            ),
            
            rx.tabs.content(
                rx.vstack(
                    rx.heading("Variance Inflation Factor (VIF) Analysis", size="5", weight="bold", margin_bottom="1em"),
                    
                    rx.text(
                        "VIF quantifies how much a feature's variance is inflated due to multicollinearity. "
                        "Rule of thumb: VIF > 10 indicates high multicollinearity requiring attention.",
                        size="3",
                        color="var(--gray-12)",
                        margin_bottom="1em",
                        line_height="1.6"
                    ),
                    
                    rx.table.root(
                        rx.table.header(
                            rx.table.row(
                                rx.table.column_header_cell("Feature"),
                                rx.table.column_header_cell("VIF"),
                                rx.table.column_header_cell("Status"),
                                rx.table.column_header_cell("Interpretation"),
                            )
                        ),
                        rx.table.body(
                            rx.table.row(
                                rx.table.cell("CPI"),
                                rx.table.cell(rx.heading("1805.21", size="4", color=rx.color("red", 10))),
                                rx.table.cell(rx.badge("Severe", color_scheme="red", size="2")),
                                rx.table.cell("Extremely high correlation with other macro variables"),
                            ),
                            rx.table.row(
                                rx.table.cell("USD_Index"),
                                rx.table.cell(rx.heading("1012.78", size="4", color=rx.color("red", 10))),
                                rx.table.cell(rx.badge("Severe", color_scheme="red", size="2")),
                                rx.table.cell("Strong correlation with interest rates and inflation"),
                            ),
                            rx.table.row(
                                rx.table.cell("Treasury_Yield_10Y"),
                                rx.table.cell(rx.heading("179.92", size="4", color=rx.color("red", 10))),
                                rx.table.cell(rx.badge("Severe", color_scheme="red", size="2")),
                                rx.table.cell("Tied to Fed policy and real interest rates"),
                            ),
                            rx.table.row(
                                rx.table.cell("Crude_Oil"),
                                rx.table.cell(rx.heading("76.45", size="4", color=rx.color("red", 10))),
                                rx.table.cell(rx.badge("High", color_scheme="red", size="2")),
                                rx.table.cell("Energy component highly correlated with CPI"),
                            ),
                            rx.table.row(
                                rx.table.cell("GPR"),
                                rx.table.cell(rx.heading("50.16", size="4", color=rx.color("orange", 10))),
                                rx.table.cell(rx.badge("High", color_scheme="orange", size="2")),
                                rx.table.cell("Geopolitical risk overlaps with market uncertainty"),
                            ),
                            rx.table.row(
                                rx.table.cell("S&P_500"),
                                rx.table.cell(rx.heading("48.05", size="4", color=rx.color("orange", 10))),
                                rx.table.cell(rx.badge("High", color_scheme="orange", size="2")),
                                rx.table.cell("Stock index correlated with macro conditions"),
                            ),
                            rx.table.row(
                                rx.table.cell("Silver_Futures"),
                                rx.table.cell(rx.heading("33.21", size="4", color=rx.color("orange", 10))),
                                rx.table.cell(rx.badge("High", color_scheme="orange", size="2")),
                                rx.table.cell("Precious metals co-movement"),
                            ),
                            rx.table.row(
                                rx.table.cell("GPRA"),
                                rx.table.cell(rx.heading("32.72", size="4", color=rx.color("orange", 10))),
                                rx.table.cell(rx.badge("High", color_scheme="orange", size="2")),
                                rx.table.cell("Action-based risk correlates with GPR"),
                            ),
                            rx.table.row(
                                rx.table.cell("Real_Interest_Rate"),
                                rx.table.cell(rx.heading("30.81", size="4", color=rx.color("orange", 10))),
                                rx.table.cell(rx.badge("High", color_scheme="orange", size="2")),
                                rx.table.cell("Derived from Fed rate and inflation"),
                            ),
                            rx.table.row(
                                rx.table.cell("Unemployment"),
                                rx.table.cell(rx.heading("23.61", size="4", color=rx.color("orange", 10))),
                                rx.table.cell(rx.badge("Moderate", color_scheme="orange", size="2")),
                                rx.table.cell("Labor market reflects macro conditions"),
                            ),
                            rx.table.row(
                                rx.table.cell("VIX"),
                                rx.table.cell(rx.heading("11.45", size="4", color=rx.color("blue", 10))),
                                rx.table.cell(rx.badge("Moderate", color_scheme="blue", size="2")),
                                rx.table.cell("Volatility index, some overlap with risk measures"),
                            ),
                            rx.table.row(
                                rx.table.cell("Fed_Funds_Rate"),
                                rx.table.cell("9.87"),
                                rx.table.cell(rx.badge("Low", color_scheme="green", size="2")),
                                rx.table.cell("Policy rate, relatively independent"),
                            ),
                        ),
                        variant="surface",
                        size="3",
                        width="100%"
                    ),
                    
                    rx.box(
                        rx.vstack(
                            rx.heading("Multicollinearity Assessment", size="4", weight="bold", margin_bottom="0.5em"),
                            rx.unordered_list(
                                rx.list_item(
                                    rx.text.strong("Severe multicollinearity (VIF > 100): "),
                                    "CPI (1805.21), USD Index (1012.78), and Treasury Yield (179.92) show extreme correlation with other macroeconomic variables"
                                ),
                                rx.list_item(
                                    rx.text.strong("High multicollinearity (VIF 30-100): "),
                                    "Crude Oil (76.45), GPR (50.16), S&P 500 (48.05), Silver Futures (33.21), GPRA (32.72), and Real Interest Rate (30.81)"
                                ),
                                rx.list_item(
                                    rx.text.strong("Moderate multicollinearity (VIF 10-30): "),
                                    "Unemployment (23.61) and VIX (11.45) indicate some correlation with other features"
                                ),
                                rx.list_item(
                                    rx.text.strong("Low multicollinearity (VIF < 10): "),
                                    "Only Fed Funds Rate (9.87) falls below the traditional threshold, suggesting relatively independent behavior"
                                ),
                                rx.list_item(
                                    rx.text.strong("Ridge regression test: "),
                                    "Applied L2 regularization showed no improvement (R² = 0.947 identical to OLS), indicating that while multicollinearity exists, it does not significantly degrade predictive performance"
                                ),
                                rx.list_item(
                                    rx.text.strong("Modeling decision: "),
                                    "Retained all features despite high VIF values. The multicollinearity reflects genuine economic relationships (e.g., inflation driving both CPI and USD strength). "
                                    "Removing features would sacrifice valuable information without meaningful performance gains."
                                ),
                                spacing="2",
                                padding_left="1.5em"
                            ),
                            spacing="2",
                            align="start"
                        ),
                        padding="1.25em",
                        background=rx.color("purple", 1),
                        border_left=f"4px solid {rx.color('purple', 9)}",
                        border_radius="var(--radius-3)",
                        margin_top="1em"
                    ),
                    
                    spacing="3",
                    align="start",
                    width="100%"
                ),
                value="vif"
            ),
            
            default_value="coef",
            width="100%"
        ),
        
        spacing="3",
        align="start",
        width="100%",
        margin_bottom="2em"
    )


def polynomial_regression_section() -> rx.Component:
    """Polynomial regression - brief section."""
    return rx.vstack(
        rx.heading("Polynomial Regression: Testing Non-linear Curves", size="6", weight="bold", margin_bottom="1em"),
        
        rx.text(
            "Can we improve univariate predictions by fitting curves instead of straight lines? "
            "We tested polynomial regression (degree=2) on each feature to capture non-linear relationships.",
            size="4",
            color="var(--gray-12)",
            line_height="1.7",
            margin_bottom="1em"
        ),
        
        rx.box(
            rx.vstack(
                rx.heading("Best Result: Silver (R² = 0.537)", size="5", weight="bold", margin_bottom="0.75em", color=rx.color("green", 10)),
                rx.text(
                    "Polynomial regression on Silver Futures achieves R²=0.537 vs linear R²=0.526 - a marginal 1% improvement. "
                    "CPI polynomial performs worse (R²=0.698 vs linear 0.720). Most features show no benefit from polynomial transformation.",
                    size="3",
                    color="var(--gray-12)",
                    line_height="1.7",
                    margin_bottom="1em"
                ),
                rx.heading("Verdict: Curves Don't Help", size="5", weight="bold", margin_bottom="0.5em"),
                rx.text(
                    "Gold-feature relationships are approximately linear. Adding polynomial terms creates overfitting risk "
                    "without meaningful performance gain. Multivariate linear models remain the better path.",
                    size="3",
                    color="var(--gray-12)",
                    line_height="1.7"
                ),
                spacing="2",
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


def time_series_section() -> rx.Component:
    """ARIMA/SARIMA - brief section."""
    return rx.vstack(
        rx.heading("Time Series Models: ARIMA & SARIMA", size="6", weight="bold", margin_bottom="1em"),
        
        rx.text(
            "Can we predict gold prices using only historical patterns (autoregressive models)? "
            "We tested ARIMA (1,1,1) and SARIMA (1,1,1)x(1,1,1,12) with monthly seasonality.",
            size="4",
            color="var(--gray-12)",
            line_height="1.7",
            margin_bottom="1em"
        ),
        
        rx.grid(
            rx.box(
                rx.vstack(
                    rx.heading("ARIMA (1,1,1)", size="5", weight="bold", margin_bottom="0.5em"),
                    rx.heading("R² = -0.480", size="6", color=rx.color("red", 10), margin_bottom="0.5em"),
                    rx.text("RMSE: $503.12 | MAE: $321.93", size="2", color="var(--gray-11)", margin_bottom="0.75em"),
                    rx.text(
                        "Negative R² means worse than predicting the mean. Autoregressive patterns alone cannot explain gold prices. "
                        "Gold responds to economic events, not just past values.",
                        size="2",
                        color="var(--gray-12)",
                        line_height="1.6"
                    ),
                    spacing="1",
                    align="start"
                ),
                padding="1.25em",
                background=rx.color("red", 1),
                border="1px solid",
                border_color=rx.color("red", 5),
                border_radius="var(--radius-4)"
            ),
            
            rx.box(
                rx.vstack(
                    rx.heading("SARIMA (1,1,1)x(1,1,1,12)", size="5", weight="bold", margin_bottom="0.5em"),
                    rx.heading("R² = 0.270", size="6", color=rx.color("orange", 10), margin_bottom="0.5em"),
                    rx.text("RMSE: $353.57 | MAE: $233.26", size="2", color="var(--gray-11)", margin_bottom="0.75em"),
                    rx.text(
                        "Seasonal component improves performance but still explains only 27% of variance. "
                        "Monthly seasonality exists but is weak compared to macroeconomic drivers.",
                        size="2",
                        color="var(--gray-12)",
                        line_height="1.6"
                    ),
                    spacing="1",
                    align="start"
                ),
                padding="1.25em",
                background=rx.color("orange", 1),
                border="1px solid",
                border_color=rx.color("orange", 5),
                border_radius="var(--radius-4)"
            ),
            
            columns="2",
            spacing="3",
            width="100%"
        ),
        
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.icon("triangle-alert", size=24, color=rx.color("red", 9)),
                    rx.heading("Why Time Series Failed", size="4", weight="bold"),
                    spacing="2",
                    align="center",
                    margin_bottom="0.5em"
                ),
                rx.text(
                    "Time series models assume future values depend on past values. Gold prices violate this assumption - "
                    "they respond to inflation, interest rates, geopolitical events. Historical patterns alone miss the fundamental drivers. "
                    "This limitation provides a clear hypothesis and leads directly to our next logical step: testing multivariate models that can see and utilize this external economic data.",
                    size="3",
                    color="var(--gray-12)",
                    line_height="1.7"
                ),
                spacing="2",
                align="start"
            ),
            padding="1.25em",
            background=rx.color("red", 2),
            border_left=f"4px solid {rx.color('red', 9)}",
            border_radius="var(--radius-3)",
            margin_top="1em"
        ),
        
        spacing="3",
        align="start",
        width="100%",
        margin_bottom="2em"
    )


def baseline_models() -> rx.Component:
    """Baseline models comparison table."""
    baseline_data = [
        ["Linear Regression", "0.947", "$115.88", "$77.06", "Strong multivariate baseline"],
        ["Ridge Regression", "0.947", "$115.88", "$77.06", "No improvement"],
        ["Polynomial (degree=2)", "0.537", "$342.78", "$270.69", "Best univariate: Silver"],
        ["ARIMA (1,1,1)", "-0.480", "$503.12", "$321.93", "Failed - worse than mean"],
        ["SARIMA (1,1,1)x(1,1,1,12)", "0.270", "$353.57", "$233.26", "Poor - weak seasonality"]
    ]
    
    return rx.vstack(
        rx.heading("Baseline Models Summary", size="6", weight="bold", margin_bottom="1em"),
        
        rx.table.root(
            rx.table.header(
                rx.table.row(
                    rx.table.column_header_cell("Model"),
                    rx.table.column_header_cell("R²"),
                    rx.table.column_header_cell("RMSE"),
                    rx.table.column_header_cell("MAE"),
                    rx.table.column_header_cell("Notes"),
                )
            ),
            rx.table.body(
                *[
                    rx.table.row(
                        rx.table.cell(
                            rx.hstack(
                                rx.text(row[0]),
                                rx.cond(
                                    i == 0,
                                    rx.icon("trophy", size=16, color=rx.color("amber", 9)),
                                    rx.fragment()
                                ),
                                spacing="2",
                                align="center"
                            )
                        ),
                        rx.table.cell(rx.badge(
                            row[1], 
                            color_scheme="green" if float(row[1]) > 0.9 else ("blue" if float(row[1]) > 0 else "red"), 
                            size="2"
                        )),
                        rx.table.cell(row[2]),
                        rx.table.cell(row[3]),
                        rx.table.cell(row[4]),
                        style={
                            "background": rx.color("green", 2) if i == 0 else "transparent",
                            "font_weight": "bold" if i == 0 else "normal"
                        }
                    )
                    for i, row in enumerate(baseline_data)
                ]
            ),
            variant="surface",
            size="3",
            width="100%",
            margin_bottom="1em"
        ),
        
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.icon("circle-check", size=24, color=rx.color("green", 9)),
                    rx.heading("Best Baseline: Multivariate Linear Regression", size="4", weight="bold"),
                    spacing="2",
                    align="center",
                    margin_bottom="0.5em"
                ),
                rx.text(
                    "R²=0.947 indicates 95% of gold price variance explained. The multivariate approach combining inflation, "
                    "interest rates, stock market, and currency data substantially outperforms all univariate and time series methods. "
                    "This demonstrates that gold prices are driven by macroeconomic interactions rather than single factors or historical patterns.",
                    size="3",
                    color="var(--gray-12)",
                    line_height="1.7"
                ),
                spacing="2",
                align="start"
            ),
            padding="1.25em",
            background=rx.color("green", 2),
            border_left=f"4px solid {rx.color('green', 9)}",
            border_radius="var(--radius-3)"
        ),
        
        spacing="3",
        align="start",
        width="100%",
        margin_bottom="2em"
    )


def traditional_ml() -> rx.Component:
    """Traditional ML models comparison."""
    ml_data = [
        ["Support Vector Regression (SVR)", "0.986", "$59.93", "$43.77", "GridSearch: C=100, gamma=0.01"],
        ["Random Forest", "0.986", "$59.93", "$43.77", "500 trees, depth=20, 1620 CV fits"],
        ["XGBoost", "0.973", "$82.67", "$51.11", "Underperformed - possible overfitting"]
    ]
    
    return rx.vstack(
        comparison_table_section(
            "Traditional Machine Learning: Non-Linear Methods",
            "Moving beyond linear assumptions, we test kernel-based (SVR) and tree-based (Random Forest, XGBoost) methods. "
            "These models can capture non-linear relationships and feature interactions. Both SVR and Random Forest achieve R²=0.986, "
            "a significant jump from the baseline, reducing error by nearly 50%.",
            ml_data,
            highlight_best=True
        ),
        
        rx.grid(
            rx.box(
                rx.vstack(
                    rx.heading("SVR: Support Vector Regression", size="4", weight="bold", margin_bottom="0.5em"),
                    rx.text(
                        "Support Vector Regression with RBF kernel maps features into high-dimensional space, "
                        "capturing complex non-linear patterns. GridSearchCV tested 27 combinations to find optimal hyperparameters.",
                        size="2",
                        color="var(--gray-12)",
                        line_height="1.6"
                    ),
                    rx.unordered_list(
                        rx.list_item("Best C=100 (regularization strength)"),
                        rx.list_item("Best gamma=0.01 (kernel coefficient)"),
                        rx.list_item("Best epsilon=0.01 (margin tolerance)"),
                        spacing="1",
                        padding_left="1em"
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
                    rx.heading("Random Forest: Ensemble", size="4", weight="bold", margin_bottom="0.5em"),
                    rx.text(
                        "Random Forest trains 500 decision trees on random subsets of features, then averages predictions. "
                        "This ensemble approach reduces overfitting while capturing non-linear patterns.",
                        size="2",
                        color="var(--gray-12)",
                        line_height="1.6"
                    ),
                    rx.unordered_list(
                        rx.list_item("500 estimators (trees)"),
                        rx.list_item("Max depth = 20 layers"),
                        rx.list_item("1,620 CV fits (5-fold x 324 configs)"),
                        spacing="1",
                        padding_left="1em"
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
            
            columns="2",
            spacing="3",
            width="100%",
            margin_y="1em"
        ),
        
        insight_box(
            "circle-alert",
            "XGBoost Surprise: Why Did It Underperform?",
            "XGBoost (R²=0.973) surprisingly performed worse than SVR and Random Forest. This suggests the model may be overfitting "
            "to training data despite regularization. XGBoost's strength lies in tabular data with complex interactions, but our "
            "relatively clean dataset with strong linear trends may not benefit as much from gradient boosting's aggressive optimization.",
            color_scheme="amber"
        ),
        
        insight_box(
            "target",
            "Feature Importance from Random Forest",
            "Top 3 most important features: 1) CPI (inflation) - 32% importance, 2) Silver_Futures - 18%, 3) S&P_500 - 15%. "
            "This confirms our EDA findings: inflation and precious metals co-movement are the strongest drivers of gold prices.",
            color_scheme="green"
        ),
        
        spacing="3",
        align="start",
        width="100%",
        margin_bottom="2em"
    )


def deep_learning_univariate() -> rx.Component:
    """Deep learning univariate models."""
    dl_uni_data = [
        ["MLP (Feedforward)", "0.960", "$100.62", "$78.85", "256->128->64->32 neurons, Dropout"],
        ["GRU (Univariate)", "0.843", "$164.93", "$122.95", "64->64 units, window=12"],
        ["LSTM (Univariate)", "0.603", "$262.55", "$193.85", "64->64 units, gates struggle"],
        ["RNN (Univariate)", "0.600", "$263.33", "$184.26", "Simple RNN insufficient"]
    ]
    
    return rx.vstack(
        comparison_table_section(
            "Deep Learning - Univariate (Gold Price Only)",
            "Before using all 13 features, we test if deep learning can extract temporal patterns from gold price history alone. "
            "MLP (feedforward) performs well (R²=0.960) as it uses all features but no sequence. "
            "Recurrent models (RNN/LSTM/GRU) use sliding windows of past prices but struggle without external features.",
            dl_uni_data,
            highlight_best=True
        ),
        
        rx.box(
            rx.vstack(
                rx.heading("Architecture Details", size="5", weight="bold", margin_bottom="1em"),
                
                rx.grid(
                    rx.vstack(
                        rx.heading("MLP", size="3", weight="bold", color=rx.color("blue", 10)),
                        rx.text("Multilayer Perceptron (Feedforward)", size="2", color="var(--gray-10)", margin_bottom="0.5em"),
                        rx.unordered_list(
                            rx.list_item("Input: 13 features (all at once)"),
                            rx.list_item("Layers: 256->128->64->32->1"),
                            rx.list_item("Dropout: 0.3, 0.2 (prevent overfitting)"),
                            rx.list_item("BatchNorm: After each hidden layer"),
                            spacing="1",
                            padding_left="1em"
                        ),
                        align="start"
                    ),
                    rx.vstack(
                        rx.heading("RNN/LSTM/GRU", size="3", weight="bold", color=rx.color("purple", 10)),
                        rx.text("Recurrent Neural Networks", size="2", color="var(--gray-10)", margin_bottom="0.5em"),
                        rx.unordered_list(
                            rx.list_item("Input: Window of 12 past prices"),
                            rx.list_item("Architecture: 64->64 recurrent units"),
                            rx.list_item("Dropout: 0.2 between layers"),
                            rx.list_item("Output: Dense(32)->Dense(1)"),
                            spacing="1",
                            padding_left="1em"
                        ),
                        align="start"
                    ),
                    columns="2",
                    spacing="3",
                    width="100%"
                ),
                
                spacing="2",
                align="start"
            ),
            padding="1.5em",
            background=rx.color("gray", 1),
            border="1px solid",
            border_color=rx.color("gray", 5),
            border_radius="var(--radius-3)",
            margin_y="1em"
        ),
        
        insight_box(
            "brain",
            "Why MLP Outperforms RNN/LSTM/GRU (Univariate)?",
            "MLP uses all 13 macroeconomic features simultaneously (CPI, interest rates, S&P 500, etc.), while univariate RNN/LSTM/GRU "
            "only see past gold prices. Without economic context, recurrent models struggle to predict sudden regime changes "
            "(e.g., 2008 crisis, COVID-19). This proves gold isn't just autoregressive - it needs external features!",
            color_scheme="purple"
        ),
        
        insight_box(
            "layers",
            "LSTM vs GRU: The Gate Dilemma",
            "LSTM (R²=0.603) and RNN (R²=0.600) performed nearly identically and poorly. GRU (R²=0.843) did better by using "
            "simpler gating mechanisms (2 gates vs LSTM's 3). With limited data (univariate), LSTM's complexity became a liability. "
            "But this changes dramatically with multivariate inputs...",
            color_scheme="blue"
        ),
        
        spacing="3",
        align="start",
        width="100%",
        margin_bottom="2em"
    )


def deep_learning_multivariate() -> rx.Component:
    """Deep learning multivariate models - achieving optimal performance."""
    dl_multi_data = [
        ["GRU (Multivariate)", "0.990", "$45.92", "$34.94", "Optimal balance of performance"],
        ["LSTM (Multivariate)", "0.990", "$45.31", "$37.84", "Slightly lower MAE"],
        ["RNN (Multivariate)", "0.972", "$76.77", "$58.99", "Good but simpler architecture limits"]
    ]
    
    return rx.vstack(
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.icon("zap", size=32, color=rx.color("purple", 9)),
                    rx.heading("Deep Learning - Multivariate", size="6", weight="bold"),
                    spacing="2",
                    align="center"
                ),
                rx.text(
                    "By combining temporal patterns (12-month windows) with macroeconomic context (13 features), "
                    "multivariate recurrent models achieve highly accurate predictive fit."
                    "GRU and LSTM both reach R²=0.990, reducing average error to just $35-38.",
                    size="4",
                    color="var(--gray-12)",
                    line_height="1.7"
                ),
                spacing="2",
                align="start"
            ),
            padding="1.5em",
            background=rx.color("amber", 2),
            border_left=f"4px solid {rx.color('amber', 9)}",
            border_radius="var(--radius-3)",
            margin_bottom="1.5em"
        ),
        
        comparison_table_section(
            "Final Showdown: Multivariate Recurrent Models",
            "These models see both time patterns AND economic drivers simultaneously. "
            "Each timestep contains 13 features (CPI, interest rates, VIX, S&P 500, etc.), allowing the model to learn "
            "how gold responds to changing economic conditions over time.",
            dl_multi_data,
            highlight_best=True
        ),
        
        rx.grid(
            metric_card("R² Improvement", "+0.147", "green", "vs RNN Univariate"),
            metric_card("Error Reduction", "-79%", "blue", "MAE: $184 -> $35"),
            metric_card("Training Time", "~5 min", "purple", "70 epochs with EarlyStopping"),
            metric_card("Parameters", "~50K", "amber", "128->64 GRU units"),
            columns="4",
            spacing="3",
            width="100%",
            margin_y="1.5em"
        ),
        
        rx.box(
            rx.vstack(
                rx.heading("GRU Multivariate", size="5", weight="bold", margin_bottom="1em"),
                
                rx.grid(
                    rx.vstack(
                        rx.text.strong("Input Layer"),
                        rx.text("Shape: (batch, 12, 13)", size="2", color="var(--gray-12)"),
                        rx.text("12 timesteps x 13 features", size="2", color="var(--gray-10)"),
                        align="start"
                    ),
                    rx.icon("arrow-right", size=24, color=rx.color("gray", 8)),
                    rx.vstack(
                        rx.text.strong("GRU Layer 1"),
                        rx.text("128 units, return sequences", size="2", color="var(--gray-12)"),
                        rx.text("Dropout: 0.2", size="2", color="var(--gray-10)"),
                        align="start"
                    ),
                    rx.icon("arrow-right", size=24, color=rx.color("gray", 8)),
                    rx.vstack(
                        rx.text.strong("GRU Layer 2"),
                        rx.text("64 units, final state", size="2", color="var(--gray-12)"),
                        rx.text("Captures long-term patterns", size="2", color="var(--gray-10)"),
                        align="start"
                    ),
                    rx.icon("arrow-right", size=24, color=rx.color("gray", 8)),
                    rx.vstack(
                        rx.text.strong("Dense Layers"),
                        rx.text("Dense(32, ReLU) -> Dense(1)", size="2", color="var(--gray-12)"),
                        rx.text("Final prediction", size="2", color="var(--gray-10)"),
                        align="start"
                    ),
                    columns="7",
                    spacing="2",
                    width="100%",
                    align="center"
                ),
                
                rx.divider(margin_y="1em"),
                
                rx.heading("Training Configuration", size="4", weight="bold", margin_top="0.5em", margin_bottom="0.5em"),
                rx.grid(
                    rx.vstack(
                        rx.text.strong("Optimizer: Adam"),
                        rx.text("Adaptive learning rate", size="2", color="var(--gray-10)"),
                        align="start"
                    ),
                    rx.vstack(
                        rx.text.strong("Loss: MSE"),
                        rx.text("Mean Squared Error", size="2", color="var(--gray-10)"),
                        align="start"
                    ),
                    rx.vstack(
                        rx.text.strong("Callbacks: 3"),
                        rx.text("EarlyStopping, ReduceLR, Checkpoint", size="2", color="var(--gray-10)"),
                        align="start"
                    ),
                    rx.vstack(
                        rx.text.strong("Batch Size: 32"),
                        rx.text("Balanced speed/stability", size="2", color="var(--gray-10)"),
                        align="start"
                    ),
                    columns="4",
                    spacing="3",
                    width="100%"
                ),
                
                spacing="3",
                align="start"
            ),
            padding="1.5em",
            background=rx.color("purple", 1),
            border="1px solid",
            border_color=rx.color("purple", 5),
            border_radius="var(--radius-4)",
            margin_y="1.5em"
        ),
        
        insight_box(
            "zap",
            "Why GRU Wins Over LSTM",
            "Both achieve R²=0.990, but GRU has lower MAE ($34.94 vs $37.84) and trains ~20% faster. "
            "GRU's simpler architecture (2 gates instead of 3) is sufficient for our dataset size. "
            "LSTM's forget gate advantage doesn't materialize because our 12-month window already captures relevant history. "
            "GRU is the Goldilocks solution: complex enough to excel, simple enough to be efficient.",
            color_scheme="amber"
        ),
        
        insight_box(
            "trending-up",
            "The Multivariate Advantage",
            "Comparing univariate vs multivariate GRU: R² jumped from 0.843 -> 0.990 (+0.147), and MAE dropped from $122.95 -> $34.94 (-72%). "
            "Why? The model now understands WHY gold prices change. When CPI rises, interest rates fall, and VIX spikes, "
            "the model learned to predict gold surges - something impossible from price history alone.",
            color_scheme="green"
        ),
        
        spacing="3",
        align="start",
        width="100%",
        margin_bottom="2em"
    )


def grand_comparison() -> rx.Component:
    """Final comparison of all models."""
    all_models_data = [
        ["GRU Multivariate", "0.990", "$45.92", "$34.94", "Top performer - best overall"],
        ["LSTM Multivariate", "0.990", "$45.31", "$37.84", "Nearly tied with GRU"],
        ["SVR (RBF Kernel)", "0.986", "$59.93", "$43.77", "Best traditional ML"],
        ["Random Forest", "0.986", "$59.93", "$43.77", "Tied with SVR"],
        ["XGBoost", "0.973", "$82.67", "$51.11", "Gradient boosting"],
        ["RNN Multivariate", "0.972", "$76.77", "$58.99", "Good but simpler"],
        ["MLP", "0.960", "$100.62", "$78.85", "Feedforward baseline"],
        ["Linear Regression", "0.947", "$115.88", "$77.06", "Strong baseline"],
        ["Ridge Regression", "0.947", "$115.88", "$77.06", "No improvement"],
        ["GRU Univariate", "0.843", "$164.93", "$122.95", "Needs features"],
        ["LSTM Univariate", "0.603", "$262.55", "$193.85", "Insufficient"],
        ["RNN Univariate", "0.600", "$263.33", "$184.26", "Insufficient"],
        ["SARIMA", "0.270", "$353.57", "$233.26", "Time series weak"],
        ["ARIMA", "-0.480", "$503.12", "$321.93", "Failed completely"]
    ]
    
    return rx.vstack(
        rx.heading("Grand Comparison: All 14 Models Ranked", size="7", weight="bold", margin_bottom="1em"),
        
        rx.text(
            "Here's the complete leaderboard of all models tested, sorted by R². "
            "The progression from baseline to deep learning shows a clear trend: "
            "complexity pays off when combined with rich multivariate features and temporal modeling.",
            size="4",
            color="var(--gray-12)",
            line_height="1.7",
            margin_bottom="1.5em"
        ),
        
        rx.table.root(
            rx.table.header(
                rx.table.row(
                    rx.table.column_header_cell("Rank"),
                    rx.table.column_header_cell("Model"),
                    rx.table.column_header_cell("R²"),
                    rx.table.column_header_cell("RMSE"),
                    rx.table.column_header_cell("MAE"),
                    rx.table.column_header_cell("Category"),
                )
            ),
            rx.table.body(
                *[
                    rx.table.row(
                        rx.table.cell(str(i+1)),
                        rx.table.cell(row[0]),
                        rx.table.cell(rx.badge(row[1], color_scheme="green" if i < 2 else ("blue" if i < 5 else "gray"), size="2")),
                        rx.table.cell(row[2]),
                        rx.table.cell(row[3]),
                        rx.table.cell(row[4]),
                        style={
                            "background": rx.color("green", 2) if i < 2 else "transparent",
                            "font_weight": "bold" if i < 2 else "normal"
                        }
                    )
                    for i, row in enumerate(all_models_data)
                ]
            ),
            variant="surface",
            size="3",
            width="100%"
        ),
    ),


def key_takeaways() -> rx.Component:
    """Key learnings and insights."""
    return rx.vstack(
        rx.heading("Key Takeaways & Lessons Learned", size="7", weight="bold", margin_bottom="1em"),
        
        rx.accordion.root(
            rx.accordion.item(
                header="1. Multivariate Deep Learning Achieves Optimal Performance",
                content=rx.text(
                    "The combination of temporal modeling (RNN/LSTM/GRU) with rich macroeconomic features (CPI, interest rates, market indices) "
                    "produces the best results. GRU Multivariate achieved R²=0.990 with MAE=$34.94, outperforming all other approaches. "
                    "This validates our hypothesis that gold prices are driven by economic fundamentals rather than momentum alone.",
                    size="3",
                    line_height="1.7"
                )
            ),
            rx.accordion.item(
                header="2. Traditional ML (SVR, Random Forest) Is Highly Competitive",
                content=rx.text(
                    "SVR and Random Forest both achieved R²=0.986, nearly matching deep learning performance at a fraction of the complexity. "
                    "For production systems prioritizing interpretability and speed, these models are excellent choices. "
                    "The 1620 CV fits from GridSearchCV ensured robust hyperparameter tuning.",
                    size="3",
                    line_height="1.7"
                )
            ),
            rx.accordion.item(
                header="3. Feature Engineering > Model Complexity",
                content=rx.text(
                    "Univariate models (even sophisticated LSTM) failed (R²=0.60), while simple Linear Regression with good features achieved R²=0.947. "
                    "This proves that feature selection (Chapter 2's work removing multicollinearity, selecting 13 key variables) was more impactful "
                    "than choosing fancy algorithms. Garbage in, garbage out applies even to neural networks!",
                    size="3",
                    line_height="1.7"
                )
            ),
            rx.accordion.item(
                header="4. Time Series Methods (ARIMA/SARIMA) Don't Work for Gold",
                content=rx.text(
                    "ARIMA achieved negative R² (-0.48), meaning it performed worse than predicting the mean. SARIMA barely improved (R²=0.27). "
                    "Gold prices are driven by economic regime shifts (inflation, crises, policy changes), not autoregressive patterns. "
                    "Pure time series methods are blind to these external drivers and thus fail catastrophically.",
                    size="3",
                    line_height="1.7"
                )
            ),
            rx.accordion.item(
                header="5. GRU vs LSTM: Simplicity Wins",
                content=rx.text(
                    "Both achieved R²=0.990, but GRU had lower MAE ($34.94 vs $37.84) and faster training. "
                    "LSTM's additional forget gate didn't provide value for our dataset size and window length (12 months). "
                    "When in doubt, start with GRU - it's the sweet spot between SimpleRNN and LSTM for most financial time series.",
                    size="3",
                    line_height="1.7"
                )
            ),
            rx.accordion.item(
                header="6. Early Stopping & Regularization Prevented Overfitting",
                content=rx.text(
                    "All deep learning models used EarlyStopping (patience=15), Dropout (0.2), and ReduceLROnPlateau callbacks. "
                    "This prevented overfitting despite 200-epoch training budgets. Most models converged around 50-70 epochs. "
                    "Training/validation loss curves showed no divergence, confirming good generalization.",
                    size="3",
                    line_height="1.7"
                )
            ),
            rx.accordion.item(
                header="7. Computational Cost vs Performance Trade-off",
                content=rx.text(
                    "Linear Regression: <1 second, R²=0.947 | Random Forest: ~5 minutes, R²=0.986 | GRU Multi: ~5 minutes, R²=0.990. "
                    "The jump from Linear to RF (+0.039 R²) costs 5 minutes. RF to GRU (+0.004 R²) costs nothing extra. "
                    "Conclusion: If you're already investing in RF GridSearch, deep learning is essentially 'free' for marginal gains.",
                    size="3",
                    line_height="1.7"
                )
            ),
            rx.accordion.item(
                header="8. What Would We Do Differently?",
                content=rx.vstack(
                    rx.text(
                        "With 20/20 hindsight, here are potential improvements:",
                        size="3",
                        line_height="1.7",
                        margin_bottom="0.5em"
                    ),
                    rx.unordered_list(
                        rx.list_item("Ensemble methods: Stack top 3 models (GRU, LSTM, Random Forest) for robustness"),
                        rx.list_item("Attention mechanisms: Add attention layers to GRU to identify key timesteps"),
                        rx.list_item("Hyperparameter tuning: Use Optuna for automated Bayesian optimization"),
                        rx.list_item("Cross-validation: Implement time-series CV instead of single train/test split"),
                        rx.list_item("Regime detection: Train separate models for bull/bear/crisis periods"),
                        rx.list_item("Exogenous shocks: Add binary flags for major events (Fed pivots, crises)"),
                        spacing="2",
                        padding_left="1.5em"
                    ),
                    spacing="2",
                    align="start"
                ),
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


def whats_next() -> rx.Component:
    """Transition to forecasting chapter."""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.icon("rocket", size=32, color=rx.color("amber", 9)),
                rx.heading("What's Next: Forecasting & Deployment", size="6", weight="bold"),
                spacing="2",
                align="center"
            ),
            
            rx.text(
                "Based on comprehensive evaluation, GRU Multivariate emerges as the optimal model with R²=0.990 and MAE=$34.94. "
                "In the next chapter, we will deploy this model for real-time forecasting:",
                size="4",
                color="var(--gray-12)",
                line_height="1.7",
                margin_y="1em"
            ),
            
            rx.grid(
                rx.vstack(
                    rx.heading("Short-term Forecasts", size="4", weight="bold"),
                    rx.text("1-day, 7-day, 30-day predictions with confidence intervals", size="2", color="var(--gray-12)"),
                    align="start"
                ),
                rx.vstack(
                    rx.heading("Scenario Analysis", size="4", weight="bold"),
                    rx.text("What if Fed raises rates 2%? What if CPI hits 5%?", size="2", color="var(--gray-12)"),
                    align="start"
                ),
                rx.vstack(
                    rx.heading("Model Explainability", size="4", weight="bold"),
                    rx.text("SHAP values, feature attributions, error analysis", size="2", color="var(--gray-12)"),
                    align="start"
                ),
                rx.vstack(
                    rx.heading("Uncertainty Quantification", size="4", weight="bold"),
                    rx.text("Prediction intervals, Monte Carlo dropout, ensemble variance", size="2", color="var(--gray-12)"),
                    align="start"
                ),
                columns="2",
                spacing="3",
                width="100%",
                margin_y="1em"
            ),
            
            rx.text(
                "We'll also compare our results against published research and industry benchmarks to validate "
                "that our R²=0.990 achievement represents genuine state-of-the-art performance.",
                size="3",
                color="var(--gray-12)",
                line_height="1.7",
                margin_top="1em"
            ),
            
            spacing="3",
            align="start"
        ),
        padding="1.5em",
        background=rx.color("amber", 2),
        border_left=f"4px solid {rx.color('amber', 9)}",
        border_radius="var(--radius-3)",
        margin_y="2em"
    )


# ======================================================================
# MAIN PAGE FUNCTION
# ======================================================================

def modeling_page() -> rx.Component:
    """Chapter 3: Modeling & Evaluation page."""
    
    return page_layout(
        rx.flex(
            rx.vstack(
                chapter_progress(current=3),
                
                rx.vstack(
                    rx.heading(
                        "Chapter 3: Modeling & Evaluation",
                        size="8",
                        weight="bold",
                        color_scheme="purple",
                        align="center"
                    ),
                    rx.heading(
                        "From Statistics to Deep Learning",
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
                
                modeling_philosophy(),
                section_divider(),
                
                univariate_regression_detail(),
                section_divider(),
                
                polynomial_regression_section(),
                section_divider(),
                
                time_series_section(),
                section_divider(),
                
                multivariate_regression_detail(),
                section_divider(),
                
                baseline_models(),
                section_divider(),
                
                traditional_ml(),
                section_divider(),
                
                deep_learning_univariate(),
                section_divider(),
                
                deep_learning_multivariate(),
                section_divider(),
                
                grand_comparison(),
                section_divider(),
                
                key_takeaways(),
                section_divider(),
                
                whats_next(),
                
                rx.flex(
                    rx.link(
                        rx.button(
                            "Next: Chapter 4 - Forecasting ➔",
                            size="3",
                            color_scheme="purple",
                            variant="solid"
                        ),
                        href="/forecast",
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
