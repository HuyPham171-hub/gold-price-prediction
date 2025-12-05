"""Chapter 3: Modeling & Evaluation - From Statistics to Deep Learning"""

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
                        rx.text.strong("simple vs Multiple: "),
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
            "Rather than jumping straight to complex solutions, we follow a ",
            rx.text.strong("systematic approach"),
            ": start with ",
            rx.text.strong("simple baselines"),
            ", understand their limitations, then ",
            rx.text.strong("progressively add complexity"),
            ". This ensures we understand what each model contributes and ",
            rx.text.strong("avoid unnecessary sophistication"),
            ".",
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


def simple_regression_detail() -> rx.Component:
    """Detailed simple regression results."""
    simple_results = [
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
        rx.heading("Simple Linear Regression: Testing Each Feature", size="6", weight="bold", margin_bottom="1em"),
        
        rx.text(
            "Before building Multiple models, we tested each of the 13 features individually to understand "
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
                                for i, row in enumerate(simple_results)
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
                # Top 9 Features Visualization
                rx.box(
                    rx.vstack(
                        rx.heading("Top 9 Features Performance", size="5", weight="bold", margin_bottom="1em"),
                        rx.image(
                            src="/modeling_plots/simple/top_9_polynomial.png",
                            width="100%",
                            border_radius="var(--radius-3)",
                            border="1px solid",
                            border_color=rx.color("gray", 5)
                        ),
                        spacing="2",
                        align="start"
                    ),
                    padding="1.5em",
                    background=rx.color("gray", 1),
                    border="1px solid",
                    border_color=rx.color("gray", 5),
                    border_radius="var(--radius-4)",
                    margin_bottom="1.5em"
                ),
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
                                    rx.text("Supply shocks create noise. Works better in Multiple context.", size="2", color="var(--gray-12)"),
                                    align="start", spacing="1"
                                ),
                                rx.vstack(
                                    rx.text.strong("Unemployment (R² = -0.002)", color=rx.color("red", 10)),
                                    rx.text("Indirect effect through Fed policy. Non-linear relationship.", size="2", color="var(--gray-12)"),
                                    align="start", spacing="1"
                                ),
                                rx.vstack(
                                    rx.text.strong("Fed Funds (R² = -0.043)", color=rx.color("red", 10)),
                                    rx.text("Multiple channels with lags. Needs Multiple context.", size="2", color="var(--gray-12)"),
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
                                rx.heading("Solution: Multiple Models", size="4", weight="bold"),
                                spacing="2",
                                align="center",
                                margin_bottom="0.5em"
                            ),
                            rx.text(
                                "These 'weak' features become valuable in Multiple models through interactions. "
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


def Multiple_regression_detail() -> rx.Component:
    """OLS regression with statistical details."""
    return rx.vstack(
        rx.heading("Multiple Linear Regression: Combining All Features", size="6", weight="bold", margin_bottom="1em"),
        
        rx.text(
            "Now we use all 13 features simultaneously. This allows the model to capture interactions between variables "
            "(e.g., inflation and interest rates together affecting gold). Results: ",
            rx.text.strong("R² = 0.947"), ", ",
            rx.text.strong("RMSE = $115.88"), ", ",
            rx.text.strong("MAE = $77.06"), ".",
            size="4",
            color="var(--gray-12)",
            line_height="1.7",
            margin_bottom="1.5em"
        ),
        
        rx.grid(
            metric_card("R²", "0.947", "green", "95% variance explained"),
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
                                    "VIX, interest rates, geopolitical indices (redundant in Multiple context)"
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
                                    "F-statistic = ",
                                    rx.text.strong("312.9"),
                                    " confirms the model explains variance ",
                                    rx.text.strong("significantly better than the null model")
                                ),
                                rx.list_item(
                                    rx.text.strong("No autocorrelation: "),
                                    "Durbin-Watson = ",
                                    rx.text.strong("2.221 ~ 2.0"),
                                    " means ",
                                    rx.text.strong("residuals are independent"),
                                    " (good for regression assumptions)"
                                ),
                                rx.list_item(
                                    rx.text.strong("Non-normal residuals: "),
                                    "Skew = ",
                                    rx.text.strong("0.928"),
                                    ", Kurtosis = ",
                                    rx.text.strong("10.38"),
                                    " indicate ",
                                    rx.text.strong("heavy-tailed distribution"),
                                    ". This affects t-test/F-test reliability."
                                ),
                                rx.list_item(
                                    rx.text.strong("Multicollinearity present: "),
                                    "Condition number = ",
                                    rx.text.strong("99,000"),
                                    " suggests some features are highly correlated (e.g., CPI <-> M2, S&P <-> NASDAQ removed earlier)"
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

                    # Diagnostic Plots Visualization
                    rx.box(
                        rx.vstack(
                            rx.heading("Diagnostic Plots (3-Panel)", size="5", weight="bold", margin_bottom="1em"),
                            rx.image(
                                src="/modeling_plots/Multiple/diagnostics_3panel.png",
                                width="100%",
                                border_radius="var(--radius-3)",
                                border="1px solid",
                                border_color=rx.color("gray", 5)
                            ),
                            rx.text(
                                "(a) Predicted vs Actual shows strong linear fit with R²=0.947. "
                                "(b) Residuals vs Predicted reveals some heteroscedasticity (wider spread at extremes). "
                                "(c) Residual Distribution shows positive skew and heavy tails (non-normal).",
                                size="2",
                                color="var(--gray-11)",
                                line_height="1.6",
                                margin_top="0.5em"
                            ),
                            spacing="2",
                            align="start"
                        ),
                        padding="1.5em",
                        background=rx.color("blue", 1),
                        border="1px solid",
                        border_color=rx.color("blue", 5),
                        border_radius="var(--radius-4)",
                        margin_top="1.5em"
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
                                    "Applied L2 regularization showed no improvement (",
                                    rx.text.strong("R² = 0.947"),
                                    " identical to OLS), indicating that while multicollinearity exists, it does not significantly degrade predictive performance"
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
    ),


def polynomial_regression_section() -> rx.Component:
    """Polynomial regression - brief section."""
    return rx.vstack(
        rx.heading("Polynomial Regression: Testing Non-linear Curves", size="6", weight="bold", margin_bottom="1em"),
        
        rx.text(
            "Can we improve simple predictions by fitting curves instead of straight lines? "
            "We tested polynomial regression (degree=2) on each feature to capture non-linear relationships.",
            size="4",
            color="var(--gray-12)",
            line_height="1.7",
            margin_bottom="1em"
        ),
        
        # Visualization: Linear vs Polynomial Comparison
        rx.box(
            rx.vstack(
                rx.heading("Linear vs Polynomial Fit: Silver Futures", size="5", weight="bold", margin_bottom="1em"),
                rx.image(
                    src="/modeling_plots/polynomial/silver_compare_polynomial.png",
                    width="100%",
                    border_radius="var(--radius-3)",
                    border="1px solid",
                    border_color=rx.color("gray", 5)
                ),
                rx.text(
                    "Comparing linear (degree 1) and polynomial (degree 2) fits for Silver Futures. "
                    "The polynomial curve captures slight non-linearity but improves R² by only 1%.",
                    size="2",
                    color="var(--gray-11)",
                    text_align="center",
                    margin_top="0.5em"
                ),
                spacing="2",
                align="start"
            ),
            padding="1.5em",
            background=rx.color("gray", 1),
            border="1px solid",
            border_color=rx.color("gray", 5),
            border_radius="var(--radius-4)",
            margin_bottom="1.5em"
        ),
        
        rx.box(
            rx.vstack(
                rx.heading("Best Result: Silver (R² = 0.537)", size="5", weight="bold", margin_bottom="0.75em", color=rx.color("green", 10)),
                rx.text(
                    "Polynomial regression on Silver Futures achieves ",
                    rx.text.strong("R² = 0.537"),
                    " versus linear ",
                    rx.text.strong("R² = 0.526"),
                    " - a marginal 1% improvement. "
                    "CPI polynomial performs worse (",
                    rx.text.strong("R² = 0.698"),
                    " versus linear ",
                    rx.text.strong("0.720"),
                    "). Most features show no benefit from polynomial transformation.",
                    size="3",
                    color="var(--gray-12)",
                    line_height="1.7",
                    margin_bottom="1em"
                ),
                rx.heading("Verdict: Curves Don't Help", size="5", weight="bold", margin_bottom="0.5em"),
                rx.text(
                    rx.text.strong("Gold-feature relationships are approximately linear"),
                    ". Adding polynomial terms creates ",
                    rx.text.strong("overfitting risk"),
                    " without meaningful performance gain. ",
                    rx.text.strong("Multiple linear models remain the better path"),
                    ".",
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


def ridge_regression_detail() -> rx.Component:
    """Ridge Regression - L2 Regularization for multicollinearity."""
    
    # Metrics comparison data
    metrics_data = [
        ["Linear Regression", "0.947", "0.928", "13427.07", "115.88", "77.06"],
        ["Ridge Regression", "0.947", "0.928", "13427.07", "115.88", "77.06"]
    ]
    
    # Coefficients comparison data
    coef_data = [
        ["Silver_Futures", "25.49", "25.79"],
        ["Unemployment", "32.04", "24.39"],
        ["CPI", "10.20", "10.73"],
        ["VIX", "1.51", "2.22"],
        ["S&P_500", "0.10", "0.09"],
        ["GPR", "0.22", "0.06"],
        ["GPRA", "0.08", "0.04"],
        ["Crude_Oil", "-2.20", "-2.86"],
        ["Fed_Funds_Rate", "5.12", "-4.88"],
        ["Real_Interest_Rate", "24.05", "-5.09"],
        ["Treasury_Yield_10Y", "-52.38", "-8.13"],
        ["USD_Index", "-7.84", "-8.35"]
    ]
    
    return rx.vstack(
        rx.heading("Ridge Regression: L2 Regularization Test", size="6", weight="bold", margin_bottom="1em"),
        
        rx.text(
            "With severe multicollinearity detected (VIF > 100 for CPI, USD Index, Treasury Yield), "
            "we tested ",
            rx.text.strong("Ridge Regression (L2 regularization)"),
            " to stabilize coefficient estimates. Ridge penalizes large coefficients, ",
            rx.text.strong("shrinking correlated features toward zero"),
            " while preserving predictive performance.",
            size="4",
            color="var(--gray-12)",
            line_height="1.7",
            margin_bottom="1em"
        ),
        
        # Metrics Comparison Table
        rx.box(
            rx.vstack(
                rx.heading("Performance Comparison: Linear vs Ridge", size="5", weight="bold", margin_bottom="1em"),
                
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.table.column_header_cell("Model"),
                            rx.table.column_header_cell("R²"),
                            rx.table.column_header_cell("Adj R²"),
                            rx.table.column_header_cell("MSE"),
                            rx.table.column_header_cell("RMSE"),
                            rx.table.column_header_cell("MAE"),
                        )
                    ),
                    rx.table.body(
                        *[
                            rx.table.row(
                                rx.table.cell(row[0]),
                                rx.table.cell(rx.badge(row[1], color_scheme="green", size="2")),
                                rx.table.cell(rx.badge(row[2], color_scheme="green", size="2")),
                                rx.table.cell(row[3]),
                                rx.table.cell(row[4]),
                                rx.table.cell(row[5]),
                            )
                            for row in metrics_data
                        ]
                    ),
                    variant="surface",
                    size="3",
                    width="100%",
                    margin_bottom="1em"
                ),
                
                rx.text(
                    rx.text.strong("Observation: "),
                    "Ridge produces ",
                    rx.text.strong("identical metrics"),
                    " to OLS Linear Regression (R² = 0.947, RMSE = $115.88). "
                    "This indicates that while multicollinearity exists in the data, ",
                    rx.text.strong("it does not degrade predictive performance"),
                    ". "
                    "The regularization constraint had no impact on test set predictions.",
                    size="3",
                    color="var(--gray-11)",
                    line_height="1.6",
                    margin_top="0.5em"
                ),
                
                spacing="2",
                align="start"
            ),
            padding="1.5em",
            background=rx.color("gray", 1),
            border="1px solid",
            border_color=rx.color("gray", 5),
            border_radius="var(--radius-4)",
            margin_bottom="1.5em"
        ),
        
        # Coefficients Comparison Table
        rx.box(
            rx.vstack(
                rx.heading("Coefficient Comparison: Regularization Effects", size="5", weight="bold", margin_bottom="1em"),
                
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.table.column_header_cell("Feature"),
                            rx.table.column_header_cell("Linear Coefficient"),
                            rx.table.column_header_cell("Ridge Coefficient"),
                        )
                    ),
                    rx.table.body(
                        *[
                            rx.table.row(
                                rx.table.cell(row[0]),
                                rx.table.cell(row[1]),
                                rx.table.cell(row[2]),
                            )
                            for row in coef_data
                        ]
                    ),
                    variant="surface",
                    size="3",
                    width="100%",
                    margin_bottom="1em"
                ),
                
                rx.vstack(
                    rx.hstack(
                        rx.icon("lightbulb", size=20, color=rx.color("purple", 9)),
                        rx.heading("Coefficient Interpretation", size="4", weight="bold"),
                        spacing="2",
                        align="center",
                        margin_bottom="0.5em"
                    ),
                    rx.unordered_list(
                        rx.list_item(
                            rx.text.strong("Silver_Futures (25.49→25.79): "),
                            "Minimal change (+1.2%) indicates a ",
                            rx.text.strong("stable, independent predictor"),
                            " with low multicollinearity"
                        ),
                        rx.list_item(
                            rx.text.strong("Unemployment (32.04→24.39): "),
                            "Moderate shrinkage (-23.9%) shows some correlation with macro variables but remains positive (safe haven effect)"
                        ),
                        rx.list_item(
                            rx.text.strong("CPI (10.20→10.73): "),
                            "Slight increase (+5.2%) - regularization stabilizes coefficient while maintaining inflation hedge relationship"
                        ),
                        rx.list_item(
                            rx.text.strong("VIX (1.51→2.22): "),
                            "Increases by 47% under Ridge - originally suppressed by multicollinearity, now reveals stronger volatility signal"
                        ),
                        rx.list_item(
                            rx.text.strong("Sign flips (red flags): "),
                            rx.text.strong("Fed_Funds_Rate (5.12→-4.88)"),
                            " and ",
                            rx.text.strong("Real_Interest_Rate (24.05→-5.09)"),
                            " reverse direction. This indicates extreme multicollinearity - coefficients are unstable and unreliable"
                        ),
                        rx.list_item(
                            rx.text.strong("Treasury_Yield_10Y (-52.38→-8.13): "),
                            "Massive 84.5% shrinkage confirms severe multicollinearity (VIF=179.92) with interest rate variables"
                        ),
                        rx.list_item(
                            rx.text.strong("USD_Index (-7.84→-8.35): "),
                            "Small increase in magnitude (+6.5%) shows stable negative relationship with gold prices"
                        ),
                        spacing="2",
                        padding_left="1.5em"
                    ),
                    spacing="2",
                    align="start"
                ),
                
                spacing="3",
                align="start"
            ),
            padding="1.5em",
            background=rx.color("purple", 1),
            border="1px solid",
            border_color=rx.color("purple", 5),
            border_radius="var(--radius-4)",
            margin_bottom="1.5em"
        ),
        
        # Final Verdict
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.icon("info", size=24, color=rx.color("cyan", 9)),
                    rx.heading("Decision: Retain Linear Regression", size="4", weight="bold"),
                    spacing="2",
                    align="center",
                    margin_bottom="0.5em"
                ),
                rx.text(
                    "Despite severe multicollinearity, ",
                    rx.text.strong("Ridge regularization provides no predictive benefit"),
                    " (identical R² and RMSE). The coefficient instability (sign flips, shrinkage) confirms multicollinearity exists, but ",
                    rx.text.strong("it does not harm generalization performance"),
                    ". "
                    "This occurs because correlated features (CPI, USD Index, Treasury Yield, Real Interest Rate) capture ",
                    rx.text.strong("genuine economic relationships"),
                    " - their correlation reflects reality, not noise. "
                    "Removing features would sacrifice information without improving predictions.",
                    size="3",
                    color="var(--gray-12)",
                    line_height="1.7",
                    margin_bottom="1em"
                ),
                rx.text(
                    rx.text.strong("Modeling decision: "),
                    "We retain all 12 features in standard Linear Regression. "
                    "The multicollinearity reflects the interconnected nature of macroeconomic variables (e.g., inflation drives both CPI and interest rates). "
                    "Regularization shrinks coefficients but does not improve out-of-sample accuracy.",
                    size="3",
                    color="var(--gray-12)",
                    line_height="1.7"
                ),
                spacing="2",
                align="start"
            ),
            padding="1.25em",
            background=rx.color("cyan", 2),
            border_left=f"4px solid {rx.color('cyan', 9)}",
            border_radius="var(--radius-3)",
            margin_top="1em"
        ),
        
        spacing="3",
        align="start",
        width="100%",
        margin_bottom="2em"
    )


def time_series_section() -> rx.Component:
    """ARIMA/SARIMA with detailed parameter explanations."""
    return rx.vstack(
        rx.heading("Time Series Models: ARIMA and SARIMA", size="6", weight="bold", margin_bottom="1em"),
        
        rx.text(
            "Can we predict gold prices using only historical patterns without external economic data? "
            "We tested traditional time series models (ARIMA and SARIMA) that rely solely on past gold price values and their temporal structure.",
            size="4",
            color="var(--gray-12)",
            line_height="1.7",
            margin_bottom="1.5em"
        ),
        
        # Explanation of ARIMA/SARIMA parameters
        rx.box(
            rx.vstack(
                rx.heading("Understanding ARIMA and SARIMA Parameters", size="5", weight="bold", margin_bottom="1em"),
                
                rx.heading("ARIMA (p, d, q): AutoRegressive Integrated Moving Average", size="4", weight="bold", margin_bottom="0.75em", color=rx.color("blue", 10)),
                rx.grid(
                    rx.vstack(
                        rx.text.strong("p (AutoRegressive order)"),
                        rx.text(
                            "Number of past time steps (lags) used to predict the current value. "
                            "For ARIMA(1,1,1), p=1 means the model uses the previous month's price to predict the next month.",
                            size="2",
                            color="var(--gray-12)",
                            line_height="1.6"
                        ),
                        align="start",
                        spacing="1"
                    ),
                    rx.vstack(
                        rx.text.strong("d (Differencing order)"),
                        rx.text(
                            "Number of times the series is differenced to make it stationary (remove trends). "
                            "d=1 means we model the change in gold price (first derivative) rather than raw prices.",
                            size="2",
                            color="var(--gray-12)",
                            line_height="1.6"
                        ),
                        align="start",
                        spacing="1"
                    ),
                    rx.vstack(
                        rx.text.strong("q (Moving Average order)"),
                        rx.text(
                            "Number of past forecast errors used to correct predictions. "
                            "q=1 means the model learns from the previous prediction error to improve the next forecast.",
                            size="2",
                            color="var(--gray-12)",
                            line_height="1.6"
                        ),
                        align="start",
                        spacing="1"
                    ),
                    columns="3",
                    spacing="3",
                    width="100%",
                    margin_bottom="1.5em"
                ),
                
                rx.heading("SARIMA (p,d,q)x(P,D,Q,s): Seasonal ARIMA", size="4", weight="bold", margin_bottom="0.75em", color=rx.color("purple", 10)),
                rx.text(
                    "SARIMA extends ARIMA by adding seasonal components. Our model SARIMA(1,1,1)x(1,1,1,12) includes:",
                    size="3",
                    color="var(--gray-12)",
                    line_height="1.7",
                    margin_bottom="0.75em"
                ),
                rx.grid(
                    rx.vstack(
                        rx.text.strong("P (Seasonal AR)"),
                        rx.text("Uses prices from 12 months ago (P=1)", size="2", color="var(--gray-12)"),
                        align="start",
                        spacing="1"
                    ),
                    rx.vstack(
                        rx.text.strong("D (Seasonal Differencing)"),
                        rx.text("Removes yearly trends (D=1)", size="2", color="var(--gray-12)"),
                        align="start",
                        spacing="1"
                    ),
                    rx.vstack(
                        rx.text.strong("Q (Seasonal MA)"),
                        rx.text("Corrects using errors from 12 months ago (Q=1)", size="2", color="var(--gray-12)"),
                        align="start",
                        spacing="1"
                    ),
                    rx.vstack(
                        rx.text.strong("s (Seasonal period)"),
                        rx.text("12 months - captures annual patterns", size="2", color="var(--gray-12)"),
                        align="start",
                        spacing="1"
                    ),
                    columns="4",
                    spacing="3",
                    width="100%"
                ),
                spacing="3",
                align="start"
            ),
            padding="1.5em",
            background=rx.color("blue", 1),
            border="1px solid",
            border_color=rx.color("blue", 5),
            border_radius="var(--radius-4)",
            margin_bottom="1.5em"
        ),
        
        # Results comparison
        rx.heading("Model Results", size="5", weight="bold", margin_bottom="1em"),
        rx.grid(
            rx.box(
                rx.vstack(
                    rx.heading("ARIMA (1,1,1)", size="5", weight="bold", margin_bottom="0.5em"),
                    rx.heading("R² = -0.480", size="6", weight="bold", color=rx.color("red", 10), margin_bottom="0.5em"),
                    rx.text(
                        rx.text.strong("RMSE: "), "$503.12 | ",
                        rx.text.strong("MAE: "), "$321.93",
                        size="2",
                        color="var(--gray-11)",
                        margin_bottom="0.75em"
                    ),
                    rx.text(
                        rx.text.strong("Negative R²"),
                        " indicates the model performs ",
                        rx.text.strong("worse than simply predicting the mean"),
                        " gold price. "
                        "Autoregressive patterns alone cannot explain gold price movements. ",
                        rx.text.strong("Gold responds to economic events, not just past values"),
                        ".",
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
                    rx.heading("R² = 0.270", size="6", weight="bold", color=rx.color("orange", 10), margin_bottom="0.5em"),
                    rx.text(
                        rx.text.strong("RMSE: "), "$353.57 | ",
                        rx.text.strong("MAE: "), "$233.26",
                        size="2",
                        color="var(--gray-11)",
                        margin_bottom="0.75em"
                    ),
                    rx.text(
                        rx.text.strong("Seasonal component improves performance"),
                        " substantially but still explains ",
                        rx.text.strong("only 27% of variance"),
                        ". Monthly seasonality exists but remains ",
                        rx.text.strong("weak compared to macroeconomic drivers"),
                        " (CPI alone explains 72%).",
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
        
        # ARIMA Forecast Visualization
        rx.box(
            rx.vstack(
                rx.heading("ARIMA Forecast vs Actual Gold Prices", size="5", weight="bold", margin_bottom="1em"),
                rx.image(
                    src="/modeling_plots/arima/arima_forcast.png",
                    width="100%",
                    border_radius="var(--radius-3)",
                    border="1px solid",
                    border_color=rx.color("gray", 5)
                ),
                rx.text(
                    "Train series (blue) shows historical patterns. SARIMA forecast (green) diverges from actual test values (black), "
                    "demonstrating the model's inability to capture gold price movements driven by external economic factors.",
                    size="2",
                    color="var(--gray-11)",
                    line_height="1.6",
                    margin_top="0.5em"
                ),
                spacing="2",
                align="start"
            ),
            padding="1.5em",
            background=rx.color("gray", 1),
            border="1px solid",
            border_color=rx.color("gray", 5),
            border_radius="var(--radius-4)",
            margin_top="1.5em"
        ),
        
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.icon("triangle-alert", size=24, color=rx.color("red", 9)),
                    rx.heading("Why Time Series Models Failed", size="4", weight="bold"),
                    spacing="2",
                    align="center",
                    margin_bottom="0.5em"
                ),
                rx.text(
                    "Time series models assume future values depend primarily on past values and temporal patterns. ",
                    rx.text.strong("Gold prices violate this assumption"),
                    " because they are ",
                    rx.text.strong("fundamentally driven by external economic factors"),
                    ": inflation expectations, interest rate policy, currency strength, and geopolitical risk. ",
                    rx.text.strong("Historical patterns alone miss these fundamental drivers"),
                    ". This limitation provides a clear hypothesis: "
                    "we need ",
                    rx.text.strong("Multiple models that can incorporate external economic data"),
                    " to capture what truly drives gold prices.",
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
        ["Linear Regression", "0.947", "$115.88", "$77.06", "Strong Multiple baseline"],
        ["Ridge Regression", "0.947", "$115.88", "$77.06", "No improvement"],
        ["Polynomial (degree=2)", "0.537", "$342.78", "$270.69", "Best simple: Silver"],
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
                    rx.heading("Best Baseline: Multiple Linear Regression", size="4", weight="bold"),
                    spacing="2",
                    align="center",
                    margin_bottom="0.5em"
                ),
                rx.text(
                    rx.text.strong("R²=0.947"),
                    " indicates 95% of gold price variance explained. The Multiple approach combining inflation, "
                    "interest rates, stock market, and currency data substantially outperforms all simple and time series methods. "
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
        ["XGBoost", "0.973", "$82.67", "$51.11", "Slightly underperformed than others"]
    ]
    
    return rx.vstack(
        comparison_table_section(
            "Traditional Machine Learning: Non-Linear Methods",
            "Moving beyond linear assumptions, we test kernel-based (SVR) and tree-based (Random Forest, XGBoost) methods. "
            "These models can capture non-linear relationships and feature interactions.",
            ml_data
        ),
        
        # Key highlights
        rx.box(
            rx.text(
                "Both SVR and Random Forest achieve ",
                rx.text.strong("R² = 0.986"),
                ", a significant jump from the baseline, reducing error by ",
                rx.text.strong("nearly 50%"),
                ".",
                size="3",
                color="var(--gray-12)",
                line_height="1.7"
            ),
            padding="1em",
            background=rx.color("green", 2),
            border_left=f"4px solid {rx.color('green', 9)}",
            border_radius="var(--radius-3)",
            margin_bottom="1em"
        ),
        
        # SVR Section
        rx.heading("Support Vector Regression (SVR)", size="5", weight="bold", margin_bottom="1em"),
        rx.box(
            rx.vstack(
                rx.text(
                    "Support Vector Regression with RBF kernel maps features into high-dimensional space, "
                    "capturing complex non-linear patterns. GridSearchCV tested 27 combinations to find optimal hyperparameters.",
                    size="3",
                    color="var(--gray-12)",
                    line_height="1.6",
                    margin_bottom="0.75em"
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
            border_radius="var(--radius-3)",
            margin_bottom="1.5em"
        ),
        
        # SVR Visualizations
        rx.grid(
            rx.box(
                rx.vstack(
                    rx.heading("SVR: Predicted vs Actual", size="4", weight="bold", margin_bottom="0.5em"),
                    rx.image(
                        src="/modeling_plots/svr/pred_vs_actual.png",
                        width="100%",
                        border_radius="var(--radius-3)",
                        border="1px solid",
                        border_color=rx.color("gray", 5)
                    ),
                    rx.text(
                        "Tight clustering around diagonal line (y=x) demonstrates SVR's ability to capture non-linear patterns with R²=0.986.",
                        size="2",
                        color="var(--gray-11)",
                        line_height="1.6",
                        margin_top="0.5em"
                    ),
                    spacing="2",
                    align="start"
                ),
                padding="1.25em",
                background=rx.color("gray", 1),
                border="1px solid",
                border_color=rx.color("gray", 5),
                border_radius="var(--radius-3)"
            ),
            
            rx.box(
                rx.vstack(
                    rx.heading("SVR: Residual Distribution", size="4", weight="bold", margin_bottom="0.5em"),
                    rx.image(
                        src="/modeling_plots/svr/residuals_dist.png",
                        width="100%",
                        border_radius="var(--radius-3)",
                        border="1px solid",
                        border_color=rx.color("gray", 5)
                    ),
                    rx.text(
                        "Residuals are approximately normal with small variance, confirming reliable model performance across price ranges.",
                        size="2",
                        color="var(--gray-11)",
                        line_height="1.6",
                        margin_top="0.5em"
                    ),
                    spacing="2",
                    align="start"
                ),
                padding="1.25em",
                background=rx.color("gray", 1),
                border="1px solid",
                border_color=rx.color("gray", 5),
                border_radius="var(--radius-3)"
            ),
            
            columns="2",
            spacing="3",
            width="100%",
            margin_bottom="2em"
        ),
        
        # Random Forest Section
        rx.heading("Random Forest", size="5", weight="bold", margin_bottom="1em"),
        rx.box(
            rx.vstack(
                rx.text(
                    "Random Forest trains 500 decision trees on random subsets of features, then averages predictions. "
                    "This ensemble approach reduces overfitting while capturing non-linear patterns.",
                    size="3",
                    color="var(--gray-12)",
                    line_height="1.6",
                    margin_bottom="0.75em"
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
            border_radius="var(--radius-3)",
            margin_bottom="1.5em"
        ),
        
        # Random Forest Feature Importance
        rx.box(
            rx.vstack(
                rx.heading("Random Forest Feature Importance", size="4", weight="bold", margin_bottom="1em"),
                rx.image(
                    src="/modeling_plots/random_forest/feature_importance.png",
                    width="100%",
                    border_radius="var(--radius-3)",
                    border="1px solid",
                    border_color=rx.color("gray", 5)
                ),
                rx.text(
                    "Top 3 features highlighted in gold: Silver Futures, CPI (inflation), and S&P 500. "
                    "Tree-based models confirm macroeconomic drivers identified in simple analysis.",
                    size="2",
                    color="var(--gray-11)",
                    line_height="1.6",
                    margin_top="0.5em"
                ),
                spacing="2",
                align="start"
            ),
            padding="1.5em",
            background=rx.color("gray", 1),
            border="1px solid",
            border_color=rx.color("gray", 5),
            border_radius="var(--radius-4)",
            margin_bottom="1.5em"
        ),
        
        rx.box(
            rx.text(
                "XGBoost (",
                rx.text.strong("R² = 0.973"),
                ") surprisingly performed worse than SVR and Random Forest. This suggests the model may be overfitting "
                "to training data despite regularization. XGBoost's strength lies in tabular data with complex interactions, but our "
                "relatively clean dataset with strong linear trends may not benefit as much from gradient boosting's aggressive optimization.",
                size="3",
                color="var(--gray-12)",
                line_height="1.7"
            ),
            padding="1.25em",
            background=rx.color("amber", 2),
            border_left=f"4px solid {rx.color('amber', 9)}",
            border_radius="var(--radius-3)"
        ),
        
        spacing="3",
        align="start",
        width="100%",
        margin_bottom="2em"
    )


def deep_learning_simple() -> rx.Component:
    """Deep learning simple models."""
    dl_uni_data = [
        ["MLP (Feedforward)", "0.960", "$100.62", "$78.85", "256->128->64->32 neurons, Dropout"],
        ["GRU (One - One)", "0.843", "$164.93", "$122.95", "64->64 units, window=12"],
        ["LSTM (One - One)", "0.603", "$262.55", "$193.85", "64->64 units, gates struggle"],
        ["RNN (One - One)", "0.600", "$263.33", "$184.26", "Simple RNN insufficient"]
    ]
    
    return rx.vstack(
        comparison_table_section(
            "Deep Learning - (One - One) (Gold Price Only)",
            "Before using all 13 features, we test if deep learning can extract temporal patterns from gold price history alone. "
            "MLP (feedforward) performs well as it uses all features but no sequence. "
            "Recurrent models (RNN/LSTM/GRU) use sliding windows of past prices but struggle without external features.",
            dl_uni_data
            # highlight_best=True
        ),
        
        # Highlight key metrics
        rx.box(
            rx.text(
                "MLP achieves ",
                rx.text.strong("R² = 0.960"),
                " while recurrent models struggle with only ",
                rx.text.strong("R² = 0.60-0.84"),
                " when using gold price history alone.",
                size="3",
                color="var(--gray-12)",
                line_height="1.7"
            ),
            padding="1em",
            background=rx.color("blue", 2),
            border_left=f"4px solid {rx.color('blue', 9)}",
            border_radius="var(--radius-3)",
            margin_bottom="1em"
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
        
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.icon("brain", size=24, color=rx.color("purple", 9)),
                    rx.heading("Why MLP Outperforms RNN/LSTM/GRU (simple)?", size="4", weight="bold"),
                    spacing="2",
                    align="center"
                ),
                rx.text(
                    "MLP uses ",
                    rx.text.strong("all 13 macroeconomic features"),
                    " simultaneously (CPI, interest rates, S&P 500, etc.), while simple RNN/LSTM/GRU "
                    "only see past gold prices. Without economic context, recurrent models struggle to predict sudden regime changes "
                    "(e.g., 2008 crisis, COVID-19). This proves gold isn't just autoregressive - it ",
                    rx.text.strong("needs external features"),
                    "!",
                    size="3",
                    color="var(--gray-12)",
                    line_height="1.7"
                ),
                spacing="2",
                align="start"
            ),
            padding="1.25em",
            background=rx.color("purple", 2),
            border_left=f"4px solid {rx.color('purple', 9)}",
            border_radius="var(--radius-3)",
            margin_bottom="1em"
        ),
        
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.icon("layers", size=24, color=rx.color("blue", 9)),
                    rx.heading("LSTM vs GRU: The Gate Dilemma", size="4", weight="bold"),
                    spacing="2",
                    align="center"
                ),
                rx.text(
                    "LSTM (",
                    rx.text.strong("R² = 0.603"),
                    ") and RNN (",
                    rx.text.strong("R² = 0.600"),
                    ") performed nearly identically and poorly. GRU (",
                    rx.text.strong("R² = 0.843"),
                    ") did better by using "
                    "simpler gating mechanisms (",
                    rx.text.strong("2 gates vs LSTM's 3"),
                    "). With limited data (simple), LSTM's complexity became a liability. "
                    "But this changes dramatically with Multiple inputs...",
                    size="3",
                    color="var(--gray-12)",
                    line_height="1.7"
                ),
                spacing="2",
                align="start"
            ),
            padding="1.25em",
            background=rx.color("blue", 2),
            border_left=f"4px solid {rx.color('blue', 9)}",
            border_radius="var(--radius-3)",
            margin_bottom="1.5em"
        ),
        
        # Model Visualizations - Tabbed Interface
        rx.heading("Model Training & Performance Visualizations", size="5", weight="bold", margin_bottom="1em", margin_top="1em"),
        
        rx.tabs.root(
            rx.tabs.list(
                rx.tabs.trigger("MLP", value="mlp"),
                rx.tabs.trigger("RNN", value="rnn"),
                rx.tabs.trigger("LSTM", value="lstm"),
                rx.tabs.trigger("GRU", value="gru"),
            ),
            
            # MLP Tab
            rx.tabs.content(
                rx.box(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("layers", size=24, color=rx.color("blue", 9)),
                            rx.heading("MLP (Multilayer Perceptron) - R² = 0.960", size="4", weight="bold", color=rx.color("blue", 10)),
                            spacing="2",
                            align="center",
                            margin_bottom="0.5em"
                        ),
                        rx.text(
                            "Best simple performer using all 13 features simultaneously (256→128→64→32 architecture with Dropout & BatchNorm).",
                            size="2",
                            color="var(--gray-11)",
                            margin_bottom="1em"
                        ),
                        
                        # Training History
                        rx.vstack(
                            rx.heading("Training History", size="3", weight="bold", margin_bottom="0.5em"),
                            rx.image(
                                src="/modeling_plots/mlp/training_history.png",
                                width="100%",
                                border_radius="var(--radius-3)",
                                border="1px solid",
                                border_color=rx.color("gray", 5)
                            ),
                            rx.text(
                                "MLP converges quickly with smooth loss reduction. Validation loss stabilizes around epoch 30, indicating good generalization.",
                                size="2",
                                color="var(--gray-11)",
                                text_align="center",
                                margin_top="0.5em"
                            ),
                            spacing="2",
                            margin_bottom="1em"
                        ),
                        
                        # Prediction vs Actual (2-column grid)
                        rx.grid(
                            rx.vstack(
                                rx.heading("Predictions vs Actual", size="3", weight="bold", margin_bottom="0.5em"),
                                rx.image(
                                    src="/modeling_plots/mlp/pred_actual.png",
                                    width="100%",
                                    border_radius="var(--radius-3)",
                                    border="1px solid",
                                    border_color=rx.color("gray", 5)
                                ),
                                rx.text(
                                    "Strong correlation between predicted and actual values (R² = 0.960).",
                                    size="2",
                                    color="var(--gray-11)",
                                    text_align="center",
                                    margin_top="0.5em"
                                ),
                                spacing="2",
                                align="start"
                            ),
                            rx.vstack(
                                rx.heading("Linear Fit Analysis", size="3", weight="bold", margin_bottom="0.5em"),
                                rx.image(
                                    src="/modeling_plots/mlp/pred_actual_linefit.png",
                                    width="100%",
                                    border_radius="var(--radius-3)",
                                    border="1px solid",
                                    border_color=rx.color("gray", 5)
                                ),
                                rx.text(
                                    "Regression line demonstrates MLP's ability to capture the full range of gold prices accurately.",
                                    size="2",
                                    color="var(--gray-11)",
                                    text_align="center",
                                    margin_top="0.5em"
                                ),
                                spacing="2",
                                align="start"
                            ),
                            columns="2",
                            spacing="3",
                            width="100%"
                        ),
                        
                        spacing="3",
                        align="start"
                    ),
                    padding="1.5em",
                    background=rx.color("blue", 1),
                    border="1px solid",
                    border_color=rx.color("blue", 5),
                    border_radius="var(--radius-4)"
                ),
                value="mlp"
            ),
            
            # RNN Tab
            rx.tabs.content(
                rx.box(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("git-branch", size=24, color=rx.color("purple", 9)),
                            rx.heading("RNN (Recurrent Neural Network) - R² = 0.600", size="4", weight="bold", color=rx.color("purple", 10)),
                            spacing="2",
                            align="center",
                            margin_bottom="0.5em"
                        ),
                        rx.text(
                            "Simple RNN struggles with vanishing gradients. Only sees past gold prices (window=12), lacks economic context.",
                            size="2",
                            color="var(--gray-11)",
                            margin_bottom="1em"
                        ),
                        
                        rx.vstack(
                            rx.heading("Training Loss", size="3", weight="bold", margin_bottom="0.5em"),
                            rx.image(
                                src="/modeling_plots/rnn/training_loss.png",
                                width="100%",
                                border_radius="var(--radius-3)",
                                border="1px solid",
                                border_color=rx.color("gray", 5)
                            ),
                            rx.text(
                                "RNN struggles with vanishing gradients - loss plateaus early. Limited capacity to capture long-term dependencies.",
                                size="2",
                                color="var(--gray-11)",
                                text_align="center",
                                margin_top="0.5em"
                            ),
                            spacing="2",
                            margin_bottom="1em"
                        ),
                        
                        rx.grid(
                            rx.vstack(
                                rx.heading("Predictions vs Actual", size="3", weight="bold", margin_bottom="0.5em"),
                                rx.image(
                                    src="/modeling_plots/rnn/pred_actual.png",
                                    width="100%",
                                    border_radius="var(--radius-3)",
                                    border="1px solid",
                                    border_color=rx.color("gray", 5)
                                ),
                                rx.text(
                                    "Wide scatter indicates poor fit (R² = 0.600). RNN fails to capture gold's complex dynamics from price history alone.",
                                    size="2",
                                    color="var(--gray-11)",
                                    text_align="center",
                                    margin_top="0.5em"
                                ),
                                spacing="2",
                                align="start"
                            ),
                            rx.vstack(
                                rx.heading("Linear Fit Analysis", size="3", weight="bold", margin_bottom="0.5em"),
                                rx.image(
                                    src="/modeling_plots/rnn/pred_actual_linefit.png",
                                    width="100%",
                                    border_radius="var(--radius-3)",
                                    border="1px solid",
                                    border_color=rx.color("gray", 5)
                                ),
                                rx.text(
                                    "Regression line deviates significantly from ideal y=x, confirming systematic prediction errors.",
                                    size="2",
                                    color="var(--gray-11)",
                                    text_align="center",
                                    margin_top="0.5em"
                                ),
                                spacing="2",
                                align="start"
                            ),
                            columns="2",
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
                    border_radius="var(--radius-4)"
                ),
                value="rnn"
            ),
            
            # LSTM Tab
            rx.tabs.content(
                rx.box(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("boxes", size=24, color=rx.color("green", 9)),
                            rx.heading("LSTM (Long Short-Term Memory) - R² = 0.603", size="4", weight="bold", color=rx.color("green", 10)),
                            spacing="2",
                            align="center",
                            margin_bottom="0.5em"
                        ),
                        rx.text(
                            "LSTM's 3-gate architecture (input, forget, output) shows marginal improvement over RNN but still struggles without economic features.",
                            size="2",
                            color="var(--gray-11)",
                            margin_bottom="1em"
                        ),
                        
                        rx.vstack(
                            rx.heading("Training Loss", size="3", weight="bold", margin_bottom="0.5em"),
                            rx.image(
                                src="/modeling_plots/lstm/training_loss.png",
                                width="100%",
                                border_radius="var(--radius-3)",
                                border="1px solid",
                                border_color=rx.color("gray", 5)
                            ),
                            rx.text(
                                "LSTM's gating mechanisms show slightly better convergence than RNN, but still struggles without economic features (R² = 0.603).",
                                size="2",
                                color="var(--gray-11)",
                                text_align="center",
                                margin_top="0.5em"
                            ),
                            spacing="2",
                            margin_bottom="1em"
                        ),
                        
                        rx.grid(
                            rx.vstack(
                                rx.heading("Predictions vs Actual", size="3", weight="bold", margin_bottom="0.5em"),
                                rx.image(
                                    src="/modeling_plots/lstm/pred_actual.png",
                                    width="100%",
                                    border_radius="var(--radius-3)",
                                    border="1px solid",
                                    border_color=rx.color("gray", 5)
                                ),
                                rx.text(
                                    "LSTM performs marginally better than RNN but still shows significant prediction errors without Multiple features.",
                                    size="2",
                                    color="var(--gray-11)",
                                    text_align="center",
                                    margin_top="0.5em"
                                ),
                                spacing="2",
                                align="start"
                            ),
                            rx.vstack(
                                rx.heading("Linear Fit Analysis", size="3", weight="bold", margin_bottom="0.5em"),
                                rx.image(
                                    src="/modeling_plots/lstm/pred_actual_linefit.png",
                                    width="100%",
                                    border_radius="var(--radius-3)",
                                    border="1px solid",
                                    border_color=rx.color("gray", 5)
                                ),
                                rx.text(
                                    "Gates help with memory but cannot compensate for missing economic context. Fit remains poor.",
                                    size="2",
                                    color="var(--gray-11)",
                                    text_align="center",
                                    margin_top="0.5em"
                                ),
                                spacing="2",
                                align="start"
                            ),
                            columns="2",
                            spacing="3",
                            width="100%"
                        ),
                        
                        spacing="3",
                        align="start"
                    ),
                    padding="1.5em",
                    background=rx.color("green", 1),
                    border="1px solid",
                    border_color=rx.color("green", 5),
                    border_radius="var(--radius-4)"
                ),
                value="lstm"
            ),
            
            # GRU Tab
            rx.tabs.content(
                rx.box(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("circuit-board", size=24, color=rx.color("amber", 9)),
                            rx.heading("GRU (Gated Recurrent Unit) - R² = 0.843", size="4", weight="bold", color=rx.color("amber", 10)),
                            spacing="2",
                            align="center",
                            margin_bottom="0.5em"
                        ),
                        rx.text(
                            "Best simple recurrent model! GRU's simplified 2-gate design (reset + update) proves more effective than LSTM for limited data.",
                            size="2",
                            color="var(--gray-11)",
                            margin_bottom="1em"
                        ),
                        
                        rx.vstack(
                            rx.heading("Training Loss", size="3", weight="bold", margin_bottom="0.5em"),
                            rx.image(
                                src="/modeling_plots/gru/training_loss.png",
                                width="100%",
                                border_radius="var(--radius-3)",
                                border="1px solid",
                                border_color=rx.color("gray", 5)
                            ),
                            rx.text(
                                "GRU achieves best simple performance (R² = 0.843) with efficient 2-gate architecture. Simpler than LSTM but more effective for limited data.",
                                size="2",
                                color="var(--gray-11)",
                                text_align="center",
                                margin_top="0.5em"
                            ),
                            spacing="2",
                            margin_bottom="1em"
                        ),
                        
                        rx.grid(
                            rx.vstack(
                                rx.heading("Predictions vs Actual", size="3", weight="bold", margin_bottom="0.5em"),
                                rx.image(
                                    src="/modeling_plots/gru/pred_actual.png",
                                    width="100%",
                                    border_radius="var(--radius-3)",
                                    border="1px solid",
                                    border_color=rx.color("gray", 5)
                                ),
                                rx.text(
                                    "Tighter cluster than RNN/LSTM, showing GRU's superior ability to learn temporal patterns from simple gold prices.",
                                    size="2",
                                    color="var(--gray-11)",
                                    text_align="center",
                                    margin_top="0.5em"
                                ),
                                spacing="2",
                                align="start"
                            ),
                            rx.vstack(
                                rx.heading("Linear Fit Analysis", size="3", weight="bold", margin_bottom="0.5em"),
                                rx.image(
                                    src="/modeling_plots/gru/pred_actual_linefit.png",
                                    width="100%",
                                    border_radius="var(--radius-3)",
                                    border="1px solid",
                                    border_color=rx.color("gray", 5)
                                ),
                                rx.text(
                                    "Best simple recurrent model, but still limited without economic features. Sets stage for Multiple improvements.",
                                    size="2",
                                    color="var(--gray-11)",
                                    text_align="center",
                                    margin_top="0.5em"
                                ),
                                spacing="2",
                                align="start"
                            ),
                            columns="2",
                            spacing="3",
                            width="100%"
                        ),
                        
                        spacing="3",
                        align="start"
                    ),
                    padding="1.5em",
                    background=rx.color("amber", 1),
                    border="1px solid",
                    border_color=rx.color("amber", 5),
                    border_radius="var(--radius-4)"
                ),
                value="gru"
            ),
            
            default_value="mlp",
            width="100%"
        ),
        
        spacing="3",
        align="start",
        width="100%",
        margin_bottom="2em"
    )


def deep_learning_Multiple() -> rx.Component:
    """Deep learning Multiple models - achieving optimal performance."""
    dl_multi_data = [
        ["GRU (Many - One)", "0.990", "$45.92", "$34.94", "Optimal balance of performance"],
        ["LSTM (Many - One)", "0.990", "$45.31", "$37.84", "Significant Improvement in MAE compare to LSTM(One - One)"],
        ["RNN (Many - One)", "0.972", "$76.77", "$58.99", "Good but simpler architecture limits"]
    ]
    
    return rx.vstack(
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.icon("zap", size=32, color=rx.color("purple", 9)),
                    rx.heading("Deep Learning (Many - One)", size="6", weight="bold"),
                    spacing="2",
                    align="center"
                ),
                rx.text(
                    "By combining temporal patterns (",
                    rx.text.strong("12-month windows"),
                    ") with macroeconomic context (",
                    rx.text.strong("13 features"),
                    "), "
                    "(Many - One) recurrent models achieve highly accurate predictive fit. "
                    "GRU and LSTM both reach ",
                    rx.text.strong("R² = 0.990"),
                    ", reducing average error to just ",
                    rx.text.strong("$35-38"),
                    ".",
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
            "Final Showdown: Multiple Recurrent Models",
            "These models see both time patterns AND economic drivers simultaneously. "
            "Each timestep contains all features, allowing the model to learn "
            "how gold responds to changing economic conditions over time.",
            dl_multi_data,
        ),
        
        # Highlight key result
        rx.box(
            rx.text(
                "Each timestep contains ",
                rx.text.strong("13 features"),
                " (CPI, interest rates, VIX, S&P 500, etc.), enabling both GRU and LSTM to achieve ",
                rx.text.strong("R² = 0.990"),
                ".",
                size="3",
                color="var(--gray-12)",
                line_height="1.7"
            ),
            padding="1em",
            background=rx.color("purple", 2),
            border_left=f"4px solid {rx.color('purple', 9)}",
            border_radius="var(--radius-3)",
            margin_bottom="1em"
        ),
        
        rx.grid(
            metric_card("R² Improvement", "+0.147", "green", "vs RNN simple"),
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
                rx.heading("GRU Multiple", size="5", weight="bold", margin_bottom="1em"),
                
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
        
        # Multiple Model Visualizations - Tabbed Interface
        rx.heading("Training & Performance Visualizations", size="5", weight="bold", margin_bottom="1em", margin_top="1em"),
        
        rx.tabs.root(
            rx.tabs.list(
                rx.tabs.trigger("RNN", value="rnn_multi"),
                rx.tabs.trigger("LSTM", value="lstm_multi"),
                rx.tabs.trigger("GRU 🏆", value="gru_multi"),
            ),
            
            # RNN Multiple Tab
            rx.tabs.content(
                rx.box(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("git-branch", size=24, color=rx.color("purple", 9)),
                            rx.heading("RNN Multiple - R² = 0.972", size="4", weight="bold", color=rx.color("purple", 10)),
                            spacing="2",
                            align="center",
                            margin_bottom="0.5em"
                        ),
                        rx.text(
                            "RNN dramatically improves from R² = 0.600 (simple) to 0.972 (Multiple) with economic features. MAE drops from $184 to $59.",
                            size="2",
                            color="var(--gray-11)",
                            margin_bottom="1em"
                        ),
                        
                        rx.vstack(
                            rx.heading("Training Loss", size="3", weight="bold", margin_bottom="0.5em"),
                            rx.image(
                                src="/modeling_plots/rnn_multivariate/training_loss.png",
                                width="100%",
                                border_radius="var(--radius-3)",
                                border="1px solid",
                                border_color=rx.color("gray", 5)
                            ),
                            rx.text(
                                "Economic features stabilize training and reduce vanishing gradient issues. Convergence is much smoother than simple RNN.",
                                size="2",
                                color="var(--gray-11)",
                                text_align="center",
                                margin_top="0.5em"
                            ),
                            spacing="2",
                            margin_bottom="1em"
                        ),
                        
                        rx.grid(
                            rx.vstack(
                                rx.heading("Predictions vs Actual", size="3", weight="bold", margin_bottom="0.5em"),
                                rx.image(
                                    src="/modeling_plots/rnn_multivariate/pred_actual.png",
                                    width="100%",
                                    border_radius="var(--radius-3)",
                                    border="1px solid",
                                    border_color=rx.color("gray", 5)
                                ),
                                rx.text(
                                    "Much tighter cluster than simple RNN. Multiple features enable RNN to capture complex gold dynamics.",
                                    size="2",
                                    color="var(--gray-11)",
                                    text_align="center",
                                    margin_top="0.5em"
                                ),
                                spacing="2",
                                align="start"
                            ),
                            rx.vstack(
                                rx.heading("Linear Fit Analysis", size="3", weight="bold", margin_bottom="0.5em"),
                                rx.image(
                                    src="/modeling_plots/rnn_multivariate/pred_actual_linefit.png",
                                    width="100%",
                                    border_radius="var(--radius-3)",
                                    border="1px solid",
                                    border_color=rx.color("gray", 5)
                                ),
                                rx.text(
                                    "Strong linear relationship demonstrates RNN's improved prediction accuracy with economic context (MAE = $58.99).",
                                    size="2",
                                    color="var(--gray-11)",
                                    text_align="center",
                                    margin_top="0.5em"
                                ),
                                spacing="2",
                                align="start"
                            ),
                            columns="2",
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
                    border_radius="var(--radius-4)"
                ),
                value="rnn_multi"
            ),
            
            # LSTM Multiple Tab
            rx.tabs.content(
                rx.box(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("boxes", size=24, color=rx.color("green", 9)),
                            rx.heading("LSTM Multiple - R² = 0.990", size="4", weight="bold", color=rx.color("green", 10)),
                            spacing="2",
                            align="center",
                            margin_bottom="0.5em"
                        ),
                        rx.text(
                            "LSTM achieves near-optimal performance (R² = 0.990, MAE = $37.84). Three-gate architecture excels at managing long-term dependencies across 13 features.",
                            size="2",
                            color="var(--gray-11)",
                            margin_bottom="1em"
                        ),
                        
                        rx.vstack(
                            rx.heading("Training Loss", size="3", weight="bold", margin_bottom="0.5em"),
                            rx.image(
                                src="/modeling_plots/lstm_multivariate/training_loss.png",
                                width="100%",
                                border_radius="var(--radius-3)",
                                border="1px solid",
                                border_color=rx.color("gray", 5)
                            ),
                            rx.text(
                                "Smooth convergence with excellent validation performance. LSTM's memory cells effectively integrate economic signals over time.",
                                size="2",
                                color="var(--gray-11)",
                                text_align="center",
                                margin_top="0.5em"
                            ),
                            spacing="2",
                            margin_bottom="1em"
                        ),
                        
                        rx.grid(
                            rx.vstack(
                                rx.heading("Predictions vs Actual", size="3", weight="bold", margin_bottom="0.5em"),
                                rx.image(
                                    src="/modeling_plots/lstm_multivariate/pred_actual.png",
                                    width="100%",
                                    border_radius="var(--radius-3)",
                                    border="1px solid",
                                    border_color=rx.color("gray", 5)
                                ),
                                rx.text(
                                    "Extremely tight scatter along diagonal - LSTM captures gold price variations with exceptional accuracy (MAE = $37.84).",
                                    size="2",
                                    color="var(--gray-11)",
                                    text_align="center",
                                    margin_top="0.5em"
                                ),
                                spacing="2",
                                align="start"
                            ),
                            rx.vstack(
                                rx.heading("Linear Fit Analysis", size="3", weight="bold", margin_bottom="0.5em"),
                                rx.image(
                                    src="/modeling_plots/lstm_multivariate/pred_actual_linefit.png",
                                    width="100%",
                                    border_radius="var(--radius-3)",
                                    border="1px solid",
                                    border_color=rx.color("gray", 5)
                                ),
                                rx.text(
                                    "Near-perfect alignment with y=x ideal line. LSTM's gates (input, forget, output) manage complex feature interactions.",
                                    size="2",
                                    color="var(--gray-11)",
                                    text_align="center",
                                    margin_top="0.5em"
                                ),
                                spacing="2",
                                align="start"
                            ),
                            columns="2",
                            spacing="3",
                            width="100%"
                        ),
                        
                        spacing="3",
                        align="start"
                    ),
                    padding="1.5em",
                    background=rx.color("green", 1),
                    border="1px solid",
                    border_color=rx.color("green", 5),
                    border_radius="var(--radius-4)"
                ),
                value="lstm_multi"
            ),
            
            # GRU Multiple Tab (CHAMPION!)
            rx.tabs.content(
                rx.box(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("trophy", size=28, color=rx.color("amber", 9)),
                            rx.heading("GRU Multiple - CHAMPION (R² = 0.990)", size="4", weight="bold", color=rx.color("amber", 10)),
                            spacing="2",
                            align="center",
                            margin_bottom="0.5em"
                        ),
                        rx.text(
                            rx.text.strong("Best overall model! "),
                            "R² = 0.990, MAE = $34.94 (lowest error of all models). GRU's 2-gate architecture achieves optimal balance: complex enough to excel, efficient enough to train ~20% faster than LSTM.",
                            size="2",
                            color="var(--gray-11)",
                            margin_bottom="1em"
                        ),
                        
                        rx.vstack(
                            rx.heading("Training Loss", size="3", weight="bold", margin_bottom="0.5em"),
                            rx.image(
                                src="/modeling_plots/gru_multivariate/training_loss.png",
                                width="100%",
                                border_radius="var(--radius-3)",
                                border="2px solid",
                                border_color=rx.color("amber", 6)
                            ),
                            rx.text(
                                rx.text.strong("Optimal training efficiency. "),
                                "GRU's simplified design (reset + update gates) converges smoothly while capturing all essential temporal patterns.",
                                size="2",
                                color="var(--gray-11)",
                                text_align="center",
                                margin_top="0.5em"
                            ),
                            spacing="2",
                            margin_bottom="1em"
                        ),
                        
                        rx.grid(
                            rx.vstack(
                                rx.heading("Predictions vs Actual", size="3", weight="bold", margin_bottom="0.5em"),
                                rx.image(
                                    src="/modeling_plots/gru_multivariate/pred_actual.png",
                                    width="100%",
                                    border_radius="var(--radius-3)",
                                    border="2px solid",
                                    border_color=rx.color("amber", 6)
                                ),
                                rx.text(
                                    rx.text.strong("Tightest cluster of all models! "),
                                    "Lowest MAE ($34.94) demonstrates GRU's superior generalization across all price ranges.",
                                    size="2",
                                    color="var(--gray-11)",
                                    text_align="center",
                                    margin_top="0.5em"
                                ),
                                spacing="2",
                                align="start"
                            ),
                            rx.vstack(
                                rx.heading("Linear Fit Analysis", size="3", weight="bold", margin_bottom="0.5em"),
                                rx.image(
                                    src="/modeling_plots/gru_multivariate/pred_actual_linefit.png",
                                    width="100%",
                                    border_radius="var(--radius-3)",
                                    border="2px solid",
                                    border_color=rx.color("amber", 6)
                                ),
                                rx.text(
                                    rx.text.strong("Perfect regression line! "),
                                    "GRU learned how CPI, interest rates, VIX, and 10 other features drive gold prices across the full historical range.",
                                    size="2",
                                    color="var(--gray-11)",
                                    text_align="center",
                                    margin_top="0.5em"
                                ),
                                spacing="2",
                                align="start"
                            ),
                            columns="2",
                            spacing="3",
                            width="100%"
                        ),
                        
                        spacing="3",
                        align="start"
                    ),
                    padding="1.5em",
                    background=rx.color("amber", 1),
                    border="2px solid",
                    border_color=rx.color("amber", 6),
                    border_radius="var(--radius-4)"
                ),
                value="gru_multi"
            ),
            
            default_value="gru_multi",
            width="100%"
        ),
        
        rx.box(

            rx.vstack(
                rx.hstack(
                    rx.icon("zap", size=24, color=rx.color("amber", 9)),
                    rx.heading("Why GRU Wins Over LSTM", size="4", weight="bold"),
                    spacing="2",
                    align="center"
                ),
                rx.text(
                    "Both achieve ",
                    rx.text.strong("R² = 0.990"),
                    ", but GRU has lower MAE (",
                    rx.text.strong("$34.94 vs $37.84"),
                    ") and trains ",
                    rx.text.strong("~20% faster"),
                    ". "
                    "GRU's simpler architecture (",
                    rx.text.strong("2 gates instead of 3"),
                    ") is sufficient for our dataset size. "
                    "LSTM's forget gate advantage doesn't materialize because our ",
                    rx.text.strong("12-month window"),
                    " already captures relevant history. "
                    "GRU is the Goldilocks solution: complex enough to excel, simple enough to be efficient.",
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
            margin_bottom="1em"
        ),
        
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.icon("trending-up", size=24, color=rx.color("green", 9)),
                    rx.heading("The Multiple Advantage", size="4", weight="bold"),
                    spacing="2",
                    align="center"
                ),
                rx.text(
                    "Comparing simple vs Multiple GRU: R² jumped from ",
                    rx.text.strong("0.843 → 0.990"),
                    " (",
                    rx.text.strong("+0.147"),
                    "), and MAE dropped from ",
                    rx.text.strong("$122.95 → $34.94"),
                    " (",
                    rx.text.strong("-72%"),
                    "). "
                    "Why? The model now understands ",
                    rx.text.strong("WHY"),
                    " gold prices change. When CPI rises, interest rates fall, and VIX spikes, "
                    "the model learned to predict gold surges - something impossible from price history alone.",
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


def grand_comparison() -> rx.Component:
    """Final comparison of all models."""
    all_models_data = [
        ["GRU Multiple", "0.990", "$45.92", "$34.94", "Top performer - best overall"],
        ["LSTM Multiple", "0.990", "$45.31", "$37.84", "Nearly tied with GRU"],
        ["SVR (RBF Kernel)", "0.986", "$59.93", "$43.77", "Best traditional ML"],
        ["Random Forest", "0.986", "$59.93", "$43.77", "Tied with SVR"],
        ["XGBoost", "0.973", "$82.67", "$51.11", "Gradient boosting"],
        ["RNN Multiple", "0.972", "$76.77", "$58.99", "Good but simpler"],
        ["MLP", "0.960", "$100.62", "$78.85", "Feedforward baseline"],
        ["Linear Regression", "0.947", "$115.88", "$77.06", "Strong baseline"],
        ["Ridge Regression", "0.947", "$115.88", "$77.06", "No improvement"],
        ["GRU simple", "0.843", "$164.93", "$122.95", "Needs features"],
        ["LSTM simple", "0.603", "$262.55", "$193.85", "Insufficient"],
        ["RNN simple", "0.600", "$263.33", "$184.26", "Insufficient"],
        ["SARIMA", "0.270", "$353.57", "$233.26", "Time series weak"],
        ["ARIMA", "-0.480", "$503.12", "$321.93", "Failed completely"]
    ]
    
    return rx.vstack(
        rx.heading("Grand Comparison: All 14 Models Ranked", size="7", weight="bold", margin_bottom="1em"),
        
        rx.text(
            "Here's the complete leaderboard of ",
            rx.text.strong("all 14 models"),
            " tested, sorted by R². "
            "The progression from baseline to deep learning shows a clear trend: "
            "complexity pays off when combined with ",
            rx.text.strong("rich Multiple features"),
            " and ",
            rx.text.strong("temporal modeling"),
            ".",
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
                header="1. Multiple Deep Learning Achieves Optimal Performance",
                content=rx.text(
                    "The combination of temporal modeling (RNN/LSTM/GRU) with rich macroeconomic features (CPI, interest rates, market indices) "
                    "produces the best results. GRU Multiple achieved R²=0.990 with MAE=$34.94, outperforming all other approaches. "
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
                    "simple models (even sophisticated LSTM) failed (R²=0.60), while simple Linear Regression with good features achieved R²=0.947. "
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
                "Based on comprehensive evaluation, ",
                rx.text.strong("GRU Multiple"),
                " emerges as the optimal model with ",
                rx.text.strong("R² = 0.990"),
                " and ",
                rx.text.strong("MAE = $34.94"),
                ". "
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
                "that our ",
                rx.text.strong("R² = 0.990"),
                " achievement represents genuine ",
                rx.text.strong("state-of-the-art performance"),
                ".",
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
                
                simple_regression_detail(),
                section_divider(),
                
                polynomial_regression_section(),
                section_divider(),
                
                time_series_section(),
                section_divider(),
                
                Multiple_regression_detail(),
                section_divider(),
                
                ridge_regression_detail(),
                section_divider(),
                
                baseline_models(),
                section_divider(),
                
                traditional_ml(),
                section_divider(),
                
                deep_learning_simple(),
                section_divider(),
                
                deep_learning_Multiple(),
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
