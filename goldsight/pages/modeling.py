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
            rx.text(label, size="2", weight="medium"),
            rx.heading(value, size="7", weight="bold", color=rx.color(color_scheme, 10)),
            rx.cond(
                description != "",
                rx.text(description, size="1", color=rx.color("gray", 11)),
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
        rx.text(description, size="3", margin_bottom="1em", line_height="1.7"),
        
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
            rx.icon(icon, size=24, color=rx.color(color_scheme, 9)),
            rx.vstack(
                rx.heading(title, size="4", weight="bold"),
                rx.text(content, size="3", line_height="1.7"),
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
        rx.heading("Executive Summary", size="6", weight="bold", color_scheme="purple"),
        
        rx.box(
            rx.vstack(
                rx.heading("Best model: GRU Multivariate", size="5", weight="bold", margin_bottom="0.5em"),
                rx.text(
                    "After testing 11 different models across 3 algorithm families, we identified GRU (Gated Recurrent Unit) "
                    "with multivariate inputs as the best performer, achieving R²=0.990 and MAE=$34.94 — "
                    "nearly perfect predictions with less than $35 average error.",
                    size="3",
                    line_height="1.7",
                    margin_bottom="1em"
                ),
                
                rx.grid(
                    metric_card("Best R²", "0.990", "green", "Explains 99% of variance"),
                    metric_card("Best RMSE", "$45.92", "blue", "±$46 typical error"),
                    metric_card("Best MAE", "$34.94", "purple", "Smallest average error"),
                    metric_card("Models Tested", "11", "amber", "3 families compared"),
                    columns="4",
                    spacing="3",
                    width="100%",
                    margin_y="1em"
                ),
                
                rx.heading("The Journey", size="5", weight="bold", margin_top="1em", margin_bottom="0.5em"),
                rx.unordered_list(
                    rx.list_item(
                        rx.text.strong("Baseline (Linear Regression): "),
                        "R²=0.947 — Strong start with multivariate features"
                    ),
                    rx.list_item(
                        rx.text.strong("Traditional ML (SVR, Random Forest): "),
                        "R²=0.986 — Significant improvement with non-linear methods"
                    ),
                    rx.list_item(
                        rx.text.strong("Deep Learning (GRU Multivariate): "),
                        "R²=0.990 — Best performance by capturing temporal patterns + feature interactions"
                    ),
                    rx.list_item(
                        rx.text.strong("Key Insight: "),
                        "Multivariate deep learning >> Univariate >> Time series alone (ARIMA failed)"
                    ),
                    spacing="2",
                    padding_left="1.5em"
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
            "Rather than jumping straight to complex deep learning, we followed a systematic approach: "
            "start with simple baselines, understand their limitations, then progressively add complexity. "
            "This ensures we understand what each model contributes and avoid unnecessary sophistication.",
            size="4",
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
                        "Linear Regression, Ridge, ARIMA/SARIMA",
                        size="3"
                    ),
                    rx.text(
                        "Establish minimum acceptable performance. If simple works, why complicate?",
                        size="2",
                        color=rx.color("gray", 11),
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
                        "SVR, Random Forest, XGBoost",
                        size="3"
                    ),
                    rx.text(
                        "Handle non-linearity and feature interactions. Test if tree-based methods outperform linear.",
                        size="2",
                        color=rx.color("gray", 11),
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
                        "MLP, RNN, LSTM, GRU (Univariate & Multivariate)",
                        size="3"
                    ),
                    rx.text(
                        "Capture temporal dependencies and complex patterns. Maximum predictive power.",
                        size="2",
                        color=rx.color("gray", 11),
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
                rx.heading("📊 Evaluation Criteria", size="5", weight="bold", margin_bottom="0.5em"),
                rx.grid(
                    rx.vstack(
                        rx.text.strong("R² (Coefficient of Determination)"),
                        rx.text("How much variance is explained? Higher is better. Target: > 0.95", size="2"),
                        align="start"
                    ),
                    rx.vstack(
                        rx.text.strong("RMSE (Root Mean Squared Error)"),
                        rx.text("Average prediction error in dollars. Lower is better. Target: < $50", size="2"),
                        align="start"
                    ),
                    rx.vstack(
                        rx.text.strong("MAE (Mean Absolute Error)"),
                        rx.text("Average absolute deviation. Most interpretable metric. Target: < $40", size="2"),
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


def baseline_models() -> rx.Component:
    """Baseline models comparison."""
    baseline_data = [
        ["Linear Regression", "0.947", "$115.88", "$77.06", "Strong multivariate baseline"],
        ["Ridge Regression", "0.947", "$115.88", "$77.06", "No improvement (low multicollinearity)"],
        ["ARIMA (Manual)", "−0.480", "$503.12", "$321.93", "Failed — univariate insufficient"],
        ["SARIMA (Manual)", "0.270", "$353.57", "$233.26", "Poor — seasonal patterns weak"]
    ]
    
    return rx.vstack(
        comparison_table_section(
            "Baseline Models: Linear & Time Series",
            "We start with the simplest models to establish a performance floor. "
            "Linear regression with 13 macroeconomic features provides a surprisingly strong baseline (R²=0.947). "
            "Traditional time series methods (ARIMA/SARIMA) fail because gold price is driven more by economic fundamentals than pure temporal patterns.",
            baseline_data,
            highlight_best=True
        ),
        
        insight_box(
            "trending-up",
            "Key Insight: Multivariate >> Univariate",
            "Linear regression with macroeconomic features (R²=0.947) vastly outperforms time series methods (SARIMA R²=0.270). "
            "This confirms our hypothesis: gold prices are driven by inflation, interest rates, and market conditions, not just historical patterns. "
            "ARIMA's negative R² means it performs worse than simply predicting the average!"
        ),
        
        insight_box(
            "info",
            "Why Ridge Didn't Help",
            "Ridge regression (L2 regularization) is designed to combat multicollinearity, but our VIF analysis showed "
            "multicollinearity was already low after feature selection. Ridge produced identical results to ordinary linear regression, "
            "confirming our feature engineering was effective.",
            color_scheme="blue"
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
        ["XGBoost", "0.973", "$82.67", "$51.11", "Underperformed — possible overfitting"]
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
                    rx.heading("🔧 SVR: Kernel Magic", size="4", weight="bold", margin_bottom="0.5em"),
                    rx.text(
                        "Support Vector Regression with RBF kernel maps features into high-dimensional space, "
                        "capturing complex non-linear patterns. GridSearchCV tested 27 combinations to find optimal hyperparameters.",
                        size="2",
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
                    rx.heading("🌲 Random Forest: Ensemble Power", size="4", weight="bold", margin_bottom="0.5em"),
                    rx.text(
                        "Random Forest trains 500 decision trees on random subsets of features, then averages predictions. "
                        "This ensemble approach reduces overfitting while capturing non-linear patterns.",
                        size="2",
                        line_height="1.6"
                    ),
                    rx.unordered_list(
                        rx.list_item("500 estimators (trees)"),
                        rx.list_item("Max depth = 20 layers"),
                        rx.list_item("1,620 CV fits (5-fold × 324 configs)"),
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
            "alert-circle",
            "XGBoost Surprise: Why Did It Underperform?",
            "XGBoost (R²=0.973) surprisingly performed worse than SVR and Random Forest. This suggests the model may be overfitting "
            "to training data despite regularization. XGBoost's strength lies in tabular data with complex interactions, but our "
            "relatively clean dataset with strong linear trends may not benefit as much from gradient boosting's aggressive optimization.",
            color_scheme="amber"
        ),
        
        insight_box(
            "target",
            "Feature Importance from Random Forest",
            "Top 3 most important features: 1) CPI (inflation) — 32% importance, 2) Silver_Futures — 18%, 3) S&P_500 — 15%. "
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
        ["MLP (Feedforward)", "0.960", "$100.62", "$78.85", "256→128→64→32 neurons, Dropout"],
        ["GRU (Univariate)", "0.843", "$164.93", "$122.95", "64→64 units, window=12"],
        ["LSTM (Univariate)", "0.603", "$262.55", "$193.85", "64→64 units, gates struggle"],
        ["RNN (Univariate)", "0.600", "$263.33", "$184.26", "Simple RNN insufficient"]
    ]
    
    return rx.vstack(
        comparison_table_section(
            "Deep Learning — Univariate (Gold Price Only)",
            "Before using all 13 features, we test if deep learning can extract temporal patterns from gold price history alone. "
            "MLP (feedforward) performs well (R²=0.960) as it uses all features but no sequence. "
            "Recurrent models (RNN/LSTM/GRU) use sliding windows of past prices but struggle without external features.",
            dl_uni_data,
            highlight_best=True
        ),
        
        rx.box(
            rx.vstack(
                rx.heading("🧠 Architecture Details", size="5", weight="bold", margin_bottom="1em"),
                
                rx.grid(
                    rx.vstack(
                        rx.heading("MLP", size="3", weight="bold", color=rx.color("blue", 10)),
                        rx.text("Multilayer Perceptron (Feedforward)", size="2", color=rx.color("gray", 11), margin_bottom="0.5em"),
                        rx.unordered_list(
                            rx.list_item("Input: 13 features (all at once)"),
                            rx.list_item("Layers: 256→128→64→32→1"),
                            rx.list_item("Dropout: 0.3, 0.2 (prevent overfitting)"),
                            rx.list_item("BatchNorm: After each hidden layer"),
                            spacing="1",
                            padding_left="1em"
                        ),
                        align="start"
                    ),
                    rx.vstack(
                        rx.heading("RNN/LSTM/GRU", size="3", weight="bold", color=rx.color("purple", 10)),
                        rx.text("Recurrent Neural Networks", size="2", color=rx.color("gray", 11), margin_bottom="0.5em"),
                        rx.unordered_list(
                            rx.list_item("Input: Window of 12 past prices"),
                            rx.list_item("Architecture: 64→64 recurrent units"),
                            rx.list_item("Dropout: 0.2 between layers"),
                            rx.list_item("Output: Dense(32)→Dense(1)"),
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
            "(e.g., 2008 crisis, COVID-19). This proves gold isn't just autoregressive — it needs external features!",
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
    """Deep learning multivariate models - the champions!"""
    dl_multi_data = [
        ["GRU (Multivariate)", "0.990", "$45.92", "$34.94", "Best balance"],
        ["LSTM (Multivariate)", "0.990", "$45.31", "$37.84", "Slightly lower MAE"],
        ["RNN (Multivariate)", "0.972", "$76.77", "$58.99", "Good but simpler architecture limits"]
    ]
    
    return rx.vstack(
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.icon("trophy", size=32, color=rx.color("amber", 9)),
                    rx.heading("Deep Learning — Multivariate: The Champions!", size="6", weight="bold"),
                    spacing="2",
                    align="center"
                ),
                rx.text(
                    "By combining temporal patterns (12-month windows) with macroeconomic context (13 features), "
                    "multivariate recurrent models achieve near-perfect predictions. This is the breakthrough moment: "
                    "GRU and LSTM both reach R²=0.990, reducing average error to just $35-38.",
                    size="4",
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
            metric_card("Error Reduction", "−79%", "blue", "MAE: $184 → $35"),
            metric_card("Training Time", "~5 min", "purple", "70 epochs with EarlyStopping"),
            metric_card("Parameters", "~50K", "amber", "128→64 GRU units"),
            columns="4",
            spacing="3",
            width="100%",
            margin_y="1.5em"
        ),
        
        rx.box(
            rx.vstack(
                rx.heading("🏗️ Winning Architecture: GRU Multivariate", size="5", weight="bold", margin_bottom="1em"),
                
                rx.grid(
                    rx.vstack(
                        rx.text.strong("Input Layer"),
                        rx.text("Shape: (batch, 12, 13)", size="2"),
                        rx.text("12 timesteps × 13 features", size="2", color=rx.color("gray", 11)),
                        align="start"
                    ),
                    rx.icon("arrow-right", size=24, color=rx.color("gray", 8)),
                    rx.vstack(
                        rx.text.strong("GRU Layer 1"),
                        rx.text("128 units, return sequences", size="2"),
                        rx.text("Dropout: 0.2", size="2", color=rx.color("gray", 11)),
                        align="start"
                    ),
                    rx.icon("arrow-right", size=24, color=rx.color("gray", 8)),
                    rx.vstack(
                        rx.text.strong("GRU Layer 2"),
                        rx.text("64 units, final state", size="2"),
                        rx.text("Captures long-term patterns", size="2", color=rx.color("gray", 11)),
                        align="start"
                    ),
                    rx.icon("arrow-right", size=24, color=rx.color("gray", 8)),
                    rx.vstack(
                        rx.text.strong("Dense Layers"),
                        rx.text("Dense(32, ReLU) → Dense(1)", size="2"),
                        rx.text("Final prediction", size="2", color=rx.color("gray", 11)),
                        align="start"
                    ),
                    columns="7",
                    spacing="2",
                    width="100%",
                    align="center"
                ),
                
                rx.divider(margin_y="1em"),
                
                rx.heading("⚙️ Training Configuration", size="4", weight="bold", margin_top="0.5em", margin_bottom="0.5em"),
                rx.grid(
                    rx.vstack(
                        rx.text.strong("Optimizer: Adam"),
                        rx.text("Adaptive learning rate", size="2", color=rx.color("gray", 11)),
                        align="start"
                    ),
                    rx.vstack(
                        rx.text.strong("Loss: MSE"),
                        rx.text("Mean Squared Error", size="2", color=rx.color("gray", 11)),
                        align="start"
                    ),
                    rx.vstack(
                        rx.text.strong("Callbacks: 3"),
                        rx.text("EarlyStopping, ReduceLR, Checkpoint", size="2", color=rx.color("gray", 11)),
                        align="start"
                    ),
                    rx.vstack(
                        rx.text.strong("Batch Size: 32"),
                        rx.text("Balanced speed/stability", size="2", color=rx.color("gray", 11)),
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
            "Comparing univariate vs multivariate GRU: R² jumped from 0.843 → 0.990 (+0.147), and MAE dropped from $122.95 → $34.94 (−72%). "
            "Why? The model now understands WHY gold prices change. When CPI rises, interest rates fall, and VIX spikes, "
            "the model learned to predict gold surges — something impossible from price history alone.",
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
        ["🏆 GRU Multivariate", "0.990", "$45.92", "$34.94", "Winner — best overall"],
        ["🥈 LSTM Multivariate", "0.990", "$45.31", "$37.84", "Nearly tied with GRU"],
        ["🥉 SVR (RBF Kernel)", "0.986", "$59.93", "$43.77", "Best traditional ML"],
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
        ["ARIMA", "−0.480", "$503.12", "$321.93", "Failed completely"]
    ]
    
    return rx.vstack(
        rx.heading("Grand Comparison: All 14 Models Ranked", size="7", weight="bold", margin_bottom="1em"),
        
        rx.text(
            "Here's the complete leaderboard of all models tested, sorted by R². "
            "The progression from baseline to deep learning shows a clear trend: "
            "complexity pays off when combined with rich multivariate features and temporal modeling.",
            size="4",
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
        
        rx.grid(
            rx.box(
                rx.vstack(
                    rx.heading("🎯 Top Tier (R² > 0.98)", size="4", weight="bold", margin_bottom="0.5em"),
                    rx.text("GRU Multi, LSTM Multi, SVR, Random Forest", size="2"),
                    rx.text("Near-perfect predictions. Production-ready.", size="2", color=rx.color("gray", 11), margin_top="0.25em"),
                    spacing="1",
                    align="start"
                ),
                padding="1.25em",
                background=rx.color("green", 1),
                border_left=f"4px solid {rx.color('green', 9)}",
                border_radius="var(--radius-3)"
            ),
            rx.box(
                rx.vstack(
                    rx.heading("📊 Mid Tier (R² 0.94-0.97)", size="4", weight="bold", margin_bottom="0.5em"),
                    rx.text("XGBoost, RNN Multi, MLP, Linear Regression", size="2"),
                    rx.text("Strong performance. Good baselines.", size="2", color=rx.color("gray", 11), margin_top="0.25em"),
                    spacing="1",
                    align="start"
                ),
                padding="1.25em",
                background=rx.color("blue", 1),
                border_left=f"4px solid {rx.color('blue', 9)}",
                border_radius="var(--radius-3)"
            ),
            rx.box(
                rx.vstack(
                    rx.heading("⚠️ Low Tier (R² < 0.90)", size="4", weight="bold", margin_bottom="0.5em"),
                    rx.text("Univariate models, ARIMA/SARIMA", size="2"),
                    rx.text("Insufficient for production use.", size="2", color=rx.color("gray", 11), margin_top="0.25em"),
                    spacing="1",
                    align="start"
                ),
                padding="1.25em",
                background=rx.color("red", 1),
                border_left=f"4px solid {rx.color('red', 9)}",
                border_radius="var(--radius-3)"
            ),
            columns="3",
            spacing="3",
            width="100%",
            margin_y="1.5em"
        ),
        
        spacing="3",
        align="start",
        width="100%",
        margin_bottom="2em"
    )


def key_takeaways() -> rx.Component:
    """Key learnings and insights."""
    return rx.vstack(
        rx.heading("Key Takeaways & Lessons Learned", size="7", weight="bold", margin_bottom="1em"),
        
        rx.accordion.root(
            rx.accordion.item(
                header="1. Multivariate Deep Learning Is The Winner",
                content=rx.text(
                    "The combination of temporal modeling (RNN/LSTM/GRU) with rich macroeconomic features (CPI, interest rates, market indices) "
                    "produces the best results. GRU Multivariate achieved R²=0.990 with MAE=$34.94, outperforming all other approaches. "
                    "This validates our hypothesis that gold prices are driven by economic fundamentals, not just momentum.",
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
                    "ARIMA achieved negative R² (−0.48), meaning it performed worse than predicting the mean. SARIMA barely improved (R²=0.27). "
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
                    "When in doubt, start with GRU — it's the sweet spot between SimpleRNN and LSTM for most financial time series.",
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
                "We've identified GRU Multivariate as our champion model with R²=0.990 and MAE=$34.94. "
                "In the next chapter, we'll deploy this model for real-time forecasting:",
                size="4",
                line_height="1.7",
                margin_y="1em"
            ),
            
            rx.grid(
                rx.vstack(
                    rx.heading("📈 Short-term Forecasts", size="4", weight="bold"),
                    rx.text("1-day, 7-day, 30-day predictions with confidence intervals", size="2"),
                    align="start"
                ),
                rx.vstack(
                    rx.heading("🎯 Scenario Analysis", size="4", weight="bold"),
                    rx.text("What if Fed raises rates 2%? What if CPI hits 5%?", size="2"),
                    align="start"
                ),
                rx.vstack(
                    rx.heading("🔍 Model Explainability", size="4", weight="bold"),
                    rx.text("SHAP values, feature attributions, error analysis", size="2"),
                    align="start"
                ),
                rx.vstack(
                    rx.heading("⚠️ Uncertainty Quantification", size="4", weight="bold"),
                    rx.text("Prediction intervals, Monte Carlo dropout, ensemble variance", size="2"),
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
