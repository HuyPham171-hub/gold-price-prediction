"""Home page - Introduction and project overview."""
import reflex as rx


def home_page() -> rx.Component:
    """Home page component for GoldSight application."""
    return rx.container(
        rx.vstack(
            # Hero Section
            rx.heading(
                "GoldSight - Gold Price Prediction", 
                size="9",
                margin_bottom="1rem"
            ),
            rx.text(
                "Advanced Machine Learning & Deep Learning for Gold Price Forecasting",
                size="5",
                color_scheme="gray",
                margin_bottom="2rem"
            ),
            
            # Introduction
            rx.box(
                rx.heading("Why Predict Gold Prices?", size="7", margin_bottom="1rem"),
                rx.unordered_list(
                    rx.list_item("Safe-haven asset during economic uncertainty"),
                    rx.list_item("Hedge against inflation and currency devaluation"),
                    rx.list_item("Portfolio diversification and risk management"),
                    rx.list_item("Strategic investment decision making"),
                ),
                padding="2rem",
                border_radius="lg",
                bg="gray.50",
                margin_bottom="2rem"
            ),
            
            # Project Objectives
            rx.box(
                rx.heading("Project Objectives", size="7", margin_bottom="1rem"),
                rx.unordered_list(
                    rx.list_item("Compare performance of multiple ML/DL models"),
                    rx.list_item("Build real-time forecasting pipeline"),
                    rx.list_item("Create interactive data visualizations"),
                    rx.list_item("Analyze key factors influencing gold prices"),
                    rx.list_item("Implement multivariate time-series prediction"),
                ),
                padding="2rem",
                border_radius="lg",
                bg="blue.50",
                margin_bottom="2rem"
            ),
            
            # Navigation Cards
            rx.heading("Explore GoldSight", size="7", margin_bottom="1rem"),
            rx.grid(
                rx.link(
                    rx.box(
                        rx.heading("📊 Data Collection", size="5", margin_bottom="0.5rem"),
                        rx.text("Collect data from Yahoo Finance, FRED API, and GPR Index"),
                        padding="1.5rem",
                        border_radius="lg",
                        bg="green.50",
                        _hover={"bg": "green.100"},
                    ),
                    href="/data-collection"
                ),
                rx.link(
                    rx.box(
                        rx.heading("🔍 Exploratory Analysis", size="5", margin_bottom="0.5rem"),
                        rx.text("In-depth data analysis and visualization"),
                        padding="1.5rem",
                        border_radius="lg",
                        bg="purple.50",
                        _hover={"bg": "purple.100"},
                    ),
                    href="/eda"
                ),
                rx.link(
                    rx.box(
                        rx.heading("🤖 Model Training", size="5", margin_bottom="0.5rem"),
                        rx.text("Train Linear, ARIMA, ML, and Deep Learning models"),
                        padding="1.5rem",
                        border_radius="lg",
                        bg="orange.50",
                        _hover={"bg": "orange.100"},
                    ),
                    href="/modeling"
                ),
                rx.link(
                    rx.box(
                        rx.heading("📈 Price Forecast", size="5", margin_bottom="0.5rem"),
                        rx.text("Real-time gold price predictions and insights"),
                        padding="1.5rem",
                        border_radius="lg",
                        bg="red.50",
                        _hover={"bg": "red.100"},
                    ),
                    href="/forecast"
                ),
                columns="2",
                spacing="4",
                width="100%"
            ),
            
            spacing="4",
            width="100%",
            max_width="1200px",
            padding="2rem"
        )
    )

