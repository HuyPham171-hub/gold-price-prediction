# goldsight/pages/home.py
import reflex as rx
from goldsight.components import page_layout
from goldsight.utils.design_system import (
    Colors,
    Typography,
    Spacing,
    Layout,
    card_style
)

# ======================================================================
# 1. HELPER COMPONENT: NAV_CARD
# ======================================================================
def nav_card(
    title: str,
    desc: str,
    route: str,
    icon_name: str,
    icon_color: str = "amber"  # Changed default to amber
) -> rx.Component:
    """
    A clickable navigation card using design system.
    """
    return rx.link(
        rx.flex(
            rx.vstack(
                # Icon
                rx.icon(
                    tag=icon_name,
                    size=32,
                    color_scheme=icon_color
                ),
                
                # Title
                rx.heading(
                    title,
                    size=Typography.SUBSECTION,  # Use design system
                    weight="bold",
                    color="var(--gray-12)"
                ),
                
                # Description
                rx.text(
                    desc,
                    color_scheme=Colors.NEUTRAL,
                    size=Typography.SMALL
                ),
                
                rx.spacer(),
                
                # "Learn more" link
                rx.hstack(
                    rx.text(
                        "Learn more",
                        color_scheme=Colors.PRIMARY,  # Amber accent
                        size=Typography.SMALL
                    ),
                    rx.icon(
                        tag="arrow-right",
                        size=16,
                        color_scheme=Colors.PRIMARY
                    ),
                    align="center",
                    spacing=Spacing.TIGHT
                ),
                
                align="start",
                spacing=Spacing.TIGHT,
                height="100%"
            ),
            
            # Use design system card style
            direction="column",
            **card_style(hover=True),
            height="100%"
        ),
        href=route,
        text_decoration="none"
    )


# ======================================================================
# 2. REDESIGNED HOME PAGE (CENTERED & LIGHTER)
# ======================================================================
def home_page() -> rx.Component:
    """
    Home page with consistent design system.
    """
    return page_layout(
        rx.container(
            rx.vstack(
                # --- Hero Section ---
                rx.vstack(
                    # Logo + Title (horizontal)
                    rx.hstack(
                        rx.image(
                            src="/Gold_Ingot.png",
                            width="64px",
                            height="64px",
                            alt="GoldSight Logo"
                        ),    
                        rx.heading(
                            "GoldSight",
                            size=Typography.HERO,  # size="9"
                            weight="bold",
                            color_scheme=Colors.PRIMARY  # Amber
                        ),
                        spacing="3",
                        align="center"
                    ),
                    # Subtitle
                    rx.heading(
                        "An AI-Powered Gold Price Prediction Journey",
                        size=Typography.SECTION,  # size="6" (not "7" - follow hierarchy)
                        color_scheme=Colors.NEUTRAL,
                        weight="medium"
                    ),
                    spacing=Spacing.COMPONENT,
                    align="center",
                    width="100%",
                    padding_top="2em",  # Reduced from 4em to 2em
                    padding_bottom="2em"
                ),

                # --- Why Predict Gold? ---
                rx.vstack(
                    rx.heading(
                        "Why Predict Gold?",
                        size=Typography.SECTION,  # size="6"
                        weight="bold",
                        align="center"
                    ),
                    rx.text(
                        "Gold is a critical asset in financial markets, serving as a hedge against inflation and "
                        "economic instability. Predicting its price helps investors make better decisions. "
                        "Traditional forecasting methods often fail to capture the complex market dynamics, "
                        "making Machine Learning an ideal approach.",
                        text_align="justify", 
                        color="var(--gray-12)",
                        size=Typography.BODY,  # size="4"
                        line_height="1.7"
                    ),
                    align="center",
                    max_width="800px",
                    spacing=Spacing.COMPONENT,
                    width="100%"
                ),

                # --- Project Objective ---
                rx.vstack(
                    rx.heading(
                        "Our Objective",
                        size=Typography.SECTION,  # size="6"
                        weight="bold",
                        align="center"
                    ),
                    rx.text(
                        "This project aims to develop a system using Machine Learning to forecast gold prices. "
                        "We collected historical gold data and 13+ related economic indicators to train predictive models. "
                        "This interactive platform visualizes trends and provides forecasts, "
                        "allowing users to analyze price models and make informed decisions.",
                        text_align="justify",
                        color="var(--gray-12)",
                        size=Typography.BODY,  # size="4"
                        line_height="1.7"
                    ),
                    align="center",
                    max_width="800px",
                    spacing=Spacing.COMPONENT,
                    width="100%"
                ),
                
                # --- Research Journey Navigation ---
                rx.vstack(
                    rx.heading(
                        "Explore Our Research Journey",
                        size=Typography.SECTION,  # size="6" (not "7")
                        weight="bold",
                        align="center"
                    ),
                    rx.text(
                        "Follow our step-by-step process, from data to deployment.",
                        size=Typography.BODY,
                        color="var(--gray-12)",
                        align="center"
                    ),
                    
                    # Navigation cards grid
                    rx.center(
                        rx.grid(
                            nav_card(
                                title="Chapter 1: The Data",
                                desc="See the 13+ market and macro indicators we collected.",
                                route="/data-collection",
                                icon_name="database",
                                icon_color=Colors.NEUTRAL
                            ),
                            nav_card(
                                title="Chapter 2: The Exploration",
                                desc="Discover the key correlations and insights from our EDA.",
                                route="/eda",
                                icon_name="bar-chart-big",
                                icon_color=Colors.WARNING  # Orange
                            ),
                            nav_card(
                                title="Chapter 3: The Models",
                                desc="Our journey comparing 1 models, from ARIMA to LSTM.",
                                route="/modeling",
                                icon_name="cpu",
                                icon_color=Colors.SECONDARY  # Blue
                            ),
                            nav_card(
                                title="Final App: The Forecast Tool",
                                desc="Try our best-performing model to get live forecasts.",
                                route="/forecast",
                                icon_name="trending-up",
                                icon_color=Colors.SUCCESS  # Green
                            ),
                            nav_card(
                                title="Source Code",
                                desc="View the complete source code and notebooks on GitHub.",
                                route="https://github.com/HuyPham171-hub/gold-price-prediction",
                                icon_name="github",
                                icon_color="gray"
                            ),
                            
                            columns="3",
                            spacing=Spacing.CONTENT,  # spacing="4"
                            width="100%"
                        ),
                        width="100%"
                    ),
                    
                    align="center",
                    spacing=Spacing.COMPONENT,
                    width="100%"
                ),
                
                # Main vstack settings
                spacing=Spacing.SECTION,  # spacing="6" (32px between sections)
                align="center",
                width="100%"
            ),
            
            max_width=Layout.MAX_WIDTH_WIDE,  # "1200px"
            padding_x="2em",
            padding_y="3em",
            margin_x="auto"
        )
    )