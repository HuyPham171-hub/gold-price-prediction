# goldsight/pages/home.py
import reflex as rx
from goldsight.components import navbar  # Assuming you have a navbar

# ======================================================================
# 1. HELPER COMPONENT: NAV_CARD
# (Based on your image reference)
# ======================================================================
def nav_card(
    title: str,
    desc: str,
    route: str,
    icon_name: str,
    icon_color: str = "blue"
) -> rx.Component:
    """
    A clickable navigation card, styled after the user's reference image.
    """
    return rx.link(
        rx.flex(
            # Vstack for all content
            rx.vstack(
                # Icon
                rx.icon(
                    tag=icon_name,
                    size=32,
                    color_scheme=icon_color,
                    margin_bottom="0.5em"
                ),
                # Title
                rx.heading(
                    title,
                    size="5",
                    weight="bold",
                    color_scheme="gray"
                ),
                # Description
                rx.text(
                    desc,
                    color_scheme="gray",
                    size="2"  # Small font for description
                ),
                
                # Spacer to push the "Learn more" link to the bottom
                rx.spacer(),
                
                # "Learn more" link
                rx.hstack(
                    rx.text("Learn more", color_scheme="blue", size="2"),
                    rx.icon(
                        tag="arrow-right",
                        size=16,
                        color_scheme="blue"
                    ),
                    align="center",
                    spacing="1",
                    margin_top="1.5em"  # Create space
                ),
                
                align="start",  # Align content to the left
                spacing="2",
                height="100%"  # Helps the spacer work correctly
            ),
            
            # Card styling
            direction="column",
            background_color=rx.color("gray", 2),  # Very light gray background
            border_radius="var(--radius-4)",  # Soft corners
            border="1px solid",
            border_color=rx.color("gray", 4),  # Light border
            padding="1.5em",
            height="100%",  # Important for grid layout
            
            # Hover effect
            _hover={
                "border_color": rx.color("blue", 7),
                "box_shadow": "0 4px 12px 0 rgba(0, 0, 0, 0.05)",
                "transform": "translateY(-2px)"
            },
            transition="all 0.2s ease-in-out"
        ),
        href=route,
        text_decoration="none",  # Remove link underline
    )


# ======================================================================
# 2. REDESIGNED HOME PAGE (STORYTELLING STYLE)
# ======================================================================
def home_page() -> rx.Component:
    """
    The redesigned home page, focusing on storytelling like a research blog.
    All components are centered, and text is justified.
    """
    return rx.fragment(
        navbar(),
        
        # The Container component sets a max_width and centers itself.
        rx.container(
            
            # This main Vstack centers all of its children (the sections).
            rx.vstack(
                
                # --- Section 1: Hero ---
                # This section's content should be centered.
                rx.vstack(
                    rx.heading(
                        "🔱 GoldSight",
                        size="9",
                        weight="bold",
                        color_scheme="yellow"  # Gold accent
                    ),
                    rx.heading(
                        "An AI-Powered Gold Price Prediction Journey",
                        size="7",
                        color_scheme="gray",
                        weight="medium"
                    ),
                    spacing="3",
                    align="center",
                    padding_top="4em",
                    padding_bottom="2em",
                    width="100%"
                ),

                # --- Section 2: Why Predict Gold? (The Story) ---
                # This Vstack is centered by its parent.
                # Its own alignment is "start" (left) to allow text justification.
                rx.vstack(
                    rx.heading("Why Predict Gold?", size="6", weight="bold"),
                    rx.text(
                        "Gold is a critical asset in financial markets, serving as a hedge against inflation and ",
                        "economic instability. Predicting its price helps investors make better decisions. ",
                        "Traditional forecasting methods often fail to capture the complex market dynamics, ",
                        "making Machine Learning an ideal approach.",
                        # This is the key change for justify
                        text_align="justify", 
                        color_scheme="gray",
                        size="4"
                    ),
                    align="start", # Changed from "center"
                    max_width="800px", # Keep text readable
                    spacing="3"
                ),

                # --- Section 3: Project Objective ---
                # Same as above: centered block, justified text.
                rx.vstack(
                    rx.heading("Our Objective", size="6", weight="bold"),
                    rx.text(
                        "This project aims to develop a system using Machine Learning to forecast gold prices. ",
                        "We collected historical gold data and 13+ related economic indicators to train predictive models. ",
                        "This interactive platform visualizes trends and provides forecasts, ",
                        "allowing users to analyze price models and make informed decisions.",
                        # This is the key change for justify
                        text_align="justify",
                        color_scheme="gray",
                        size="4"
                    ),
                    align="start", # Changed from "center"
                    max_width="800px", # Keep text readable
                    spacing="3"
                ),
                
                # --- Section 4: The Research Journey (Navigation) ---
                # This section's content (grid) should be centered.
                rx.vstack(
                    rx.heading("Explore Our Research Journey", size="7", padding_top="1em"),
                    rx.text(
                        "Follow our step-by-step process, from data to deployment.",
                        size="4",
                        color_scheme="gray"
                    ),
                    
                    rx.grid(
                        nav_card(
                            title="Chapter 1: The Data",
                            desc="See the 13+ market and macro indicators we collected.",
                            route="/data-collection",
                            icon_name="database",
                            icon_color="gray"
                        ),
                        nav_card(
                            title="Chapter 2: The Exploration",
                            desc="Discover the key correlations and insights from our EDA.",
                            route="/eda",
                            icon_name="bar-chart-big",
                            icon_color="orange"
                        ),
                        nav_card(
                            title="Chapter 3: The Models",
                            desc="Our journey comparing 8 models, from ARIMA to LSTM.",
                            route="/modeling",
                            icon_name="cpu"
                        ),
                        nav_card(
                            title="Chapter 4: The Results",
                            desc="Analyze model performance, error, and key findings.",
                            route="/insights",
                            icon_name="pie-chart",
                            icon_color="purple"
                        ),
                        nav_card(
                            title="Final App: The Forecast Tool",
                            desc="Try our best-performing LSTM model to get live forecasts.",
                            route="/forecast",
                            icon_name="trending-up",
                            icon_color="green"
                        ),
                        nav_card(
                            title="Source Code",
                            desc="View the complete source code and notebooks on GitHub.",
                            route="/api-docs",
                            icon_name="github",
                            icon_color="black"
                        ),
                        
                        columns="3",  # 3-column grid
                        spacing="4",  # Spacing between cards
                        width="100%",
                        max_width="1000px", # Constrain grid width
                        margin_top="2em"
                    ),
                    align="center",
                    spacing="3",
                    width="100%"
                ),
                
                # --- Main Vstack Settings ---
                spacing="7", # Larger spacing between main sections
                align="center", # This centers all children Vstacks
                padding_x="2em",
                padding_bottom="4em"
            ),
            
            # --- Container Settings ---
            max_width="1200px", # Max width of the content
            padding=0 # Ensure container has no padding to conflict with vstack
        )
    )