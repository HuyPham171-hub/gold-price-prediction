import reflex as rx

def navbar() -> rx.Component:
    """Navigation bar for all pages."""
    return rx.box(
        rx.hstack(
            # Logo
            rx.heading("🔱 GoldSight", size="7", color="gold"),
            
            # Navigation links
            rx.link("Home", href="/", color="white"),
            rx.link("Data Collection", href="/data-collection", color="white"),
            rx.link("EDA", href="/eda", color="white"),
            rx.link("Modeling", href="/modeling", color="white"),
            rx.link("Forecast", href="/forecast", color="white"),
            
            justify="start", 
            spacing="6",            
            align="center",
            padding="1em",
            width="100%",
        ),
        bg="linear-gradient(90deg, #1a202c 0%, #2d3748 100%)",
        width="100%",
        position="sticky",
        top="0",
        z_index="1000",
    )