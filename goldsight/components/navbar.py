import reflex as rx


class NavbarState(rx.State):
    """State for navbar to track current route."""
    
    @rx.var
    def current_route(self) -> str:
        """Get current route path, ensuring it's never empty."""
        path = self.router.url.path
        # Return "/" if path is empty or None
        return path if path else "/"
    
    @rx.var
    def is_home_active(self) -> bool:
        """Check if home page is active."""
        return self.current_route == "/"
    
    @rx.var
    def is_data_collection_active(self) -> bool:
        """Check if data collection page is active."""
        current = self.current_route
        return current == "/data-collection" or current == "/data-collection/"
    
    @rx.var
    def is_eda_active(self) -> bool:
        """Check if EDA page is active."""
        current = self.current_route
        return current == "/eda" or current == "/eda/"
    
    @rx.var
    def is_modeling_active(self) -> bool:
        """Check if modeling page is active."""
        current = self.current_route
        return current == "/modeling" or current == "/modeling/"
    
    @rx.var
    def is_forecast_active(self) -> bool:
        """Check if forecast page is active."""
        current = self.current_route
        return current == "/forecast" or current == "/forecast/"


def navbar() -> rx.Component:
    """Navigation bar with professional gold theme and active page highlighting."""
    
    return rx.box(
        rx.container(
            rx.hstack(
                # Logo
                rx.image(
                        src="/Gold_Ingot.png",
                        width="40px",
                        height="40px",
                        alt="GoldSight Logo"
                ),
                rx.heading(
                    "GoldSight",
                    size="7",
                    color_scheme="amber",
                    weight="bold"
                ),
                
                rx.spacer(),
                
                # Navigation links
                rx.hstack(
                    rx.link(
                        "Home",
                        href="/",
                        color=rx.cond(
                            NavbarState.is_home_active,
                            rx.color("amber", 9),  # Active
                            rx.color("gray", 11)  # Inactive
                        ),
                        _hover={"color": rx.color("amber", 9)},
                        font_weight=rx.cond(
                            NavbarState.is_home_active,
                            "bold",
                            "medium"
                        ),
                        text_decoration=rx.cond(
                            NavbarState.is_home_active,
                            "underline",
                            "none"
                        ),
                        text_underline_offset="4px"
                    ),
                    rx.link(
                        "Data Collection",
                        href="/data-collection",
                        color=rx.cond(
                            NavbarState.is_data_collection_active,
                            rx.color("amber", 9),
                            rx.color("gray", 11)
                        ),
                        _hover={"color": rx.color("amber", 9)},
                        font_weight=rx.cond(
                            NavbarState.is_data_collection_active,
                            "bold",
                            "medium"
                        ),
                        text_decoration=rx.cond(
                            NavbarState.is_data_collection_active,
                            "underline",
                            "none"
                        ),
                        text_underline_offset="4px"
                    ),
                    rx.link(
                        "EDA",
                        href="/eda",
                        color=rx.cond(
                            NavbarState.is_eda_active,
                            rx.color("amber", 9),
                            rx.color("gray", 11)
                        ),
                        _hover={"color": rx.color("amber", 9)},
                        font_weight=rx.cond(
                            NavbarState.is_eda_active,
                            "bold",
                            "medium"
                        ),
                        text_decoration=rx.cond(
                            NavbarState.is_eda_active,
                            "underline",
                            "none"
                        ),
                        text_underline_offset="4px"
                    ),
                    rx.link(
                        "Modeling",
                        href="/modeling",
                        color=rx.cond(
                            NavbarState.is_modeling_active,
                            rx.color("amber", 9),
                            rx.color("gray", 11)
                        ),
                        _hover={"color": rx.color("amber", 9)},
                        font_weight=rx.cond(
                            NavbarState.is_modeling_active,
                            "bold",
                            "medium"
                        ),
                        text_decoration=rx.cond(
                            NavbarState.is_modeling_active,
                            "underline",
                            "none"
                        ),
                        text_underline_offset="4px"
                    ),
                    rx.link(
                        "Forecast",
                        href="/forecast",
                        color=rx.cond(
                            NavbarState.is_forecast_active,
                            rx.color("amber", 11),
                            rx.color("amber", 9)
                        ),
                        _hover={"color": rx.color("amber", 11)},
                        font_weight="bold",
                        text_decoration=rx.cond(
                            NavbarState.is_forecast_active,
                            "underline",
                            "none"
                        ),
                        text_underline_offset="4px"
                    ),
                    spacing="9",
                    align="center"
                ),
                
                width="100%",
                align="center",
                justify="between"
            ),
            max_width="1200px"
        ),
        
        background_color=rx.color("gray", 1),
        border_bottom=f"1px solid {rx.color('gray', 4)}",
        padding_y="1em",
        width="100%",
        position="sticky",
        top="0",
        z_index="1000",
        box_shadow="0 1px 3px 0 rgba(0, 0, 0, 0.05)",
        backdrop_filter="blur(8px)"
    )