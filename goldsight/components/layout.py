"""Layout components to handle navbar spacing and page structure."""

import reflex as rx
from .navbar import navbar


def page_layout(*children) -> rx.Component:
    """
    Wraps page content with navbar and proper spacing.
    Use this instead of rx.fragment(navbar(), ...) to prevent content from being hidden under sticky navbar.
    
    Example:
        def my_page():
            return page_layout(
                rx.heading("My Page"),
                rx.text("Content here...")
            )
    """
    return rx.fragment(
        navbar(),
        rx.box(
            *children,
            padding_top="80px",  # Offset for sticky navbar height
            min_height="100vh"    # Full viewport height
        )
    )
