"""Reusable button components following design system."""

import reflex as rx
from goldsight.utils.design_system import Colors, button_primary_style, button_secondary_style


def primary_button(text: str, on_click=None, **kwargs) -> rx.Component:
    """
    Primary call-to-action button (Amber/Gold).
    Use for main actions: "Run Forecast", "Next Chapter", etc.
    """
    return rx.button(
        text,
        on_click=on_click,
        **button_primary_style(),
        **kwargs
    )


def secondary_button(text: str, on_click=None, **kwargs) -> rx.Component:
    """
    Secondary button (Gray/Subtle).
    Use for secondary actions: "Download Data", "View Details", etc.
    """
    return rx.button(
        text,
        on_click=on_click,
        **button_secondary_style(),
        **kwargs
    )


def link_button(text: str, href: str, primary: bool = True, **kwargs) -> rx.Component:
    """
    Button styled as a link.
    
    Args:
        text: Button text
        href: Link destination
        primary: If True, uses amber theme. If False, uses gray theme.
    """
    style = button_primary_style() if primary else button_secondary_style()
    
    return rx.link(
        rx.button(text, **style, **kwargs),
        href=href,
        text_decoration="none"
    )


def icon_button(icon: str, on_click=None, color_scheme: str = Colors.PRIMARY, **kwargs) -> rx.Component:
    """
    Icon-only button.
    
    Args:
        icon: Icon name (e.g., "download", "external-link")
        on_click: Click handler
        color_scheme: Color theme
    """
    return rx.button(
        rx.icon(icon, size=20),
        on_click=on_click,
        color_scheme=color_scheme,
        variant="soft",
        size="2",
        **kwargs
    )
