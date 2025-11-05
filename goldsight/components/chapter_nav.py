"""Chapter navigation components for storytelling flow."""

import reflex as rx
from goldsight.utils.design_system import (
    Colors, 
    Spacing, 
    button_primary_style,
    section_divider
)


def chapter_intro(title: str, subtitle: str, intro_text: str) -> rx.Component:
    """
    Standard introduction section for chapter pages.
    
    Args:
        title: Chapter title (e.g., "Chapter 1: The Data")
        subtitle: Short subtitle
        intro_text: Narrative introduction paragraph
    """
    return rx.vstack(
        # Chapter title
        rx.heading(
            title,
            size="8",
            weight="bold",
            color_scheme=Colors.PRIMARY,
            align="center"
        ),
        
        # Subtitle
        rx.heading(
            subtitle,
            size="5",
            weight="medium",
            color_scheme=Colors.NEUTRAL,
            align="center"
        ),
        
        # Intro narrative
        rx.text(
            intro_text,
            size="4",
            color_scheme=Colors.NEUTRAL,
            text_align="justify",
            line_height="1.7"
        ),
        
        spacing=Spacing.COMPONENT,
        align="center",
        width="100%",
        margin_bottom="3em"
    )


def next_chapter_navigation(
    next_title: str,
    next_route: str,
    prev_title: str = None,
    prev_route: str = None
) -> rx.Component:
    """
    Navigation to previous/next chapters.
    
    Args:
        next_title: Title of next chapter
        next_route: Route to next chapter
        prev_title: Optional previous chapter title
        prev_route: Optional previous chapter route
    """
    return rx.box(
        section_divider(),
        
        rx.vstack(
            rx.heading(
                "Continue the Journey",
                size="6",
                weight="bold",
                align="center",
                color_scheme=Colors.NEUTRAL
            ),
            
            rx.hstack(
                # Previous chapter (if exists)
                rx.cond(
                    prev_title is not None,
                    rx.link(
                        rx.button(
                            rx.hstack(
                                rx.icon("arrow-left", size=20),
                                rx.text(prev_title),
                                spacing="2"
                            ),
                            color_scheme=Colors.NEUTRAL,
                            variant="soft",
                            size="3"
                        ),
                        href=prev_route,
                        text_decoration="none"
                    )
                ),
                
                rx.spacer(),
                
                # Next chapter
                rx.link(
                    rx.button(
                        rx.hstack(
                            rx.text(next_title),
                            rx.icon("arrow-right", size=20),
                            spacing="2"
                        ),
                        **button_primary_style()
                    ),
                    href=next_route,
                    text_decoration="none"
                ),
                
                width="100%",
                justify="between",
                align="center"
            ),
            
            spacing=Spacing.COMPONENT,
            align="center",
            width="100%",
            padding="2em",
            background_color=rx.color(Colors.NEUTRAL, 2),
            border_radius="var(--radius-4)",
            border=f"1px solid {rx.color(Colors.NEUTRAL, 4)}"
        ),
        
        padding_y="3em",
        width="100%"
    )


def chapter_progress(current: int) -> rx.Component:
    """
    Fixed layout progress bar for 4 chapters.
    Args:
        current: Current chapter number (1-4)
    """
    
    def circle(number: str, label: str, position: int) -> rx.Component:
        is_current = (position == current)
        is_completed = (position < current)
        
        return rx.vstack(
            rx.box(
                rx.text(
                    number,
                    size="5",
                    weight="bold",
                    color=rx.cond(is_current, "white", rx.color("gray", 11))
                ),
                width="50px",
                height="50px",
                border_radius="50%",
                background_color=rx.cond(
                    is_current,
                    rx.color("amber", 9),
                    rx.color("gray", 3)
                ),
                border=rx.cond(
                    is_completed | is_current,
                    f"2px solid {rx.color('amber', 9)}",
                    f"2px solid {rx.color('gray', 5)}"
                ),
                display="flex",
                align_items="center",
                justify_content="center"
            ),
            rx.text(
                label,
                size="2",
                weight=rx.cond(is_current, "bold", "medium"),
                color=rx.cond(
                    is_current,
                    rx.color("amber", 11),
                    rx.color("gray", 10)
                ),
                text_align="center",
                white_space="nowrap"
            ),
            spacing="2",
            align="center",
            width="110px",
            flex_shrink="0"
        )
    
    def connector_line(position: int) -> rx.Component:
        is_completed = (position < current)
        
        return rx.box(
            width="80px",
            height="4px",
            background_color=rx.cond(
                is_completed,
                rx.color("amber", 9),
                rx.color("gray", 5)
            ),
            border_radius="2px",
            margin_top="23px",
            margin_x="0.75em",
            flex_shrink="0"
        )
    
    return rx.flex(
        circle("1", "Data Collection", 1),
        connector_line(1),
        circle("2", "EDA", 2),
        connector_line(2),
        circle("3", "Modeling", 3),
        connector_line(3),
        circle("4", "Forecast", 4),
        align_items="flex-start",
        justify_content="center",
        width="100%",
        max_width="800px",
        margin_x="auto",
        margin_bottom="2em",
        padding_x="2em"
    )
