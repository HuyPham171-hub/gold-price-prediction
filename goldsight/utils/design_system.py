"""
GoldSight Design System
-----------------------
Centralized design tokens for consistent styling across the application.

Color Scheme:
- Primary (Accent): Amber (gold theme)
- Secondary: Blue (trust, finance)
- Neutral: Gray (90% of UI)
- Semantic: Green (success), Red (error)

Typography Scale:
- size="9": Hero titles (1x per page)
- size="7": Chapter titles
- size="6": Section headings
- size="5": Subsection headings
- size="4": Body text (primary)
- size="3": Small text, captions

Spacing Scale (Radix):
- spacing="6": Between major sections (32px)
- spacing="4": Between content blocks (16px)
- spacing="3": Between related items (12px)

Layout:
- max_width="1200px": Homepage, wide layouts
- max_width="900px": Content pages (blog/research style)
"""

import reflex as rx

# ============================================================================
# COLOR TOKENS
# ============================================================================

class Colors:
    """Color scheme using Radix color names."""
    
    # Primary/Accent (Gold/Amber theme)
    PRIMARY = "amber"
    
    # Secondary (Trust/Finance)
    SECONDARY = "blue"
    
    # Neutrals
    NEUTRAL = "gray"
    
    # Semantic
    SUCCESS = "green"
    ERROR = "red"
    WARNING = "orange"
    INFO = "blue"
    
    @staticmethod
    def bg_primary():
        """Primary background (very light gray, almost white)."""
        return rx.color("gray", 1)
    
    @staticmethod
    def bg_secondary():
        """Secondary background (light gray for cards)."""
        return rx.color("gray", 2)
    
    @staticmethod
    def border():
        """Default border color."""
        return rx.color("gray", 5)
    
    @staticmethod
    def text_primary():
        """Primary text color (dark gray/black)."""
        return rx.color("gray", 12)
    
    @staticmethod
    def text_secondary():
        """Secondary text color (medium gray)."""
        return rx.color("gray", 11)


# ============================================================================
# SPACING TOKENS
# ============================================================================

class Spacing:
    """Consistent spacing scale."""
    
    SECTION = "6"      # 32px - Between major sections
    CONTENT = "4"      # 16px - Between content blocks
    COMPONENT = "3"    # 12px - Between related components
    TIGHT = "2"        # 8px - Tight spacing


# ============================================================================
# TYPOGRAPHY TOKENS
# ============================================================================

class Typography:
    """Font size hierarchy."""
    
    HERO = "9"          # Hero title (1x per page)
    CHAPTER = "7"       # Chapter titles
    SECTION = "6"       # Section headings
    SUBSECTION = "5"    # Subsection headings
    BODY = "4"          # Primary body text
    SMALL = "3"         # Small text, captions
    TINY = "2"          # Very small text


# ============================================================================
# LAYOUT TOKENS
# ============================================================================

class Layout:
    """Layout dimensions."""
    
    MAX_WIDTH_WIDE = "1200px"   # Homepage, dashboards
    MAX_WIDTH_CONTENT = "900px"  # Content pages (blog style)
    MAX_WIDTH_NARROW = "700px"   # Very focused content


# ============================================================================
# REUSABLE STYLE DICTIONARIES
# ============================================================================

def card_style(hover: bool = True) -> dict:
    """Standard card styling."""
    base = {
        "background_color": Colors.bg_secondary(),
        "border": f"1px solid {Colors.border()}",
        "border_radius": "var(--radius-4)",
        "padding": "1.5em",
    }
    
    if hover:
        base["_hover"] = {
            "border_color": rx.color(Colors.SECONDARY, 7),
            "box_shadow": "0 4px 12px 0 rgba(0, 0, 0, 0.05)",
            "transform": "translateY(-2px)"
        }
        base["transition"] = "all 0.2s ease-in-out"
    
    return base


def button_primary_style() -> dict:
    """Primary button (Amber/Gold CTA)."""
    return {
        "color_scheme": Colors.PRIMARY,
        "size": "3",
        "variant": "solid"
    }


def button_secondary_style() -> dict:
    """Secondary button (Gray/Subtle)."""
    return {
        "color_scheme": Colors.NEUTRAL,
        "size": "3",
        "variant": "soft"
    }


def section_divider() -> rx.Component:
    """Standard section divider."""
    return rx.divider(
        margin_y="3em",
        border_color=Colors.border()
    )


# ============================================================================
# COMPONENT BUILDERS
# ============================================================================

def chapter_header(title: str, subtitle: str = "") -> rx.Component:
    """Standard chapter header for content pages."""
    return rx.vstack(
        rx.heading(
            title,
            size=Typography.CHAPTER,
            weight="bold",
            color_scheme=Colors.PRIMARY
        ),
        rx.cond(
            subtitle != "",
            rx.text(
                subtitle,
                size=Typography.BODY,
                color_scheme=Colors.NEUTRAL,
                text_align="center"
            )
        ),
        spacing=Spacing.COMPONENT,
        align="center",
        margin_bottom="2em"
    )


def section_header(title: str) -> rx.Component:
    """Standard section header."""
    return rx.heading(
        title,
        size=Typography.SECTION,
        weight="bold",
        margin_top="2em",
        margin_bottom="1em"
    )


def next_chapter_cta(title: str, route: str) -> rx.Component:
    """Call-to-action button to navigate to next chapter."""
    return rx.box(
        rx.link(
            rx.button(
                f"{title} ➔",
                **button_primary_style()
            ),
            href=route
        ),
        padding_y="3em",
        text_align="center",
        width="100%"
    )


def content_container(children: list, max_width: str = Layout.MAX_WIDTH_CONTENT) -> rx.Component:
    """
    Standard content container for blog-style pages.
    Centers content and applies max-width.
    """
    return rx.container(
        rx.vstack(
            *children,
            spacing=Spacing.CONTENT,
            align="start",
            width="100%"
        ),
        max_width=max_width,
        padding_x="2em",
        padding_y="3em"
    )
