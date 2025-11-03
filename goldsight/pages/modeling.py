"""Modeling page - Model training and evaluation."""
import reflex as rx


def modeling_page() -> rx.Component:
    """Modeling page component."""
    return rx.container(
        rx.vstack(
            rx.heading("Model Training & Evaluation", size="9", margin_bottom="2rem"),
            
            rx.text(
                "Nội dung về các models sẽ được thêm vào đây.",
                size="4",
                color_scheme="gray"
            ),
            
            rx.link(
                rx.button("← Back to Home"),
                href="/"
            ),
            
            spacing="4",
            padding="2rem"
        )
    )
