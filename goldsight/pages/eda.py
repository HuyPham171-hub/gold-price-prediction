"""EDA page - Exploratory Data Analysis."""
import reflex as rx


def eda_page() -> rx.Component:
    """EDA page component."""
    return rx.container(
        rx.vstack(
            rx.heading("Exploratory Data Analysis", size="9", margin_bottom="2rem"),
            
            rx.text(
                "Nội dung phân tích dữ liệu sẽ được thêm vào đây.",
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
