"""Data Collection page - Data sources and pipeline."""
import reflex as rx


def data_collection_page() -> rx.Component:
    """Data collection page component."""
    return rx.container(
        rx.vstack(
            rx.heading("Data Collection & Processing", size="9", margin_bottom="2rem"),
            
            rx.text(
                "Nội dung chi tiết về thu thập dữ liệu sẽ được thêm vào đây.",
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
