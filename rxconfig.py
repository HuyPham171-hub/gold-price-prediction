import reflex as rx
import os

config = rx.Config(
    app_name="goldsight",
    # API URL for production
    api_url=os.getenv("API_URL", "0.0.0.0:8000"),
    # Database URL (if needed later)
    db_url=os.getenv("DATABASE_URL", "sqlite:///reflex.db"),
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ]
)