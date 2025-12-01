"""GoldSight - Gold Price Prediction Web Application."""

import reflex as rx
from rxconfig import config

# Import pages
from goldsight.pages.home import home_page
from goldsight.pages.data_collection import data_collection_page
from goldsight.pages.eda import eda_page
from goldsight.pages.modeling import modeling_page
from goldsight.pages.forecast import forecast_page


class State(rx.State):
    """The app state."""
    pass


# Create app with Amber (Gold) theme
app = rx.App(
    theme=rx.theme(
        appearance="light",
        accent_color="amber",  # Gold/Amber for finance theme
        gray_color="gray",     # Neutral colors
        radius="large",        # Rounded corners
    ),
)

# Add pages with routes
app.add_page(home_page, route="/", title="Home - Gold Price Prediction")
app.add_page(data_collection_page, route="/data-collection", title="Data Collection")
app.add_page(eda_page, route="/eda", title="Exploratory Data Analysis")
app.add_page(modeling_page, route="/modeling", title="Model Training")
app.add_page(forecast_page, route="/forecast", title="Forecast")
