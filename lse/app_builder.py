"""Main module for assembling Dash app."""

import locale
import logging

import pandas as pd
from dash import Dash

from lse.libs.dash_core import layout
from lse.libs.dash_core.callbacks import (
    add_computation_callbacks,
    add_data_callbacks,
    add_dropdowns_callbacks,
    add_output_callbacks,
    add_plot_callbacks,
)
from lse.utils.cache import FlaskCacheManager
from lse.utils.constants import ASSETS_PATH
from lse.utils.database import init_redis_client
from lse.utils.logging_config import configure_logging

# Apply logging configuration
configure_logging()

logger = logging.getLogger(__name__)


def build_dash_app(  # noqa: C901, PLR0915, PLR0912
    assets_path: str | None = None,
    dashboard_title: str | None = None,
    data: pd.DataFrame | None = None,
    disable_input_data_tables: bool = False,
) -> Dash:
    """Builds the dash webapp.

    Args:
        assets_path:
            A string of the path of the css stylesheet.
        dashboard_title:
            A string used for title of the dashboard.
        data:
            A :obj:`pandas.DataFrame` to explore.
        disable_input_data_tables:
            If True, hides the input data table.

    Returns:
        A :obj:`dash.Dash` application.
    """
    # App title
    dashboard_title = (
        "Mutidimensional data explorer" if dashboard_title is None else dashboard_title
    )

    # App style
    assets_path = ASSETS_PATH if assets_path is None else assets_path

    # Set locale
    locale.setlocale(locale.LC_ALL, "")

    # Initialize app
    _app = Dash(
        __package__,
        assets_folder=assets_path,
        suppress_callback_exceptions=True,
    )

    # Build layout
    serve_layout = layout.build_html_canvas(
        dashboard_title=dashboard_title, enable_data_upload=(data is None)
    )
    _app.layout = serve_layout
    _app.title = dashboard_title

    # Init Redis
    init_redis_client()
    logger.info("Redis database connection established!")

    # Init Flask cache manager
    FlaskCacheManager().init_app_cache(_app)
    logger.info("Flask cache manager initialized!")

    ### Callbacks
    ## Open session and prepare data
    add_data_callbacks(
        _app,
        data=data,
        disable_input_data_tables=disable_input_data_tables,
    )
    ## Update dropdowns
    add_dropdowns_callbacks(_app)
    ## Computation
    add_computation_callbacks(_app)
    ## Plotting
    add_plot_callbacks(_app)
    ## Tables
    add_output_callbacks(_app)

    return _app
