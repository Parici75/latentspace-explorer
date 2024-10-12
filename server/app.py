# noqa: INP001

import logging
import os

from lse.app_builder import build_dash_app
from lse.utils.strtobool import strtobool

# Redirect Dash app logs on the gunicorn logger
gunicorn_logger = logging.getLogger("gunicorn.error")
root_logger = logging.getLogger()
root_logger.handlers = gunicorn_logger.handlers
root_logger.setLevel(gunicorn_logger.level)  # Unify gunicorn and Dash app logging level

# App
app = build_dash_app(
    assets_path=os.environ.get("ASSETS_PATH", None),
    dashboard_title=os.environ.get("DASHBOARD_TITLE", None),
    disable_input_data_tables=strtobool(os.environ.get("DISABLE_DATA_TABLES", False)),
)

server = app.server
