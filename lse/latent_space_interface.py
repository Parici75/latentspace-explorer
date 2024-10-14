import dash
import pandas as pd

from lse.app_builder import build_dash_app


class LatentSpaceInterface:
    """Entry class to interface with the Dash app from a Python process.

    Attributes:
        data:
            A {obj}`pandas.DataFrame` to explore.
        host:
            The address to expose the app on.
        port:
            The port the app listen to.
        dashboard_title:
            The title of the dashboard.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        host: str = "127.0.0.1",
        port: str = "8050",
        dashboard_title: str | None = None,
    ) -> None:
        self.data = data
        self.host = host
        self.port = port
        self.dashboard_title = dashboard_title

    def start(
        self, disable_input_data_tables: bool = False, jupyter_mode: str = "external"
    ) -> dash.Dash:
        app = build_dash_app(
            dashboard_title=self.dashboard_title,
            data=self.data,
            disable_input_data_tables=disable_input_data_tables,
        )
        app.run(
            jupyter_mode=jupyter_mode,
            host=self.host,
            port=self.port,
            dev_tools_hot_reload=True,
            dev_tools_ui=True,
            debug=True,
        )

        return app
