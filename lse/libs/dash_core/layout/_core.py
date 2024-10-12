# noqa: E501

import uuid
from collections.abc import Callable
from typing import Any

from dash import dcc, html

from lse.libs.dash_core import components_models as cm

from ._explore_tab import build_tab as build_explore_tab
from ._export_tab import build_tab as build_export_tab
from ._import_tab import build_tab as build_import_tab


def build_html_canvas(dashboard_title: str, enable_data_upload: bool) -> Callable[..., Any]:
    """Returns a function building the HTML layout."""

    def serve_layout() -> html.Div:
        """Serves HTML layout.

        When Dash sees a function for the layout property,
        it evaluates the function on each page load
        (as opposed to only evaluating it only once on the initial app load).
        """
        session_id = str(uuid.uuid4())

        layout = html.Div(
            [
                dcc.Store(data=session_id, id=cm.SessionComponent.SESSION_ID),
                dcc.Tabs(
                    [
                        build_import_tab(
                            dashboard_title=dashboard_title, enable_data_upload=enable_data_upload
                        ),
                        build_explore_tab(),
                        build_export_tab(),
                    ],
                    className="tabs",
                ),
            ],
            className="container",
        )

        return layout

    return serve_layout
