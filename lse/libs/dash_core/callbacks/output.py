import logging
import os
from typing import Any

import dash
import dill as pickle
import flask
import werkzeug
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from lse.libs.backend.io.utils import read_feathered_dataframe
from lse.libs.dash_core.callbacks.common import _reconstruct_data
from lse.libs.dash_core.callbacks.utils import (
    build_download_button,
    get_datapoints_from_selection_or_hover,
    load_offline_session,
    load_online_session,
)
from lse.libs.dash_core.components_models import (
    CheckpointComponent,
    DropdownComponent,
    ExportComponent,
    OutputComponent,
    PlotAreaComponent,
    PlotControlComponent,
    SessionComponent,
    StatusComponent,
)
from lse.libs.data_models import VariableReference
from lse.libs.utils.data_table_formatting import DataTableFormatter
from lse.utils.cache import FlaskCacheManager
from lse.utils.constants import MODEL_EXPORT_DIRECTORY

logger = logging.getLogger()

CACHE_MANAGER = FlaskCacheManager()


def add_callbacks(  # noqa: C901
    app: dash.Dash,
) -> None:

    # Data table
    @app.callback(
        Output(OutputComponent.EXPORT_DATA_TABLE, "data"),
        Output(OutputComponent.EXPORT_DATA_TABLE, "columns"),
        Output(OutputComponent.EXPORT_DATA_TABLE, "hidden_columns"),
        Output(OutputComponent.EXPORT_DATA_TABLE, "style_data_conditional"),
        Input(CheckpointComponent.FILTERED_DATA, "data"),
        Input(PlotAreaComponent.DATA_PROJECTION, "selectedData"),
        Input(DropdownComponent.SIGNATURE, "value"),
        Input(PlotControlComponent.VARIANCE_SLIDER, "value"),
        State(SessionComponent.SESSION_ID, "data"),
        prevent_initial_call=True,
    )
    def display_reconstructed_data_table(
        filtered_data_check: bool,
        selected_data: dict[str, Any] | None,
        signature_variables: list[str],
        var_explained: float,
        session_id: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
        ctx = dash.callback_context
        logger.debug(
            f"{display_reconstructed_data_table.__name__} triggered by {ctx.triggered[0]['prop_id'].split('.')[0]} property"
        )

        if filtered_data_check:
            reconstruct_data = CACHE_MANAGER.memoize_function(_reconstruct_data)

            app_backend = load_online_session(session_id)

            # Join reconstructed data with probas, mixture component prediction,
            # and original identifiers
            data = (
                app_backend.user_session.filtered_data.loc[
                    :,
                    [
                        col
                        for col in app_backend.user_session.data_wrapper.data.columns
                        if col not in app_backend.user_session.features
                    ],
                ]
                .join(app_backend.compute_datapoints_probas())
                .join(app_backend.predict_dominant_component())
                .join(
                    read_feathered_dataframe(
                        reconstruct_data(
                            session_id=session_id,
                            signature_variables=signature_variables,
                            var_explained=var_explained,
                        )
                    )
                )
            )
            data_table_formatter = DataTableFormatter.format_table(
                get_datapoints_from_selection_or_hover(data=data, selected_data=selected_data)
            )
            styles = data_table_formatter.get_style(
                columns_to_style=signature_variables
                + [VariableReference.LOG_PROBABILITY, VariableReference.DOMINANT_COMPONENT]
            )

            return (
                data_table_formatter.data_table,
                data_table_formatter.columns,
                data_table_formatter.hidden_columns,
                styles,
            )

        raise PreventUpdate

    # Export model
    @app.callback(
        Output(OutputComponent.DOWNLOAD_AREA, "children", allow_duplicate=True),
        Output(StatusComponent.EXPORT_STATUS, "children", allow_duplicate=True),
        Input(ExportComponent.PREPARE_EXPORT_LATENT_SPACE_MODEL, "n_clicks"),
        State(SessionComponent.SESSION_ID, "data"),
        prevent_initial_call="initial_duplicate",
    )
    def export_latent_model(n_clicks: int, session_id: str) -> tuple[html.Form, str]:
        if n_clicks > 0:
            user_session_parameters = load_online_session(session_id).user_session_parameters
            app_backend = load_offline_session(
                session_id, user_session_parameters=user_session_parameters
            )

            updated_export_status = "Latent model ready for download: "
            if n_clicks > 1:
                updated_export_status = f"Update {n_clicks} of latent model ready for download: "

            filename = (
                "_".join((session_id, app_backend.__class__.__name__, f"v{n_clicks}")) + ".pkl"
            )
            uri = os.path.join(MODEL_EXPORT_DIRECTORY, filename)

            with open(uri, "wb") as file:
                pickle.dump(app_backend, file)

            return build_download_button(uri), updated_export_status
        raise PreventUpdate

    @app.server.route(f"/{MODEL_EXPORT_DIRECTORY}/<path:filename>")
    def download_link(filename: str) -> werkzeug.wrappers.response.Response:
        @flask.after_this_request
        def delete_file(
            response: werkzeug.wrappers.response.Response,
        ) -> werkzeug.wrappers.response.Response:
            try:
                os.remove(os.path.join(MODEL_EXPORT_DIRECTORY, filename))
            except Exception as exc:
                logger.error(f"Error deleting file: {exc}")
            return response

        return flask.send_from_directory(MODEL_EXPORT_DIRECTORY, filename, as_attachment=True)
