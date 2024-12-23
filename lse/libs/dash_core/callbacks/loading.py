import logging
from typing import Any

import dash
import pandas as pd
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from lse.libs.backend import process_and_cache_data
from lse.libs.dash_core.callbacks.utils import (
    get_datapoints_from_selection_or_hover,
    load_online_session,
)
from lse.libs.dash_core.components_models import (
    CheckpointComponent,
    ComputeComponent,
    DataLoadingComponent,
    OutputComponent,
    PlotAreaComponent,
    SessionComponent,
    StatusComponent,
)
from lse.libs.utils.data_loading import parse_spreadsheet_contents
from lse.libs.utils.data_table_formatting import DataTableFormatter

logger = logging.getLogger()


def add_callbacks(  # noqa: C901
    app: dash.Dash,
    data: pd.DataFrame | None = None,
    disable_input_data_tables: bool = False,
) -> None:
    if data is None:

        @app.callback(
            Output(CheckpointComponent.DATA, "data"),
            Output(DataLoadingComponent.UPLOAD_DATA, "contents"),
            Input(DataLoadingComponent.UPLOAD_DATA, "contents"),
            State(DataLoadingComponent.UPLOAD_DATA, "filename"),
            State(SessionComponent.SESSION_ID, "data"),
            prevent_initial_call=True,
        )
        def upload_data(
            contents: list[Any], filenames: list[str], session_id: str
        ) -> tuple[bool, None]:
            if contents is not None:
                process_and_cache_data(
                    session_id=session_id,
                    data=pd.concat(
                        [
                            parse_spreadsheet_contents(content, name)
                            for content, name in zip(contents, filenames, strict=True)
                        ]
                    ),
                )

                return (
                    True,
                    None,
                )  # Always return None for contents of the UPLOAD_DATA component to flush it and allow reuploading the same file (see https://github.com/plotly/dash-core-components/issues/816)

            raise PreventUpdate

    else:

        @app.callback(
            Output(CheckpointComponent.DATA, "data"),
            Input(DataLoadingComponent.LOAD_DATA, "n_clicks"),
            State(SessionComponent.SESSION_ID, "data"),
            prevent_initial_call=True,
        )
        def load_data(n_clicks: int, session_id: str) -> bool:
            if n_clicks > 0:
                process_and_cache_data(
                    session_id=session_id,
                    data=data,
                )
                return True
            else:
                raise PreventUpdate

    @app.callback(
        Output(CheckpointComponent.NUMERIC_FEATURES, "data"),
        Output(ComputeComponent.RUN_MODEL_PIPELINE, "disabled"),
        Output(StatusComponent.LOADING_STATUS, "children", allow_duplicate=True),
        Output(PlotAreaComponent.DATA_PROJECTION, "selectedData"),
        Output(CheckpointComponent.PCA_MODEL, "data", allow_duplicate=True),
        Output(CheckpointComponent.LATENT_MODEL, "data", allow_duplicate=True),
        Output(CheckpointComponent.ANOMALY_MODEL, "data", allow_duplicate=True),
        Output(CheckpointComponent.FILTERED_DATA, "data", allow_duplicate=True),
        Input(CheckpointComponent.DATA, "data"),
        State(SessionComponent.SESSION_ID, "data"),
        prevent_initial_call=True,
    )
    def update_loading_status_and_workflow(
        data_checkpoint: bool,
        session_id: str,
    ) -> tuple[bool, bool, str, None, bool, bool, bool, bool]:
        if data_checkpoint:
            try:
                load_online_session(session_id)
            except Exception as exc:
                logger.error(exc)
                message = str(exc)
                features_checkpoint = False
            else:
                message = "Data loaded!"
                features_checkpoint = True

            return (features_checkpoint, False, message, None) + (False,) * 4

        raise PreventUpdate

    # Display input data table
    @app.callback(
        Output(OutputComponent.PREVIEW_DATA_TABLE, "data"),
        Output(OutputComponent.PREVIEW_DATA_TABLE, "columns"),
        Output(OutputComponent.PREVIEW_DATA_TABLE, "hidden_columns"),
        Output(OutputComponent.PREVIEW_DATA_TABLE, "style_data_conditional"),
        Input(CheckpointComponent.NUMERIC_FEATURES, "data"),
        Input(CheckpointComponent.FILTERED_DATA, "data"),
        Input(PlotAreaComponent.DATA_PROJECTION, "selectedData"),
        State(SessionComponent.SESSION_ID, "data"),
        prevent_initial_call=True,
    )
    def display_data_table(
        data_checkpoint: bool,
        filtered_data_checkpoint: bool,
        selected_data: dict[str, Any] | None,
        session_id: str,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[str], list[dict[str, Any]]]:

        if data_checkpoint:
            app_backend = load_online_session(session_id)

            try:
                data = get_datapoints_from_selection_or_hover(
                    data=app_backend.user_session.filtered_data,
                    selected_data=selected_data,
                )
            except KeyError:
                data = app_backend.user_session.data_wrapper.data

            # Add the conditional layout
            data_table_formatter = DataTableFormatter.format_table(data)
            styles = data_table_formatter.get_style(
                columns_to_style=app_backend.user_session.features
            )

            return (
                data_table_formatter.data_table,
                data_table_formatter.columns,
                (
                    data_table_formatter.data.columns
                    if disable_input_data_tables
                    else data_table_formatter.hidden_columns
                ),
                styles,
            )

        return (None, [], [], [])
