import logging
from typing import Any

import dash
import numpy as np
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from statsplotly.constants import BUILTIN_COLORSCALES

from lse.libs.dash_core.callbacks.utils import load_online_session
from lse.libs.dash_core.components_models import (
    CheckpointComponent,
    DropdownComponent,
    PlotControlComponent,
    SessionComponent,
)
from lse.libs.data_models import ClusterType, LatentModelType, VariableReference
from lse.libs.metaparameters import N_DISPLAYED_FEATURES

logger = logging.getLogger()


def add_callbacks(app: dash.Dash) -> None:  # noqa: C901 PLR0915
    # Features selection
    @app.callback(
        Output(DropdownComponent.FEATURES, "options"),
        Output(DropdownComponent.FEATURES, "value"),
        Input(CheckpointComponent.NUMERIC_FEATURES, "data"),
        State(SessionComponent.SESSION_ID, "data"),
        prevent_initial_call=True,
    )
    def display_available_features(
        data_check: bool, session_id: str
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if data_check:
            app_backend = load_online_session(session_id)
            return (
                [{"label": var, "value": var} for var in app_backend.user_session.features],
                app_backend.user_session.features,
            )

        return [], []

    @app.callback(
        Output(DropdownComponent.PC_SELECTOR, "options"),
        Output(DropdownComponent.PC_SELECTOR, "value"),
        Input(CheckpointComponent.PCA_MODEL, "data"),
        Input(PlotControlComponent.GET_FULL_COMPONENTS, "n_clicks"),
        Input(PlotControlComponent.RESET_COMPONENTS, "n_clicks"),
        State(DropdownComponent.PC_SELECTOR, "value"),
        State(SessionComponent.SESSION_ID, "data"),
        prevent_initial_call=True,
    )
    def update_principal_components_selector(
        data_check: bool,
        full_components: int,
        components_reset: int,
        pc_selector: list[int],
        session_id: str,
    ) -> tuple[list[dict[str, Any]], list[int]]:
        # Components preselection
        ctx = dash.callback_context

        if data_check:
            app_backend = load_online_session(session_id)

            if (
                input_id := ctx.triggered[0]["prop_id"].split(".")[0]
            ) == CheckpointComponent.PCA_MODEL:
                return [
                    {"label": i + 1, "value": i}
                    for i in range(
                        app_backend.user_session.models[LatentModelType.PCA].model.n_components_
                    )
                ], [0, 1]

            if input_id == PlotControlComponent.GET_FULL_COMPONENTS:
                return dash.no_update, list(
                    range(app_backend.user_session.models[LatentModelType.PCA].model.n_components_)
                )

            if input_id == PlotControlComponent.RESET_COMPONENTS:
                if pc_selector == [0, 1]:
                    logger.debug("Principal components selector is already reset")
                    raise PreventUpdate
                return dash.no_update, [0, 1]

        raise PreventUpdate

    @app.callback(
        Output(DropdownComponent.SIGNATURE, "options"),
        Output(DropdownComponent.SIGNATURE, "value"),
        Input(DropdownComponent.PC_SELECTOR, "value"),
        Input(PlotControlComponent.GET_FULL_SIGNATURE, "n_clicks"),
        Input(PlotControlComponent.RESET_SIGNATURE, "n_clicks"),
        State(SessionComponent.SESSION_ID, "data"),
        prevent_initial_call=True,
    )
    def update_signature_selector(
        pc_selector: list[int],
        full_signature: int,
        signature_reset: int,
        session_id: str,
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        # Signature preselection
        ctx = dash.callback_context

        if len(pc_selector) > 0:
            app_backend = load_online_session(session_id)

            sorted_loadings = None
            try:
                sorted_loadings = list(
                    app_backend.user_session.models[LatentModelType.PCA]
                    .get_sorted_loadings(pc=np.sort(pc_selector)[0] + 1)
                    .index
                )
            except KeyError:
                logger.debug(f"No model available yet for the session {session_id}")
                return [], []

            if sorted_loadings is not None:
                signature_options = [{"label": i, "value": i} for i in sorted_loadings]
                if (
                    ctx.triggered[0]["prop_id"].split(".")[0]
                    == PlotControlComponent.GET_FULL_SIGNATURE
                ):
                    signature_preselection = [option["value"] for option in signature_options]
                else:
                    signature_preselection = [
                        option["value"] for option in signature_options[:N_DISPLAYED_FEATURES]
                    ]

                return signature_options, signature_preselection

        raise PreventUpdate

    ## Selector callback
    # Variables selection
    @app.callback(
        Output(DropdownComponent.DATA_SLICER, "options"),
        Output(DropdownComponent.DATA_SLICER, "value"),
        Output(DropdownComponent.SIZE_CODE, "options"),
        Output(DropdownComponent.COLOR_CODE, "options"),
        Output(DropdownComponent.COLOR_CODE, "value"),
        Output(DropdownComponent.COLOR_CODE, "disabled"),
        Output(DropdownComponent.COLORSCALE, "options"),
        Output(DropdownComponent.COLORSCALE, "value"),
        Output(DropdownComponent.MARKER_CODE, "options"),
        Input(CheckpointComponent.PCA_MODEL, "data"),
        Input(DropdownComponent.DATA_SLICER, "value"),
        Input(DropdownComponent.COLOR_CODE, "value"),
        Input(PlotControlComponent.CLUSTERIZE_RADIO_BUTTON, "value"),
        State(DropdownComponent.COLORSCALE, "value"),
        State(SessionComponent.SESSION_ID, "data"),
        prevent_initial_call=True,
    )
    def update_graph_options(
        data_check: bool,
        data_slicer: str | None,
        color_code: str | None,
        clusterize_radio_button: int,
        current_colorscale: str,
        session_id: str,
    ) -> tuple[
        list[dict[str, Any]],
        str | None,
        list[dict[str, Any]],
        list[dict[str, Any]],
        str | None,
        bool,
        list[dict[str, Any]],
        str,
        list[dict[str, Any]],
    ]:
        """Set the dropdown selectors"""
        if data_check:
            app_backend = load_online_session(session_id)

            # Slice options
            slice_options = [
                {"label": i, "value": i}
                for i in app_backend.user_session.data_wrapper.identifiers
                if app_backend.user_session.data_wrapper.data.reset_index()[i].nunique()
                < len(app_backend.user_session.data_wrapper.data)
            ]
            slice_options.append(
                {
                    "label": ClusterType.GAUSSIAN_COMPONENT,
                    "value": VariableReference.DOMINANT_COMPONENT,
                }
            )
            # Data can not be sliced when clusterized
            if clusterize_radio_button != 0:
                color_code_disabled = True
                color_code = None
            else:
                color_code_disabled = False

            # Size
            size_options = [
                {"label": i, "value": i}
                for i in app_backend.user_session.data_wrapper.numerical_identifiers
            ]
            # Color
            color_options = [
                {"label": i, "value": i} for i in app_backend.user_session.data_wrapper.identifiers
            ]
            # Colorscale
            colorscale_options = [{"label": i, "value": i} for i in sorted(BUILTIN_COLORSCALES)]
            if (current_colorscale is not None) and (
                current_colorscale not in [option["value"] for option in colorscale_options]
            ):
                current_colorscale = colorscale_options[0]["value"]
            # Marker
            marker_options = [
                {"label": i, "value": i}
                for i in app_backend.user_session.data_wrapper.identifiers
                if i not in app_backend.user_session.data_wrapper.numerical_identifiers
            ]

            return (
                slice_options,
                data_slicer,
                size_options,
                color_options,
                color_code,
                color_code_disabled,
                colorscale_options,
                current_colorscale,
                marker_options,
            )
        raise PreventUpdate

    # Plotting options
    @app.callback(
        Output(PlotControlComponent.PERPLEXITY, "disabled"),
        Output(PlotControlComponent.LOADINGS_CHECKER, "options"),
        Input(PlotControlComponent.LATENT_SPACE_PLOT, "value"),
        State(PlotControlComponent.LOADINGS_CHECKER, "options"),
        prevent_initial_call=False,
    )
    def enable_plotting_options(
        latent_space_plot_selector: str, loadings_checker_options: list[Any]
    ) -> dash.Dash:
        loadings_checker_options[0]["disabled"] = False
        perplexity_slider_disabled = False

        if (model_type := LatentModelType(latent_space_plot_selector)) is LatentModelType.TSNE:
            loadings_checker_options[0]["disabled"] = True
        elif model_type is LatentModelType.PCA:
            loadings_checker_options[0]["disabled"] = False
            perplexity_slider_disabled = True

        return perplexity_slider_disabled, loadings_checker_options
