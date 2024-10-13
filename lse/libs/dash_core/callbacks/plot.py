import logging
from typing import Any

import dash
import numpy as np
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from lse.libs.backend.io.utils import read_feathered_dataframe
from lse.libs.dash_core.callbacks.common import _reconstruct_data
from lse.libs.dash_core.callbacks.utils import (
    get_datapoints_from_selection_or_hover,
    load_online_session,
)
from lse.libs.dash_core.components_models import (
    CheckpointComponent,
    DropdownComponent,
    InputComponent,
    OutputComponent,
    PlotAreaComponent,
    PlotControlComponent,
    SessionComponent,
)
from lse.libs.data_models import ClusterType, LatentModelType, VariableReference
from lse.libs.metaparameters import DATAPOINTS_LIMIT, N_RANDOM_SAMPLE
from lse.utils.cache import FlaskCacheManager

logger = logging.getLogger()

CACHE_MANAGER = FlaskCacheManager()


def add_callbacks(app: dash.Dash) -> None:  # noqa: C901, PLR0915

    # PC variance
    @app.callback(
        Output(PlotControlComponent.VARIANCE_FILTER, "min"),
        Output(PlotControlComponent.VARIANCE_FILTER, "marks"),
        Output(PlotAreaComponent.LOADINGS_PLOT, "figure"),
        Input(CheckpointComponent.PCA_MODEL, "data"),
        Input(DropdownComponent.PC_SELECTOR, "value"),
        Input(DropdownComponent.SIGNATURE, "value"),
        State(SessionComponent.SESSION_ID, "data"),
        prevent_initial_call=True,
    )
    def plot_pc_variance(
        model_checkpoint: bool,
        pc_selector: list[int],
        selected_features: list[str],
        session_id: str,
    ) -> tuple[float, dict[str, Any], go.Figure]:
        """Adapts the scale of the variance slider and plot PC loadings"""
        if model_checkpoint and len(pc_selector) > 0 and len(selected_features) > 0:
            app_backend = load_online_session(session_id)

            min_variance = app_backend.get_minimum_var_explained(selected_features)

            # PC loadings plot
            fig = app_backend.plot_loadings_barplot(pc_selector, selected_features)

            return (
                min_variance,
                {str(min_variance): f"{min_variance*100:2.0f}%", str(1): "100%"},
                fig,
            )

        raise PreventUpdate

    # Latent space plot
    @app.callback(
        Output(CheckpointComponent.FILTERED_DATA, "data"),
        Output(PlotAreaComponent.DATA_PROJECTION, "figure"),
        Output(InputComponent.N_KMEANS_CLUSTERS, "value"),
        Output(PlotControlComponent.PROBA_FILTER, "value"),
        Output(OutputComponent.BIC, "children"),
        Input(CheckpointComponent.ANOMALY_MODEL, "data"),
        Input(DropdownComponent.PC_SELECTOR, "value"),
        Input(PlotControlComponent.PROBA_FILTER, "value"),
        Input(PlotControlComponent.RANDOM_SAMPLE, "n_clicks"),
        Input(PlotControlComponent.RANDOM_SAMPLE_SIZE, "value"),
        Input(PlotControlComponent.RESET_SAMPLE, "n_clicks"),
        Input(PlotControlComponent.LOADINGS_CHECKER, "value"),
        Input(DropdownComponent.SIGNATURE, "value"),
        Input(PlotControlComponent.LATENT_SPACE_PLOT, "value"),
        Input(PlotControlComponent.CLUSTERIZE_RADIO_BUTTON, "value"),
        Input(InputComponent.N_KMEANS_CLUSTERS, "value"),
        Input(InputComponent.N_GAUSSIAN_KERNELS, "value"),
        Input(DropdownComponent.DATA_SLICER, "value"),
        Input(DropdownComponent.SIZE_CODE, "value"),
        Input(DropdownComponent.COLOR_CODE, "value"),
        Input(DropdownComponent.COLORSCALE, "value"),
        Input(DropdownComponent.MARKER_CODE, "value"),
        Input(OutputComponent.EXPORT_DATA_TABLE, "selected_rows"),
        State(PlotAreaComponent.DATA_PROJECTION, "figure"),
        State(SessionComponent.SESSION_ID, "data"),
        prevent_initial_call=True,
    )
    def update_latent_space_projection_figure(  # noqa: C901, PLR0912
        anomaly_model_check: bool,
        pc_selector: list[int],
        proba_threshold: list[float],
        random_sample: int,
        random_sample_size: int,
        reset_sample: int,
        plot_loadings: bool,
        signature_variables: list[str],
        latent_space_plot: str,
        clusterize_radio_button: str | None,
        n_clusters: int | None,
        n_kernels: int,
        data_slicer: str | None,
        size_dimension: str | None,
        color_dimension: str | None,
        colorscale: str | None,
        marker_symbol_dimension: str | None,
        selected_points: list[int],
        fig: go.Figure,
        session_id: str,
    ) -> tuple[bool, go.Figure, int | None, list[float], str]:
        ctx = dash.callback_context

        if anomaly_model_check and n_kernels is not None:
            app_backend = load_online_session(session_id)
            model_type = LatentModelType(latent_space_plot)

            # Filter with gaussian distribution
            plot_data, _ = app_backend.apply_gaussian_filter(1 - np.array(proba_threshold))

            # Make sure we don't show too many points
            if len(plot_data) > DATAPOINTS_LIMIT:
                logger.info(
                    f"Removing {len(plot_data) - DATAPOINTS_LIMIT} points, limiting"
                    f" plotting to {DATAPOINTS_LIMIT} data points"
                )
                plot_data = plot_data.sort_values(
                    VariableReference.LOG_PROBABILITY, ascending=True
                )[:DATAPOINTS_LIMIT]
                # Adjust the proba slider to the constrains of the dataset
                proba_threshold[1] = DATAPOINTS_LIMIT / app_backend.user_session.n_samples
            else:
                proba_threshold = dash.no_update

            # Optimal k-means
            if (input_id := ctx.triggered[0]["prop_id"].split(".")[0]) in (
                PlotControlComponent.PROBA_FILTER,
                CheckpointComponent.ANOMALY_MODEL,
            ) or n_clusters is None:
                n_clusters = app_backend.get_optimal_n_kmeans_clusters(
                    app_backend.get_pipeline_output(model_ref=model_type, pc_list=pc_selector).loc[
                        plot_data.index
                    ]
                )

            if input_id == PlotControlComponent.RANDOM_SAMPLE:
                plot_data = plot_data.sample(
                    min([random_sample_size or N_RANDOM_SAMPLE, len(plot_data)])
                )

            if (
                input_id in (PlotControlComponent.LOADINGS_CHECKER, DropdownComponent.SIGNATURE)
                and model_type is LatentModelType.PCA
            ):
                # Update plot
                fig = app_backend.remove_loadings(go.Figure(fig))
                if plot_loadings:
                    fig = app_backend.plot_loadings_direction(
                        fig=fig,
                        pc_indices=pc_selector,
                        vars_to_plot=signature_variables,
                    )

            else:
                # Plot
                try:
                    fig = app_backend.plot_in_latent_space(
                        plot_data=plot_data,
                        latent_model=model_type,
                        pcs=pc_selector,
                        n_kmeans_clusters=(
                            n_clusters
                            if clusterize_radio_button
                            and ClusterType(clusterize_radio_button) is ClusterType.PLOT
                            else None
                        ),
                        data_slicer=data_slicer,
                        size_dimension=size_dimension,
                        color_dimension=(
                            VariableReference.DOMINANT_COMPONENT
                            if clusterize_radio_button
                            and ClusterType(clusterize_radio_button)
                            is ClusterType.GAUSSIAN_COMPONENT
                            else color_dimension
                        ),
                        colorscale=colorscale,
                        marker_symbol_dimension=marker_symbol_dimension,
                        loadings=signature_variables if plot_loadings else None,
                    )

                except Exception:
                    logger.exception("Error updating latent space projection figure")
                    fig = go.Figure(data=[], layout={})

            # Update figure
            fig.update_layout(clickmode="event+select", uirevision=latent_space_plot)

            # Update filter
            app_backend.user_session.data_infilter = plot_data.index

            return True, fig, n_clusters, proba_threshold, f"{app_backend.get_bic():n}"

        raise PreventUpdate

    # Data point signature
    @app.callback(
        Output(PlotAreaComponent.DATA_POINT_SIGNATURE, "figure"),
        Input(CheckpointComponent.PCA_MODEL, "data"),
        Input(PlotAreaComponent.DATA_PROJECTION, "hoverData"),
        Input(DropdownComponent.SIGNATURE, "value"),
        Input(PlotControlComponent.VARIANCE_FILTER, "value"),
        State(SessionComponent.SESSION_ID, "data"),
        prevent_initial_call=True,
    )
    def display_signature(
        pca_data_checkpoint: bool,
        hover_data: dict[str, Any] | None,
        signature_variables: list[str],
        var_explained: float,
        session_id: str,
    ) -> go.Figure:
        if pca_data_checkpoint and hover_data is not None:
            if len(hover_data["points"]) > 0:
                reconstruct_data = CACHE_MANAGER.memoize_function(_reconstruct_data)

                app_backend = load_online_session(session_id)

                # Reconstruct data
                reconstructed_data = read_feathered_dataframe(
                    reconstruct_data(
                        session_id=session_id,
                        signature_variables=signature_variables,
                        var_explained=var_explained,
                    ),
                )

                try:
                    # Draw the radar plot
                    return app_backend.plot_radar(
                        datapoint_series=get_datapoints_from_selection_or_hover(
                            selected_data=hover_data, data=reconstructed_data
                        ).iloc[0],
                        radial_range=(
                            reconstructed_data.min().min(),
                            reconstructed_data.max().max(),
                        ),
                    )
                except Exception:
                    logger.error(f"No signature to plot for hover data: {hover_data['points']}")
                    return go.Figure(data=[], layout={})

        raise PreventUpdate

    # Selected data points
    @app.callback(
        Output(PlotAreaComponent.ORIGINAL_COORDINATES_PLOT, "figure"),
        Input(CheckpointComponent.FILTERED_DATA, "data"),
        Input(PlotAreaComponent.DATA_PROJECTION, "selectedData"),
        Input(PlotControlComponent.PLOT_TYPE, "value"),
        Input(DropdownComponent.SIGNATURE, "value"),
        Input(PlotControlComponent.VARIANCE_FILTER, "value"),
        Input(DropdownComponent.DATA_SLICER, "value"),
        Input(DropdownComponent.SIZE_CODE, "value"),
        Input(DropdownComponent.COLOR_CODE, "value"),
        Input(DropdownComponent.COLORSCALE, "value"),
        Input(DropdownComponent.MARKER_CODE, "value"),
        State(SessionComponent.SESSION_ID, "data"),
        prevent_initial_call=True,
    )
    def plot_data_selection(
        data_check: bool,
        selected_data: dict[str, Any] | None,
        graph_type: str,
        signature_variables: list[str],
        var_explained: float,
        data_slicer: str | None,
        size_dimension: str | None,
        color_dimension: str | None,
        colorscale: str | None,
        marker_symbol_dimension: str | None,
        session_id: str,
    ) -> go.Figure:
        if data_check:
            reconstruct_data = CACHE_MANAGER.memoize_function(_reconstruct_data)

            app_backend = load_online_session(session_id)

            reconstructed_data = get_datapoints_from_selection_or_hover(
                read_feathered_dataframe(
                    reconstruct_data(
                        session_id=session_id,
                        signature_variables=signature_variables,
                        var_explained=var_explained,
                    )
                ).loc[app_backend.user_session.data_infilter],
                selected_data=selected_data,
            )

            return app_backend.plot_in_original_space(
                graph_type=graph_type,
                data=reconstructed_data,
                data_slicer=data_slicer,
                size_dimension=size_dimension,
                color_dimension=color_dimension,
                colorscale=colorscale,
                marker_symbol_dimension=marker_symbol_dimension,
            )

        raise PreventUpdate
