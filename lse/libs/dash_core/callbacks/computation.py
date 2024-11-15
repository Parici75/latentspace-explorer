import logging
from typing import Any

import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from lse.libs.dash_core.callbacks.common import _reconstruct_data
from lse.libs.dash_core.callbacks.utils import load_online_session, update_data_cache
from lse.libs.dash_core.components_models import (
    CheckpointComponent,
    ComputeComponent,
    DropdownComponent,
    ExportComponent,
    InputComponent,
    OutputComponent,
    PlotAreaComponent,
    PlotControlComponent,
    SessionComponent,
    StatusComponent,
)
from lse.libs.metaparameters import N_KERNELS_SPAN
from lse.utils.cache import FlaskCacheManager

logger = logging.getLogger()


def _get_gmm_hyperparameters(
    session_id: str,
    var_explained: float,
    n_gaussian_components: int,
) -> tuple[int, str]:
    app_backend = load_online_session(session_id)

    # Find optimal number of kernels
    n_kernels, covariance_type = app_backend.get_gmm_hyperparameters(
        var_explained=var_explained,
        n_components=(
            range(
                max(1, n_gaussian_components - N_KERNELS_SPAN),
                n_gaussian_components + N_KERNELS_SPAN,
            )
            if n_gaussian_components > 1
            else 1
        ),
    )

    return n_kernels, covariance_type


CACHE_MANAGER = FlaskCacheManager()


def add_callbacks(app: dash.Dash) -> None:  # noqa: C901

    @app.callback(
        Output(CheckpointComponent.PCA_MODEL, "data"),
        Output(InputComponent.N_GAUSSIAN_KERNELS, "value"),
        Output(PlotControlComponent.PROBA_FILTER, "value", allow_duplicate=True),
        Input(ComputeComponent.RUN_MODEL_PIPELINE, "n_clicks"),
        State(ComputeComponent.INITIAL_GAUSSIAN_MIXTURE_GUESS, "value"),
        State(PlotControlComponent.VARIANCE_FILTER, "value"),
        State(ComputeComponent.STANDARDIZE, "value"),
        State(DropdownComponent.FEATURES, "value"),
        State(PlotAreaComponent.DATA_PROJECTION, "selectedData"),
        State(SessionComponent.SESSION_ID, "data"),
        prevent_initial_call=True,
    )
    def initialize_pipeline(
        n_clicks_pipeline: int,
        n_gaussian_components: int,
        var_explained: float,
        standardize: bool,
        features: list[str],
        selected_data: dict[str, Any] | None,
        session_id: str,
    ) -> tuple[bool, int, list[float]]:

        if n_clicks_pipeline > 0:

            # Flush user cach
            CACHE_MANAGER.flush_cache(_get_gmm_hyperparameters, _reconstruct_data)
            get_gmm_hyperparameters = CACHE_MANAGER.memoize_function(_get_gmm_hyperparameters)

            # Update the cached data
            update_data_cache(session_id=session_id, selected_data=selected_data)

            # Initialize the pipeline
            app_backend = load_online_session(session_id)
            logger.debug(
                f"Initializing Latent space models of session {session_id} with {features} features"
            )
            app_backend.initialize_latent_models_pipeline(
                features=features,
                standardize=bool(standardize),
            )

            # Find optimal number of kernels
            logger.debug(
                f"Finding optimal number of kernels explaining {var_explained*100}% of the variance in the data..."
            )
            n_kernels, _ = get_gmm_hyperparameters(
                session_id=session_id,
                var_explained=var_explained,
                n_gaussian_components=n_gaussian_components,
            )

            app_backend.user_session.data_infilter = None

            # Signal computation is complete
            return True, n_kernels, [0, 1]

        raise PreventUpdate

    @app.callback(
        Output(StatusComponent.LOADING_STATUS, "children"),
        Input(CheckpointComponent.LATENT_MODEL, "data"),
        prevent_initial_call=True,
    )
    def update_computation_status(model_checkpoint: bool) -> str:
        if model_checkpoint:
            return "Data ready to be explored!"

        raise PreventUpdate

    @app.callback(
        Output(CheckpointComponent.LATENT_MODEL, "data"),
        Output(ExportComponent.PREPARE_EXPORT_LATENT_SPACE_MODEL, "disabled"),
        Input(CheckpointComponent.PCA_MODEL, "data"),
        Input(DropdownComponent.PC_SELECTOR, "value"),
        Input(PlotControlComponent.PERPLEXITY, "value"),
        State(SessionComponent.SESSION_ID, "data"),
        prevent_initial_call=True,
    )
    def compute_latent_models(
        data_check: bool,
        pc_selector: list[int],
        perplexity: int,
        session_id: str,
    ) -> tuple[bool, bool]:

        if data_check:
            app_backend = load_online_session(session_id)
            logger.debug(f"Fitting Latent space models with PCs: {pc_selector}")

            app_backend.fit_latent_models(
                pc_list=pc_selector,
                perplexity=perplexity,
            )

            # Signal computation is complete
            return True, False
        raise PreventUpdate

    # Gaussian modeling
    @app.callback(
        Output(CheckpointComponent.ANOMALY_MODEL, "data"),
        Input(CheckpointComponent.LATENT_MODEL, "data"),
        Input(InputComponent.N_GAUSSIAN_KERNELS, "value"),
        Input(PlotControlComponent.VARIANCE_FILTER, "value"),
        State(SessionComponent.SESSION_ID, "data"),
        prevent_initial_call=True,
    )
    def compute_gaussian_model(
        model_checkpoint: bool, n_kernels: int, var_explained: float, session_id: str
    ) -> bool:
        if model_checkpoint and n_kernels is not None:

            # Find optimal covariance scheme
            get_gmm_hyperparameters = CACHE_MANAGER.memoize_function(_get_gmm_hyperparameters)
            if n_kernels > 1:
                logger.debug(f"Finding optimal covariance matrix for {n_kernels} kernels...")
                _, covariance_type = get_gmm_hyperparameters(
                    session_id=session_id,
                    var_explained=var_explained,
                    n_gaussian_components=n_kernels,
                )

            else:
                covariance_type = "diag"

            # Fit
            app_backend = load_online_session(session_id)
            logger.debug(
                f"Fitting Gaussian model with {n_kernels} kernels and {covariance_type} covariance"
                " matrix"
            )
            app_backend.fit_gaussian_model(
                n_kernels=n_kernels,
                covariance_type=covariance_type,
                min_var_retained=var_explained,
            )

            return True
        raise PreventUpdate

    # Gini impurity
    @app.callback(
        Output(OutputComponent.GINI, "children"),
        Output(OutputComponent.SLICER, "children"),
        Output(OutputComponent.GINI.container, "className"),
        Input(DropdownComponent.DATA_SLICER, "value"),
        Input(CheckpointComponent.ANOMALY_MODEL, "data"),
        State(SessionComponent.SESSION_ID, "data"),
        prevent_initial_call=True,
    )
    def compute_gini_impurity(
        data_slice: str,
        model_checkpoint: bool,
        session_id: str,
    ) -> tuple[str, str, str]:
        if data_slice is not None and model_checkpoint:
            app_backend = load_online_session(session_id)
            if data_slice in app_backend.user_session.data_wrapper.identifiers:
                return (
                    f"{app_backend.get_weighted_average_gini(data_slice):.2f}",
                    f"({data_slice})",
                    OutputComponent.GINI.container,
                )
        return "", "", f"{'-'.join((OutputComponent.GINI.container, 'hidden'))}"
