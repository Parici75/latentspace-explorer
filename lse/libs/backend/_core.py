from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Self, TypeVar, Unpack

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import statsplotly as sts
from joblib import Parallel, delayed
from mltoolbox.anomaly_detection import (
    GaussianAnomalyQuantifier,
    GMMHyperparameterTuner,
)
from mltoolbox.latent_space import PCALatentSpace, TSNELatentSpace
from numpy.typing import NDArray

from lse.libs.backend._kmeans import KMeansClassifier
from lse.libs.backend._plotting import (
    GenericPlotter,
    PcSpacePlotAnnotater,
    PcSpacePlotter,
    concatenate_text_dimensions,
    get_plot_dimensions,
)
from lse.libs.backend._utils import add_customdata
from lse.libs.backend.data_models import LatentModelArg, latent_models
from lse.libs.backend.io.data_models import UserSession, UserSessionParameters
from lse.libs.data_models import (
    ClusterType,
    LatentModelType,
    PlotType,
    VariableReference,
)
from lse.libs.exceptions import (
    DataProcessingError,
    DataReconstructionError,
    LatentSpaceModelFittingError,
    LatentSpacePlottingError,
)
from lse.libs.metaparameters import (
    DATAPOINTS_LIMIT,
    DEFAULT_PERPLEXITY,
    GMM_INIT_PARAMS,
    MAX_N_COMPONENTS,
    N_GAUSSIAN_MIXTURE_INIT,
    TSNE_DIMENSION,
)
from lse.libs.utils.plot_style import PlotCues, get_marker_symbol_map

PARALLELIZE = False

logger = logging.getLogger()

F = TypeVar("F", bound=Callable[..., Any])


class AppBackend(KMeansClassifier, GenericPlotter):
    """Main class for backend computations and plotting."""

    def __init__(self, user_session: UserSession) -> None:
        super().__init__()
        self.user_session = user_session

        if len(self.user_session.pca_data) == 0:
            raise DataProcessingError("No numeric values to process, check your data")

    @property
    def user_session_parameters(self) -> UserSessionParameters:
        return UserSessionParameters(
            latent_models=self.user_session.models,
            data_infilter=self.user_session.data_infilter,
            features=self.user_session.features,
            anomaly_quantifier=self.user_session.anomaly_quantifier,
        )

    def initialize_latent_models_pipeline(self, features: list[str], standardize: bool) -> Self:
        """Initializes pipeline."""
        self.user_session.features = features
        models = self.user_session.models
        _data = self.user_session.pca_data

        # Update the models with PCA
        models.update(
            {
                LatentModelType.PCA: PCALatentSpace.initialize(
                    standardize=standardize,
                    n_components=min(
                        MAX_N_COMPONENTS,
                        *_data.shape,
                    ),
                ).fit(_data)
            }
        )

        # Save models
        self.user_session.models = models

        return self

    def _transform_data_in_pc_space(
        self, data: pd.DataFrame, pc_list: list[int] | None = None
    ) -> pd.DataFrame:
        """Returns the data projected in Principal Components space."""

        transformed_data = self.user_session.models[LatentModelType.PCA].project_data(data)
        if pc_list is not None:
            transformed_data = transformed_data.iloc[:, pc_list]
        return transformed_data

    def fit_latent_models(
        self, pc_list: list[int] | None = None, **kwargs: Unpack[LatentModelArg]
    ) -> Self:
        """Fit latent models."""

        def _fit_model(
            model_ref: LatentModelType,
            model: TSNELatentSpace,
        ) -> tuple[LatentModelType, TSNELatentSpace]:
            logger.debug(f"Fitting {model_ref.value} to {len(_data)} data points")

            model_params: dict[str, Any] = {}
            if model_ref is LatentModelType.TSNE:
                model_params.update(
                    {
                        "perplexity": kwargs.get(
                            "perplexity", min(DEFAULT_PERPLEXITY, len(_data) - 1)
                        ),
                        "n_components": min(_data.shape[1], TSNE_DIMENSION),
                    }
                )

            try:
                return model_ref, model.initialize(**model_params).fit(_data)
            except Exception as exc:
                raise LatentSpaceModelFittingError(model_ref) from exc

        _data = self._transform_data_in_pc_space(self.user_session.pca_data, pc_list=pc_list)
        models = self.user_session.models

        # Fit latent models
        if PARALLELIZE:
            models.update(
                dict(
                    Parallel(n_jobs=-1)(
                        delayed(_fit_model)(model_ref, model)
                        for (model_ref, model) in latent_models.items()
                    )
                )
            )
        else:
            models.update(
                _fit_model(model_ref, model) for (model_ref, model) in latent_models.items()
            )

        # Save models
        self.user_session.models = models

        return self

    def fit_gaussian_model(
        self, n_kernels: int, covariance_type: str, min_var_retained: float
    ) -> Self:
        """Fit the gaussian model."""
        self.user_session.anomaly_quantifier = GaussianAnomalyQuantifier.initialize(
            pca_model=self.user_session.models[LatentModelType.PCA],
            min_var_retained=min_var_retained,
            n_components=n_kernels,
            covariance_type=covariance_type,
            n_init=N_GAUSSIAN_MIXTURE_INIT,
        ).fit(self.user_session.pca_data)

        return self

    def get_gmm_hyperparameters(
        self,
        var_explained: float,
        n_components: range | int | None = None,
    ) -> tuple[int, str]:
        """Finds best Gaussian model hyperparameters.

        Returns:
            A tuple of the optimal pair of (n_components, covariance type)
        """
        _data = self.user_session.pca_data

        hyperparameter_tuner = GMMHyperparameterTuner(
            pca_model=self.user_session.models[LatentModelType.PCA],
            min_var_retained=var_explained,
            max_n_components=len(_data),
        )
        sample_data = (
            _data
            if len(_data) <= DATAPOINTS_LIMIT
            else _data.sample(n=DATAPOINTS_LIMIT, replace=False)
        )
        best_n_components, covariance_type = hyperparameter_tuner.find_best_param(
            sample_data,
            n_components,
            init_params=GMM_INIT_PARAMS,
        )

        return best_n_components, covariance_type

    def get_pipeline_output(
        self, model_ref: LatentModelType, pc_list: list[int] | None = None
    ) -> pd.DataFrame:
        """Returns projections of the data by the model specified by `model_ref`."""

        if model_ref is LatentModelType.PCA:
            return self.user_session.models[model_ref].project_data(self.user_session.pca_data)

        return self.user_session.models[model_ref].project_data(
            self._transform_data_in_pc_space(self.user_session.pca_data, pc_list)
        )

    def compute_datapoints_probas(self) -> pd.Series:
        """Computes the probability of the data in `self.user_session` under the generative model.

        Returns:
            A {obj}`pandas.Series` of log probabilities.
        """
        return self.user_session.anomaly_quantifier.score_samples(  # type: ignore
            self.user_session.pca_data
        ).rename(VariableReference.LOG_PROBABILITY.value)

    def apply_gaussian_filter(
        self, proba_threshold: NDArray[Any]
    ) -> tuple[pd.DataFrame, NDArray[Any]]:
        """Filters the dataset with the gaussian model.

        Args:
            proba_threshold:
                A 2-values quantile array.

        Returns:
            A tuple of (filtered {obj}`pandas.DataFrame` of datapoints with log_probabilities
            comprised in quantile (log_probabilities, proba_threshold),
            quantile).
        """

        # Get the proba of each point
        log_probas = self.compute_datapoints_probas()
        # Filter
        quantile = np.quantile(log_probas, proba_threshold)
        logical_mask = (log_probas <= quantile[0]) & (log_probas >= quantile[1])

        # Return the joint with plotting data
        return (
            log_probas.to_frame().loc[logical_mask].join(self.user_session.data_wrapper.data),
            quantile,
        )

    def get_bic(self) -> float:
        """Returns the Bayesian Information Criterion for the model fitted."""

        return self.user_session.anomaly_quantifier.compute_bic(self.user_session.pca_data)  # type: ignore

    def predict_dominant_component(self) -> pd.Series:
        """Returns the most likely gaussian the data belongs to.

        Returns:
            A {obj}`pandas.Series` of prediction.
        """

        predicted_components = self.user_session.anomaly_quantifier.predict_component(  # type: ignore
            self.user_session.pca_data
        ).rename(
            VariableReference.DOMINANT_COMPONENT
        )

        return predicted_components

    def get_mixture_densities(self) -> pd.Series[NDArray]:
        """Returns the density of each gaussian components for the data.

        Returns:
            A {obj}`pandas.Series` of arrays.
        """

        mixture_components = self.user_session.anomaly_quantifier.predict_components_proba(  # type: ignore
            self.user_session.pca_data
        ).rename(
            VariableReference.DOMINANT_COMPONENT
        )

        return mixture_components

    def reconstruct_data(
        self, signature_variables: list[str], var_explained: float
    ) -> pd.DataFrame:
        """Reconstructs data from principal components explaining at least a fraction=`var_explained` of the
        variance."""
        logger.debug(
            f"Reconstructing data preserving {var_explained*100}% of the original variance..."
        )

        try:
            return (
                self.user_session.models[LatentModelType.PCA]
                .reconstruct_variables(self.user_session.pca_data, var_explained)
                .loc[:, signature_variables]
            )

        except Exception as exc:
            logger.exception("Error reconstructing data")
            raise DataReconstructionError from exc

    @staticmethod
    def gini(x: NDArray[Any]) -> float:
        """Returns the gini impurity index of the array."""
        gini = 0
        n = len(x)
        for label in np.unique(x):
            gini += (np.sum(x == label) / n) ** 2

        return 1 - gini

    def get_weighted_average_gini(self, data_slicer: str) -> float:
        """Returns the weighted average gini impurity index of the gaussian model on the partition defined by `data slicer`."""

        labels_data = (
            self.user_session.data_wrapper.data.reset_index()
            .set_index(VariableReference.UNIQUE_ID)[[data_slicer]]
            .join(
                self.predict_dominant_component()
                .astype(str)
                .reset_index()
                .set_index(VariableReference.UNIQUE_ID)[VariableReference.DOMINANT_COMPONENT],
            )
        )
        slice_mapper = dict(
            zip(
                labels_data[data_slicer].unique(),
                np.arange(len(labels_data[data_slicer].unique())) + 1,
                strict=False,
            )
        )
        label_counts = labels_data[VariableReference.DOMINANT_COMPONENT].value_counts(
            normalize=True
        )

        ginis = []
        for label in label_counts.index:
            ginis.append(
                self.gini(
                    labels_data.loc[
                        labels_data[VariableReference.DOMINANT_COMPONENT] == label, data_slicer
                    ]
                    .map(slice_mapper)
                    .to_numpy()
                )
            )

        return np.average(ginis, weights=label_counts)

    def get_minimum_var_explained(self, selected_features: list[str]) -> float:
        """Returns the minimum variance explained for a subset of variables."""
        min_variance_list = []
        for variable in selected_features:
            var_idx = np.where(self.user_session.models[LatentModelType.PCA].orig_vars == variable)[
                0
            ][0]
            min_variance_list.append(
                np.sort(
                    self.user_session.models[LatentModelType.PCA].pc_var_correlations[var_idx, :]
                )[::-1][0]
                ** 2
            )

        return min(min_variance_list)

    ## Plotting
    def _extract_plot_cues(
        self,
        plot_data: pd.DataFrame,
        data_dimensions: list[str],
        data_slicer: str | None,
        color_dimension: str | None,
        marker_symbol_dimension: str | None,
        size_dimension: str | None,
        n_kmeans_clusters: int | None = None,
    ) -> tuple[pd.DataFrame, PlotCues]:
        """Returns a {obj}`pandas.DataFrame` of the plotting data, along with a {obj}`PlotCues` object
        specifying plot styling dimensions."""

        # Negative log-likelihood is more convenient for plotting
        plot_data[VariableReference.NEGATIVE_LOG_PROBABILITY] = -plot_data[
            VariableReference.LOG_PROBABILITY
        ]

        # Color
        color: ClusterType | VariableReference | str | None
        if n_kmeans_clusters is not None:
            # Add the cluster_idx info
            plot_data[ClusterType.PLOT] = self.get_kmeans_clusters(
                plot_data.loc[:, data_dimensions],
                n_kmeans_clusters,
            )
            color = ClusterType.PLOT
        elif color_dimension is VariableReference.DOMINANT_COMPONENT:
            color = color_dimension
            plot_data["opacity"] = self.get_mixture_densities().map(np.max)

        elif color_dimension is not None:
            color = color_dimension
        else:
            color = VariableReference.NEGATIVE_LOG_PROBABILITY

        # Slicer
        plot_data = plot_data.reset_index()
        slicer: ClusterType | str
        if data_slicer is None:
            # Add a dummy "data" dimension for user interaction
            slicer = "data"
            plot_data[slicer] = slicer
        else:
            slicer = data_slicer

        # Marker
        if marker_symbol_dimension is not None:
            plot_data = plot_data.assign(
                marker_symbol=get_marker_symbol_map(plot_data[marker_symbol_dimension])
            )

        # Text
        text_dimension = concatenate_text_dimensions(plot_data=plot_data, plot_cues=(slicer, color))

        return plot_data, PlotCues(
            slicer=slicer,
            color=color,
            marker="marker_symbol" if marker_symbol_dimension is not None else None,
            opacity=(
                "opacity"
                if "opacity" in plot_data.columns and len(data_dimensions) <= 2  # noqa: PLR2004
                else None
            ),
            size=size_dimension,
            text=text_dimension,
        )

    def plot_loadings_barplot(
        self, pc_indices: list[int], vars_to_plot: list[str] | None = None
    ) -> go.Figure:
        return PcSpacePlotter(
            pca_model=self.user_session.models[LatentModelType.PCA], pcs=pc_indices
        ).loadings_barplot(variables=vars_to_plot)

    def remove_loadings(self, fig: go.Figure) -> go.Figure:
        return PcSpacePlotAnnotater(
            pca_model=self.user_session.models[LatentModelType.PCA], pca_fig=fig
        ).erase_loadings()

    def plot_loadings_direction(
        self, fig: go.Figure, pc_indices: list[int], vars_to_plot: list[str]
    ) -> go.Figure:
        return PcSpacePlotAnnotater(
            pca_model=self.user_session.models[LatentModelType.PCA], pcs=pc_indices, pca_fig=fig
        ).plot_loadings_direction(variables=vars_to_plot)

    def plot_in_latent_space(  # noqa: C901, PLR0912, PLR0915
        self,
        plot_data: pd.DataFrame,
        latent_model: str | LatentModelType,
        pcs: list[int],
        n_kmeans_clusters: int | None = None,
        data_slicer: str | None = None,
        size_dimension: str | None = None,
        color_dimension: str | None = None,
        colorscale: str | None = None,
        marker_symbol_dimension: str | None = None,
        loadings: list[str] | None = None,
    ) -> go.Figure:
        """Returns the plot of the data in the latent space."""

        latent_model = LatentModelType(latent_model)

        if VariableReference.LOG_PROBABILITY not in plot_data.columns:
            plot_data = plot_data.join(self.compute_datapoints_probas())

        plot_data = plot_data.join(
            self.get_pipeline_output(model_ref=latent_model, pc_list=pcs)
        ).join(self.predict_dominant_component().astype(str))

        # Figure title
        fig_title = f"{latent_model.value.upper()} on multidimensional data"

        if plot_data.index.names[0] is None:
            plot_data.index.name = "index"

        # Data dimensions
        data_dimensions = plot_data.filter(
            regex=self.user_session.models[latent_model].projection_dimension
        ).columns
        if latent_model is LatentModelType.PCA:
            data_dimensions = data_dimensions[pcs]

        # Extract plot cues
        plot_data, plot_cues = self._extract_plot_cues(
            plot_data=plot_data,
            data_dimensions=data_dimensions,
            n_kmeans_clusters=n_kmeans_clusters,
            data_slicer=data_slicer,
            color_dimension=color_dimension,
            marker_symbol_dimension=marker_symbol_dimension,
            size_dimension=size_dimension,
        )

        fig = sts.plot(
            data=plot_data.sort_values([plot_cues.slicer, plot_cues.color]),
            **get_plot_dimensions(data_dimensions),
            mode="markers",
            color=plot_cues.color,
            slicer=plot_cues.slicer,
            size=plot_cues.size,
            marker=plot_cues.marker,
            opacity=plot_cues.opacity,
            color_palette=colorscale,
            title=fig_title,
            text=plot_cues.text,
            shared_coloraxis=True,
        )

        if latent_model is LatentModelType.PCA:
            pc_plotter = PcSpacePlotAnnotater(
                pca_model=self.user_session.models[latent_model], pcs=pcs, pca_fig=fig
            )
            # Annotate axes
            fig = pc_plotter.annotate_variance()

            # Plot loadings
            if loadings:
                fig = pc_plotter.plot_loadings_direction(variables=loadings)

        # Force-show the legend
        fig.update_layout(showlegend=True)

        # Custom data
        try:
            fig = add_customdata(
                fig,
                plot_data.sort_values([plot_cues.slicer, plot_cues.color]).reset_index(),
                plot_cues.slicer,
            )
        except KeyError as exc:
            raise LatentSpacePlottingError("Impossible to update trace custom data") from exc

        return fig

    def plot_in_original_space(
        self,
        graph_type: str,
        data: pd.DataFrame,
        data_slicer: str | None = None,
        size_dimension: str | None = None,
        color_dimension: str | None = None,
        colorscale: str | None = None,
        marker_symbol_dimension: str | None = None,
    ) -> go.Figure:
        """Returns a scatter or a heatmap plot of the original data."""

        title = "Selected data points"
        graph_type = PlotType(graph_type)
        data_dimensions = get_plot_dimensions(data.columns)

        plot_data = (
            data.join(self.compute_datapoints_probas())
            .join(self.predict_dominant_component().astype(str))
            .join(
                self.user_session.data_wrapper.data.loc[
                    :,
                    [
                        col
                        for col in self.user_session.data_wrapper.data.columns
                        if col not in data.columns
                    ],
                ]
            )
        )

        # Extract plot cues
        plot_data, plot_cues = self._extract_plot_cues(
            plot_data=plot_data,
            data_dimensions=[dim for dim in data_dimensions.values() if dim is not None],
            data_slicer=data_slicer,
            color_dimension=color_dimension,
            marker_symbol_dimension=marker_symbol_dimension,
            size_dimension=size_dimension,
        )

        match graph_type:
            case PlotType.SCATTER:
                fig = sts.plot(
                    data=plot_data.sort_values([plot_cues.slicer, plot_cues.color]),
                    **data_dimensions,
                    mode="markers",
                    color=plot_cues.color,
                    slicer=plot_cues.slicer,
                    size=plot_cues.size,
                    marker=plot_cues.marker,
                    opacity=plot_cues.opacity,
                    color_palette=colorscale,
                    shared_coloraxis=True,
                    text=plot_cues.text,
                    title=title,
                )

            case PlotType.HEATMAP:
                fig = sts.heatmap(
                    data=plot_data.melt(
                        ignore_index=False,
                        id_vars=[col for col in plot_data.columns if col not in data.columns],
                    ),
                    y=VariableReference.UNIQUE_ID,
                    x="variable",
                    z="value",
                    slicer=plot_cues.slicer,
                    opacity=plot_cues.opacity,
                    color_palette=colorscale,
                    shared_coloraxis=True,
                    text=plot_cues.text,
                    title=title,
                )
                fig.update_yaxes(type="category")

        return fig
