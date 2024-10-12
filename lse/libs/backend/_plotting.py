from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import statsplotly as sts
from mltoolbox.latent_space import PCALatentSpace
from numpy.typing import NDArray
from pymodules.pandas_utils import format_series_name

from lse.libs.data_models import BaseModel, VariableReference

AXES_REFERENCES = ["xaxis", "yaxis", "zaxis"]


logger = logging.getLogger()


class GenericPlotter:

    @staticmethod
    def plot_radar(datapoint_series: pd.Series, radial_range: tuple[Any, Any]) -> go.Figure:
        """Plots the features of a datapoint in a form of a radar.

        Args:
            datapoint_series:
                A `pandas.Series` of the coordinates of the data points.
            radial_range:
                A tuple specifying the range of the radar plot.

        Returns:
            A `plotly.graph_objects.Figure`.
        """

        # Draw figure
        fig = go.Figure(
            go.Scatterpolar(
                r=np.append(datapoint_series, datapoint_series.iloc[0]),
                theta=np.append(datapoint_series.index, datapoint_series.index[0]),
                line_color="black",
                fill="toself",
                showlegend=False,
            )
        )

        # Title
        title = f"Data point {format_series_name(datapoint_series.name)}"
        fig.layout.title.text = title

        # Set Radial range
        if radial_range is not None:
            fig.update_polars(
                radialaxis={"range": radial_range, "gridwidth": 2, "tickfont": {"size": 16}},
                angularaxis={"tickfont": {"size": 16}},
                angularaxis_type="category",
            )

        return fig


def get_plot_dimensions(
    dimension_array: list[Any],
) -> dict[str, str | None]:
    dimensions_dict = {}

    dimensions_dict["x"] = (
        dimension_array[0]
        if len(dimension_array) >= 2  # noqa: PLR2004
        else VariableReference.UNIQUE_ID
    )
    dimensions_dict["y"] = dimension_array[0] if len(dimension_array) == 1 else dimension_array[1]
    dimensions_dict["z"] = (
        dimension_array[2] if len(dimension_array) >= 3 else None  # noqa: PLR2004
    )

    return dimensions_dict


def concatenate_text_dimensions(plot_data: pd.DataFrame, plot_cues: tuple[str | None, ...]) -> str:
    text_dimension = "+".join(
        (VariableReference.NEGATIVE_LOG_PROBABILITY, VariableReference.UNIQUE_ID)
    )
    for cue in plot_cues:
        if cue is not None and cue not in text_dimension and cue in plot_data.reset_index().columns:
            text_dimension = "+".join((text_dimension, cue))

    return text_dimension


class PcSpacePlotter(BaseModel):
    """PCA plotting methods."""

    pca_model: PCALatentSpace
    pcs: list[int] | None = None

    @property
    def pc_indices(self) -> list[int]:
        return self.pcs or list(range(len(self.pca_model.loadings)))

    def loadings_barplot(self, variables: list[str] | None = None) -> go.Figure:
        """Plots loadings bar plot.

        Args:
            variables:
                The variables whose contributions to plot.

        Returns:
            A `plotly.graph_objects.Figure`

        """
        if variables is None:
            var_indices = np.arange(len(self.pca_model.orig_vars))
        else:
            var_indices = np.array(
                [np.where(self.pca_model.orig_vars == var)[0][0] for var in variables]
            )

        fig = sts.barplot(
            x="Variable",
            y="Loadings",
            title=f"Loadings on Principal Components {[pc_idx+1 for pc_idx in self.pc_indices]}",
            slicer="Principal Component",
            data=pd.concat(
                [
                    pd.Series(
                        self.pca_model.loadings[var_indices, pc_idx],
                        index=self.pca_model.orig_vars[var_indices],
                        name="Loadings",
                    )
                    for pc_idx in self.pc_indices
                ],
                keys=[str(pc + 1) for pc in self.pc_indices],
                names=["Principal Component", "Variable"],
            ),
        )

        return fig


class PcSpacePlotAnnotater(PcSpacePlotter):
    """PCA plot annotation methods."""

    pca_fig: go.Figure

    @property
    def explained_variance_ratios(self) -> NDArray[Any]:
        return self.pca_model.model.explained_variance_ratio_

    def plot_loadings_direction(self, variables: list[str]) -> go.Figure:
        """Plots loadings on a PCA plot.

        Args:
            variables:
                The variables whose loadings to plot

        Returns:
            The updated `plotly.graph_objects.Figure`.
        """
        # Retrieve the indices
        var_indices = [np.where(self.pca_model.orig_vars == var)[0][0] for var in variables]

        plotting_params = {
            "mode": "lines",
            "marker": {"color": "black"},
            "line": {"width": 2},
            "showlegend": False,
        }
        annotations: list[dict[str, Any]] = []

        # Loop through loadings
        for var_idx, var_name in zip(var_indices, variables, strict=True):
            plotting_params.update({"name": var_name})
            coordinates = {
                axis_coord: [
                    0,
                    self.pca_model.loadings[var_idx, idx],
                ]
                for axis_coord, idx in zip(
                    [x.replace("axis", "") for x in AXES_REFERENCES],
                    self.pc_indices,
                    strict=False,
                )
            }
            if self.pca_fig.layout.scene.xaxis.title.text is None:
                plotting_params.update({"line": {"width": 1}})
                self.pca_fig.add_trace(go.Scatter(**coordinates, **plotting_params))
            else:
                plotting_params.update({"line": {"width": 4}})
                self.pca_fig.add_trace(
                    go.Scatter3d(
                        **coordinates,
                        **plotting_params,
                    )
                )

            # Annotate tip of the loading with variable name
            annotations.append(
                {
                    **{dimension: coordinate[1] for dimension, coordinate in coordinates.items()},
                    "text": var_name,
                    "font": {"color": "black", "size": 16},
                    "xshift": 0,
                    "yshift": -10,
                    "showarrow": False,
                    "arrowhead": 1,
                    "arrowwidth": 1,
                    "ax": -20,
                    "ay": -40,
                }
            )

        if self.pca_fig.layout.scene.xaxis.title.text is None:
            self.pca_fig.update_layout({"annotations": annotations})
        else:
            self.pca_fig.update_scenes({"annotations": annotations})

        return self.pca_fig

    def erase_loadings(self) -> go.Figure:
        """Erase loadings."""
        if self.pca_fig.layout.scene.xaxis.title.text is None:
            self.pca_fig.layout.annotations = []
        else:
            self.pca_fig.layout.scene.annotations = []

        for trace in self.pca_fig.data:
            if trace.mode == "lines":
                trace.visible = False
        return self.pca_fig

    def annotate_variance(self) -> go.Figure:
        """Annotates PCA plot axes with their variance."""

        if len(self.pc_indices) == 1:
            axes_references = [AXES_REFERENCES[1]]
        else:
            axes_references = AXES_REFERENCES[: len(self.pc_indices)]

        explained_variance_ratio: float

        for explained_variance_ratio, axis in zip(
            self.explained_variance_ratios[self.pc_indices], axes_references, strict=False
        ):
            axis_title_suffix = f" ({explained_variance_ratio * 100:.2f} %)"
            try:
                self.pca_fig.update_scenes(
                    {
                        axis: {
                            "title": {
                                "text": " ".join(
                                    (
                                        self.pca_fig.layout.scene[axis].title.text,
                                        axis_title_suffix,
                                    )
                                )
                            }
                        }
                    }
                )
            except TypeError:
                try:
                    self.pca_fig.update_layout(
                        {
                            axis: {
                                "title": {
                                    "text": " ".join(
                                        (
                                            self.pca_fig.layout[axis].title.text,
                                            axis_title_suffix,
                                        )
                                    )
                                }
                            }
                        }
                    )
                except Exception:
                    logger.exception("Error annotating variance")

        return self.pca_fig
