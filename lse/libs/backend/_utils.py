from __future__ import annotations

import pandas as pd
import plotly.graph_objs as go

from lse.libs.data_models import VariableReference


def add_customdata(fig: go.Figure, plot_data: pd.DataFrame, slicer: str) -> go.Figure:
    """Adds {obj}`VariableReference.UNIQUE_ID` to the `customdata` field of Plotly Traces."""
    for level in plot_data[slicer].unique():
        fig.for_each_trace(
            lambda trace, level=level: (
                trace.update(
                    customdata=plot_data.loc[
                        plot_data[slicer] == level,
                        VariableReference.UNIQUE_ID.value,
                    ]
                )
                if trace.name == str(level)
                else ()
            )
        )

    return fig
