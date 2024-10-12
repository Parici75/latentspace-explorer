from __future__ import annotations

import logging
from typing import Any

import matplotlib
import pandas as pd
from dash import html
from statsplotly.utils import rgb_string_array_from_colormap

from lse.libs.data_models import BaseModel, VariableReference
from lse.libs.metaparameters import (
    DEFAULT_N_COLOR_BINS,
    FEATURES_CMAP,
    LOG_PROBABILITY_CMAP,
    MAX_DATATABLE_NCOLUMNS,
)
from lse.libs.utils.plot_style import shifted_colormap

logger = logging.getLogger(__name__)


def get_data_table_css_style(
    df: pd.DataFrame, columns: list[str] | None = None, n_color_bins: int = DEFAULT_N_COLOR_BINS
) -> tuple[list[dict[str, Any]], html.Legend]:
    bounds = [i * (1.0 / n_color_bins) for i in range(n_color_bins + 1)]
    if columns is None:
        df_numeric_columns = df.select_dtypes("number").columns
    else:
        df_numeric_columns = df[columns].select_dtypes("number").columns
        if len(df_numeric_columns) < len(columns):
            logger.warning(
                f"{[col for col in columns if col not in df_numeric_columns]} columns are not"
                " numeric and can not be color-coded"
            )
    styles = []
    legend = []
    for column in df_numeric_columns:
        column_min, column_max = df[column].min(), df[column].max()

        # Get a centered colormap
        cmap: str | matplotlib.colors.Colormap
        if column == VariableReference.LOG_PROBABILITY:
            cmap = LOG_PROBABILITY_CMAP
        else:
            try:
                cmap = shifted_colormap(
                    FEATURES_CMAP, start=0, midpoint=(column_min, column_max), stop=1
                )
            except ZeroDivisionError:
                cmap = FEATURES_CMAP
        try:
            color_palette = rgb_string_array_from_colormap(n_color_bins, cmap)
        except ValueError as exc:
            logger.error(f"Impossible to map {column} data values to a color:\n{exc}")
            continue
        for i in range(1, len(bounds)):
            ranges = [((column_max - column_min) * i) + column_min for i in bounds]
            min_bound = ranges[i - 1]
            max_bound = ranges[i]
            # Set cell color and background color
            background_color = color_palette[i - 1]
            color = "white" if i > len(bounds) / 2.0 else "inherit"

            styles.append(
                {
                    "if": {
                        "filter_query": (
                            "{{{column}}} >= {min_bound}"
                            + (" && {{{column}}} < {max_bound}" if (i < len(bounds) - 1) else "")
                        ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                        "column_id": column,
                    },
                    "backgroundColor": background_color,
                    "color": color,
                }
            )
        legend.append(
            html.Div(
                style={"display": "inline-block", "width": "60px"},
                children=[
                    html.Div(
                        style={
                            "backgroundColor": background_color,
                            "borderLeft": "1px rgb(50, 50, 50) solid",
                            "height": "10px",
                        }
                    ),
                    html.Small(round(min_bound, 2), style={"paddingLeft": "2px"}),
                ],
            )
        )

    return styles, html.Div(legend, style={"padding": "5px 0 5px 0"})


class DataTableFormatter(BaseModel):
    data: pd.DataFrame
    columns: list[dict[str, Any]]
    hidden_columns: list[str]

    @property
    def data_table(self) -> dict[str, Any]:
        return self.data.to_dict("records")

    @classmethod
    def format_table(cls, data: pd.DataFrame) -> DataTableFormatter:
        # Prepare columns
        original_columns = [
            col for col in data.columns if col not in [member.value for member in VariableReference]
        ]

        new_columns = [member.value for member in VariableReference if member.value in data.columns]

        table_columns = [
            {
                "name": " ".join(i.split("_")),
                "id": i,
                "selectable": True,
            }
            for i in list(data.index.names) + original_columns + new_columns
        ]
        return cls(
            data=data.loc[:, original_columns + new_columns].reset_index(),
            columns=table_columns,
            hidden_columns=original_columns[MAX_DATATABLE_NCOLUMNS:],
        )

    def get_style(self, columns_to_style: list[str] | None = None) -> list[dict[str, Any]]:
        if columns_to_style is not None:
            columns_to_style = [col for col in columns_to_style if col in self.data.columns]

        styles, _ = get_data_table_css_style(
            df=self.data,
            columns=columns_to_style,
        )

        return styles
