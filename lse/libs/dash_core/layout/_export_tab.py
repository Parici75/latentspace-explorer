import pandas as pd
from dash import dash_table, dcc, html

from lse.libs.dash_core import components_models as cm
from lse.libs.data_models import PlotType


def build_tab() -> dcc.Tab:
    """Build the export tab."""
    return dcc.Tab(
        label="Export",
        children=[
            html.Div(
                [
                    html.Div(
                        [
                            html.Button(
                                "Export Latent Model as Pickle",
                                id=cm.ExportComponent.PREPARE_EXPORT_LATENT_SPACE_MODEL,
                                n_clicks=0,
                                disabled=True,
                                className="button",
                            ),
                            html.P(id=cm.StatusComponent.EXPORT_STATUS),
                            dcc.Download(
                                id=cm.ExportComponent.EXPORT_LATENT_SPACE_MODEL,
                            ),
                            html.Div(
                                id=cm.OutputComponent.DOWNLOAD_AREA,
                                children=[],
                            ),
                        ],
                        className="button-container",
                    ),
                    dcc.Tabs(
                        [
                            # Data table
                            dcc.Tab(
                                label="Data table",
                                children=[
                                    dash_table.DataTable(
                                        id=cm.OutputComponent.EXPORT_DATA_TABLE,
                                        sort_action="native",
                                        page_action="native",
                                        columns=[],
                                        data=pd.DataFrame().to_dict("records"),
                                        style_header={"fontWeight": "bold"},
                                        row_selectable=False,
                                        column_selectable=False,
                                        export_columns="all",
                                        selected_rows=[],
                                        selected_columns=[],
                                        style_table={"width": "inherit"},
                                        fill_width=False,
                                        style_cell={
                                            "whiteSpace": "normal",
                                            "height": "auto",
                                        },
                                        style_cell_conditional=[],
                                        export_format="csv",
                                    )
                                ],
                            ),
                            # Data plot
                            dcc.Tab(
                                label="Data plot",
                                children=[
                                    # Browse data tab
                                    html.Div(
                                        [
                                            dcc.Graph(
                                                id=cm.PlotAreaComponent.ORIGINAL_COORDINATES_PLOT,
                                            )
                                        ],
                                        className="plot-inset",
                                    ),
                                    html.Div(
                                        [
                                            dcc.RadioItems(
                                                id=cm.PlotControlComponent.PLOT_TYPE,
                                                options=[
                                                    {
                                                        "label": (member.value),
                                                        "value": (member.value),
                                                    }
                                                    for member in PlotType
                                                ],
                                                value=[member.value for member in PlotType][0],
                                            ),
                                        ],
                                        className="selector",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ]
            )
        ],
        className="section-tab",
    )
