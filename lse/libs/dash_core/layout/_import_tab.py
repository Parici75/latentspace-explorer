import pandas as pd
from dash import dash_table, dcc, html

from lse.libs.dash_core import components_models as cm


def _build_data_loading_div(enable_data_upload: bool) -> html:
    if enable_data_upload:
        return dcc.Upload(
            id=cm.DataLoadingComponent.UPLOAD_DATA,
            children=html.Div(
                [
                    "Drag and Drop or ",
                    html.A("Select Files"),
                ]
            ),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            # Allow multiple files to be uploaded
            multiple=True,
        )

    return html.Button(
        id=cm.DataLoadingComponent.LOAD_DATA,
        n_clicks=0,
        children=cm.DataLoadingComponent.LOAD_DATA.child,
        className="button",
    )


def build_tab(dashboard_title: str, enable_data_upload: bool) -> dcc.Tab:
    """Build the import tab."""
    return dcc.Tab(
        label="Import",
        children=[
            html.Div(
                [html.H4(dashboard_title)],
                className="row",
            ),
            html.Div(
                [
                    _build_data_loading_div(enable_data_upload),
                    dcc.Loading(
                        id=cm.ProcessComponent.DATA_LOADING_PROCESS,
                        children=dcc.Store(id=cm.CheckpointComponent.DATA),
                        type="cube",
                    ),
                ],
                className="button-container",
            ),
            # Selector row
            html.Div(
                [
                    html.Label(
                        [
                            "Feature selection",
                            html.Div(
                                [
                                    dcc.Dropdown(
                                        id=cm.DropdownComponent.FEATURES,
                                        options={},
                                        value=[],
                                        multi=True,
                                        disabled=False,
                                    )
                                ],
                                className="selector",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Button(
                                id=cm.ComputeComponent.RUN_MODEL_PIPELINE,
                                n_clicks=0,
                                children=cm.ComputeComponent.RUN_MODEL_PIPELINE.child,
                                className="button",
                                disabled=True,
                            ),
                            html.Div(
                                [
                                    dcc.Checklist(
                                        id=cm.ComputeComponent.STANDARDIZE,
                                        options=[
                                            {
                                                "label": "Standardize",
                                                "value": 1,
                                            },
                                        ],
                                        value=[],
                                    ),
                                ],
                                className="checker",
                                id=cm.ComputeComponent.STANDARDIZE.container,
                            ),
                            dcc.Loading(
                                id=cm.ProcessComponent.PCA_MODEL_PROCESS,
                                children=[dcc.Store(id=cm.CheckpointComponent.PCA_MODEL)],
                                type="graph",
                            ),
                            html.P(id=cm.StatusComponent.LOADING_STATUS),
                        ],
                        className="button-container",
                    ),
                ],
                className="selector-row row",
            ),
            html.Div(
                [
                    html.Label(
                        [
                            ("Number of clusters"),
                            html.Div(
                                [
                                    dcc.Input(
                                        id=cm.ComputeComponent.INITIAL_GAUSSIAN_MIXTURE_GUESS,
                                        type="number",
                                        step=1,
                                        value=1,
                                    ),
                                ],
                                className="selector",
                            ),
                        ]
                    ),
                ],
                className="selector-block",
            ),
            html.Div(
                [
                    dash_table.DataTable(
                        id=cm.OutputComponent.PREVIEW_DATA_TABLE,
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
                ]
            ),
        ],
        className="section-tab",
    )
