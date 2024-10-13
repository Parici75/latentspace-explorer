from dash import dcc, html

from lse.libs.dash_core import components_models as cm
from lse.libs.data_models import ClusterType, LatentModelType
from lse.libs.metaparameters import (
    DEFAULT_PERPLEXITY,
    DEFAULT_VARIANCE_RETAINED,
    N_RANDOM_SAMPLE,
)


def build_tab() -> dcc.Tab:
    """Build the exploration tab."""
    return dcc.Tab(
        label="Explore",
        children=[
            # Plots row
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label(
                                                [
                                                    "Data slicer",
                                                    html.Div(
                                                        [
                                                            dcc.Dropdown(
                                                                id=cm.DropdownComponent.DATA_SLICER,
                                                                options=[],
                                                                value=None,
                                                                multi=False,
                                                                disabled=False,
                                                            )
                                                        ],
                                                        className="selector",
                                                    ),
                                                ]
                                            ),
                                            html.Label(
                                                [
                                                    "Marker size",
                                                    html.Div(
                                                        [
                                                            dcc.Dropdown(
                                                                id=cm.DropdownComponent.SIZE_CODE,
                                                                options=[],
                                                                value=None,
                                                                multi=False,
                                                                disabled=False,
                                                            )
                                                        ],
                                                        className="selector",
                                                    ),
                                                ]
                                            ),
                                            html.Label(
                                                [
                                                    "Marker symbol",
                                                    html.Div(
                                                        [
                                                            dcc.Dropdown(
                                                                id=cm.DropdownComponent.MARKER_CODE,
                                                                options=[],
                                                                value=None,
                                                                multi=False,
                                                                disabled=False,
                                                            )
                                                        ],
                                                        className="selector",
                                                    ),
                                                ]
                                            ),
                                            html.Label(
                                                [
                                                    "Color code",
                                                    html.Div(
                                                        [
                                                            dcc.Dropdown(
                                                                id=cm.DropdownComponent.COLOR_CODE,
                                                                options=[],
                                                                value=None,
                                                                multi=False,
                                                                disabled=False,
                                                            )
                                                        ],
                                                        className="selector",
                                                    ),
                                                ]
                                            ),
                                            html.Label(
                                                [
                                                    "Colorscale",
                                                    html.Div(
                                                        [
                                                            dcc.Dropdown(
                                                                id=cm.DropdownComponent.COLORSCALE,
                                                                options=[],
                                                                value=None,
                                                                multi=False,
                                                                disabled=False,
                                                            )
                                                        ],
                                                        className="selector",
                                                    ),
                                                ]
                                            ),
                                        ],
                                        className="selector-row row",
                                    ),
                                    # Plot
                                    html.Div(
                                        [
                                            dcc.Loading(
                                                dcc.Graph(
                                                    id=cm.PlotAreaComponent.DATA_PROJECTION,
                                                    hoverData=None,
                                                    selectedData=None,
                                                ),
                                                type="dot",
                                            ),
                                            html.Div(
                                                [
                                                    html.P("Probability filter"),
                                                    dcc.RangeSlider(
                                                        id=cm.PlotControlComponent.PROBA_FILTER,
                                                        updatemode="mouseup",
                                                        min=0,
                                                        max=1,
                                                        step=0.0001,
                                                        value=[0, 1],
                                                        marks={0: "-", 1: "+"},
                                                        vertical=True,
                                                        tooltip={"always_visible": False},
                                                        className="slider",
                                                    ),
                                                ],
                                                className="vert-slider-container",
                                            ),
                                        ],
                                        className="slider-plot",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        [
                                                            ("Latent Space plot"),
                                                            html.Div(
                                                                [
                                                                    dcc.RadioItems(
                                                                        id=cm.PlotControlComponent.LATENT_SPACE_PLOT,
                                                                        options=[
                                                                            {
                                                                                "label": (
                                                                                    member.value
                                                                                ),
                                                                                "value": (
                                                                                    member.value
                                                                                ),
                                                                            }
                                                                            for member in LatentModelType
                                                                        ],
                                                                        value=[
                                                                            member.value
                                                                            for member in LatentModelType
                                                                        ][0],
                                                                    ),
                                                                ],
                                                                className=("selector"),
                                                            ),
                                                        ],
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Label(
                                                                [
                                                                    ("Number of plot clusters"),
                                                                    html.Div(
                                                                        [
                                                                            dcc.Input(
                                                                                id=cm.InputComponent.N_KMEANS_CLUSTERS,
                                                                                type="number",
                                                                                min=1,
                                                                                step=1,
                                                                                value=None,
                                                                                className="number-input",
                                                                            ),
                                                                        ],
                                                                        className=("selector"),
                                                                    ),
                                                                ]
                                                            ),
                                                        ],
                                                        className="selector",
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.Button(
                                                                        id=cm.PlotControlComponent.RANDOM_SAMPLE,
                                                                        n_clicks=0,
                                                                        children=(
                                                                            f"Draw a {cm.PlotControlComponent.RANDOM_SAMPLE.child}"
                                                                        ),
                                                                        className="button",
                                                                    ),
                                                                    dcc.Input(
                                                                        id=cm.PlotControlComponent.RANDOM_SAMPLE_SIZE,
                                                                        type="number",
                                                                        step=1,
                                                                        placeholder=(
                                                                            f"{N_RANDOM_SAMPLE} samples"
                                                                        ),
                                                                    ),
                                                                ],
                                                                className="button-container",
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Button(
                                                                        id=cm.PlotControlComponent.RESET_SAMPLE,
                                                                        n_clicks=0,
                                                                        children=(
                                                                            "Display full dataset"
                                                                        ),
                                                                        className="button",
                                                                    ),
                                                                ],
                                                                className="button-container",
                                                            ),
                                                        ],
                                                        className="selector",
                                                    ),
                                                ],
                                                className="selector-block",
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.P("Perplexity"),
                                                            dcc.Slider(
                                                                id=cm.PlotControlComponent.PERPLEXITY,
                                                                updatemode=("mouseup"),
                                                                min=5,
                                                                max=50,
                                                                step=1,
                                                                value=DEFAULT_PERPLEXITY,
                                                                marks={
                                                                    5: "-",
                                                                    50: "+",
                                                                },
                                                                vertical=False,
                                                                tooltip={"always_visible": (False)},
                                                                disabled=True,
                                                                className=("slider"),
                                                            ),
                                                        ],
                                                        id="model-slider",
                                                        className="horiz-slider-container",
                                                    ),
                                                    html.Div(
                                                        dcc.Checklist(
                                                            id=cm.PlotControlComponent.LOADINGS_CHECKER,
                                                            options=[
                                                                {
                                                                    "label": ("Plot loadings"),
                                                                    "value": 1,
                                                                    "disabled": (True),
                                                                },
                                                            ],
                                                            value=[],
                                                        ),
                                                        className="checker",
                                                    ),
                                                ],
                                                className="selector-block",
                                            ),
                                            html.Div(
                                                [
                                                    # Number of kernel
                                                    html.Label(
                                                        [
                                                            ("Number of" " gaussian" " kernels"),
                                                            html.Div(
                                                                [
                                                                    dcc.Input(
                                                                        id=cm.InputComponent.N_GAUSSIAN_KERNELS,
                                                                        type="number",
                                                                        min=1,
                                                                        step=1,
                                                                        value=None,
                                                                        className="number-input",
                                                                    ),
                                                                    dcc.Loading(
                                                                        id=cm.ProcessComponent.GAUSSIAN_MODEL_PROCESS,
                                                                        children=[
                                                                            dcc.Store(
                                                                                id=cm.CheckpointComponent.ANOMALY_MODEL
                                                                            )
                                                                        ],
                                                                        type="dot",
                                                                    ),
                                                                ],
                                                                className=("selector"),
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.P(
                                                                        "Bayesian"
                                                                        " information"
                                                                        " criterion"
                                                                    ),
                                                                    html.P(
                                                                        id=cm.OutputComponent.BIC
                                                                    ),
                                                                ],
                                                                className=("text-box"),
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.P("Gini impurity"),
                                                                    html.P(
                                                                        id=cm.OutputComponent.SLICER
                                                                    ),
                                                                    html.P(
                                                                        id=cm.OutputComponent.GINI
                                                                    ),
                                                                ],
                                                                className="text-box",
                                                                id=cm.OutputComponent.GINI.container,
                                                            ),
                                                            dcc.Store(
                                                                id=cm.CheckpointComponent.FILTERED_DATA
                                                            ),
                                                        ]
                                                    ),
                                                ],
                                                className="selector-block",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        [
                                                            "Show clusters",
                                                            html.Div(
                                                                [
                                                                    dcc.RadioItems(
                                                                        id=cm.PlotControlComponent.CLUSTERIZE_RADIO_BUTTON,
                                                                        options=[
                                                                            {
                                                                                "label": "Disabled",
                                                                                "value": False,
                                                                            },
                                                                            {
                                                                                "label": (
                                                                                    str(
                                                                                        ClusterType.PLOT
                                                                                    )
                                                                                ),
                                                                                "value": (
                                                                                    ClusterType.PLOT
                                                                                ),
                                                                            },
                                                                            {
                                                                                "label": (
                                                                                    str(
                                                                                        ClusterType.GAUSSIAN_COMPONENT
                                                                                    )
                                                                                ),
                                                                                "value": (
                                                                                    ClusterType.GAUSSIAN_COMPONENT
                                                                                ),
                                                                            },
                                                                        ],
                                                                        value=False,
                                                                    )
                                                                ],
                                                                className=cm.PlotControlComponent.CLUSTERIZE_RADIO_BUTTON.container,
                                                            ),
                                                        ]
                                                    )
                                                ],
                                                className="selector-block",
                                            ),
                                        ],
                                        className="selector-row row",
                                    ),
                                ],
                                className="plot-inset",
                            ),
                        ],
                        className="column",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label(
                                                [
                                                    "Signature features",
                                                    html.Div(
                                                        [
                                                            dcc.Dropdown(
                                                                id=cm.DropdownComponent.SIGNATURE,
                                                                options=[],
                                                                value=[],
                                                                multi=True,
                                                                disabled=False,
                                                                clearable=False,
                                                            )
                                                        ],
                                                        className="selector",
                                                    ),
                                                    html.Button(
                                                        id=cm.PlotControlComponent.RESET_SIGNATURE,
                                                        n_clicks=0,
                                                        children="Reset",
                                                        className="button",
                                                    ),
                                                    html.Button(
                                                        id=cm.PlotControlComponent.GET_FULL_SIGNATURE,
                                                        n_clicks=0,
                                                        children="Select all",
                                                        className="button",
                                                    ),
                                                ]
                                            ),
                                            html.Label(
                                                [
                                                    "Principal component",
                                                    html.Div(
                                                        [
                                                            dcc.Dropdown(
                                                                id=cm.DropdownComponent.PC_SELECTOR,
                                                                options=[],
                                                                value=[],
                                                                multi=True,
                                                                disabled=False,
                                                                clearable=False,
                                                            ),
                                                            dcc.Loading(
                                                                id=cm.ProcessComponent.LATENT_MODEL_PROCESS,
                                                                children=[
                                                                    dcc.Store(
                                                                        id=cm.CheckpointComponent.LATENT_MODEL
                                                                    ),
                                                                ],
                                                                type="graph",
                                                            ),
                                                        ],
                                                        className="selector",
                                                    ),
                                                    html.Button(
                                                        id=cm.PlotControlComponent.RESET_COMPONENTS,
                                                        n_clicks=0,
                                                        children="Reset",
                                                        className="button",
                                                    ),
                                                    html.Button(
                                                        id=cm.PlotControlComponent.GET_FULL_COMPONENTS,
                                                        n_clicks=0,
                                                        children="Select all",
                                                        className="button",
                                                    ),
                                                ]
                                            ),
                                        ],
                                        className="selector-row row",
                                    ),
                                    # Tabs
                                    dcc.Tabs(
                                        [
                                            dcc.Tab(
                                                label="Datum reconstruction",
                                                children=[
                                                    # Browse data tab
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                id=cm.PlotAreaComponent.DATA_POINT_SIGNATURE,
                                                            ),
                                                            html.Div(
                                                                html.Label(
                                                                    [
                                                                        ("Variance" " to retain"),
                                                                        dcc.Slider(
                                                                            id=cm.PlotControlComponent.VARIANCE_FILTER,
                                                                            updatemode="mouseup",
                                                                            min=0,
                                                                            max=1,
                                                                            step=0.1,
                                                                            value=DEFAULT_VARIANCE_RETAINED,
                                                                            marks={
                                                                                0.1: "10%",
                                                                                1: "100%",
                                                                            },
                                                                            vertical=False,
                                                                            tooltip={
                                                                                "always_visible": False  # noqa: E501
                                                                            },
                                                                            className="slider",
                                                                        ),
                                                                    ],
                                                                    className="horiz-slider-container",
                                                                ),
                                                            ),
                                                        ],
                                                        className="plot-inset",
                                                    ),
                                                ],
                                            ),
                                            # Loadings
                                            dcc.Tab(
                                                label="PC loadings",
                                                children=[
                                                    dcc.Graph(
                                                        id=cm.PlotAreaComponent.LOADINGS_PLOT,
                                                    )
                                                ],
                                            ),
                                        ]
                                    ),
                                ],
                                className="plot-inset",
                            )
                        ],
                        className="column",
                    ),
                ],
                className="plots-row row",
            ),
        ],
        className="section-tab",
    )
