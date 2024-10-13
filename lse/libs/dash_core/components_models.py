from enum import Enum


class _Component(Enum):
    @property
    def child(self) -> str:
        return " ".join([x.capitalize() for x in self.value.split("-")])

    @property
    def container(self) -> str:
        return "-".join([self.value, "container"])


class DataLoadingComponent(str, _Component):
    LOAD_DATA = "load-data"
    UPLOAD_DATA = "upload-data"


class ComputeComponent(str, _Component):
    RUN_MODEL_PIPELINE = "run-model-pipeline"
    STANDARDIZE = "standardize-checker"
    INITIAL_GAUSSIAN_MIXTURE_GUESS = "initial-gaussian-mixture-guess"


class ProcessComponent(str, _Component):
    DATA_LOADING_PROCESS = "data-loading-process"
    PCA_MODEL_PROCESS = "pca-model-process"
    LATENT_MODEL_PROCESS = "latent-model-process"
    GAUSSIAN_MODEL_PROCESS = "gaussian-model-process"


class StatusComponent(str, _Component):
    LOADING_STATUS = "loading-status"
    EXPORT_STATUS = "export-status"


class CheckpointComponent(str, _Component):
    DATA = "data"
    NUMERIC_FEATURES = "numeric-features"
    PCA_MODEL = "pca-model"
    LATENT_MODEL = "latent-model"
    ANOMALY_MODEL = "anomaly-model"
    FILTERED_DATA = "filtered-data"


class PlotControlComponent(str, _Component):
    VARIANCE_FILTER = "variance-filter"
    PROBA_FILTER = "proba-filter"
    RANDOM_SAMPLE = "random-sample"
    RANDOM_SAMPLE_SIZE = "random-sample-size"
    RESET_SAMPLE = "reset-sample"
    LOADINGS_CHECKER = "loadings-checker"
    LATENT_SPACE_PLOT = "latent-space-plot"
    PERPLEXITY = "perplexity"
    CLUSTERIZE_RADIO_BUTTON = "clusterize-radio-button"
    RESET_SIGNATURE = "reset-signatu<re"
    GET_FULL_SIGNATURE = "get-full-signature"
    GET_FULL_COMPONENTS = "get-full-components"
    RESET_COMPONENTS = "reset-components"
    PLOT_TYPE = "plot-type"


class PlotAreaComponent(str, Enum):
    DATA_PROJECTION = "multidimensional-projection"
    DATA_POINT_SIGNATURE = "data-point-signature"
    LOADINGS_PLOT = "loadings-plot"
    ORIGINAL_COORDINATES_PLOT = "original-coordinates-plot"


class DropdownComponent(str, _Component):
    FEATURES = "features"
    PC_SELECTOR = "pc-selector"
    SIGNATURE = "signature"
    DATA_SLICER = "data-slicer"
    SIZE_CODE = "size-code"
    COLOR_CODE = "color-code"
    COLORSCALE = "colorscale"
    MARKER_CODE = "marker-code"


class InputComponent(str, _Component):
    N_KMEANS_CLUSTERS = "n-kmeans-clusters"
    N_GAUSSIAN_KERNELS = "n-gaussian-kernels"


class OutputComponent(str, _Component):
    BIC = "bic"
    DOWNLOAD_AREA = "download-area"
    GINI = "gini"
    SLICER = "slicer"
    PREVIEW_DATA_TABLE = "preview-data-table"
    EXPORT_DATA_TABLE = "export_data-table"


class ExportComponent(str, _Component):
    PREPARE_EXPORT_LATENT_SPACE_MODEL = "prepare-export-latent-space-model"
    EXPORT_LATENT_SPACE_MODEL = "export-latent-space-model"


class SessionComponent(str, _Component):
    SESSION_ID = "session_id"
    PUBSUB = "pub_sub"
