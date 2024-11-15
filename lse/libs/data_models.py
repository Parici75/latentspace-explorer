from enum import Enum

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict


class BaseModel(PydanticBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class PrettyEnum(Enum):
    def __str__(self) -> str:
        return " ".join(self.value.split("_")).capitalize()


class ClusterType(str, PrettyEnum):
    PLOT = "plot"
    GAUSSIAN_COMPONENT = "gaussian_component"


class VariableReference(str, PrettyEnum):
    LOG_PROBABILITY = "log_probability"
    NEGATIVE_LOG_PROBABILITY = "negative_log_probability"
    DOMINANT_COMPONENT = "dominant_component"
    KMEANS_CLUSTER = "k-means_cluster"
    UNIQUE_ID = "unique_id"


class PlotType(str, PrettyEnum):
    SCATTER = "scatter"
    HEATMAP = "heatmap"


class LatentModelType(str, Enum):
    PCA = "pca"
    TSNE = "tsne"
