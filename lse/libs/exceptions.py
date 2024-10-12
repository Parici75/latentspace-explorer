"""Project specific exceptions"""

from lse.libs.data_models import LatentModelType


class DataProcessingError(Exception):
    """Raises when an error occurs during data processing."""


class LatentSpaceModelFittingError(Exception):
    def __init__(self, model: LatentModelType) -> None:
        message = f"{model.value} could not be fitted"
        super().__init__(message)


class LatentSpacePlottingError(Exception):
    """Raises when an error occurs during latent space plotting."""


class DataReconstructionError(Exception):
    def __init__(self, message: str = "Data could not be reconstructed from PCA") -> None:
        super().__init__(message)
