from typing import TypedDict

from mltoolbox.latent_space import TSNELatentSpace

from lse.libs.data_models import LatentModelType


class LatentModelArg(TypedDict):
    perplexity: int


latent_models = {LatentModelType.TSNE: TSNELatentSpace}
