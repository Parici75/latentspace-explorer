from __future__ import annotations

from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Any

import pandas as pd
from mltoolbox.anomaly_detection import GaussianAnomalyQuantifier
from mltoolbox.latent_space import PCALatentSpace, TSNELatentSpace
from pandas.api.types import is_numeric_dtype

from lse.libs.data_models import BaseModel, LatentModelType


class Session(BaseModel):
    session_id: str


class RedisKeyType(str, Enum):
    FEATURES = "features"
    LATENT_MODELS = "latent_models"
    DATA_INFILTER = "data_infilter"
    ANOMALY_QUANTIFIER = "anomaly_quantifier"


class RedisKeyGroupType(str, Enum):
    DATA = "data"
    USER = "user"


class UserSessionParameters(BaseModel):
    latent_models: dict[LatentModelType, PCALatentSpace | TSNELatentSpace]
    data_infilter: pd.Index
    features: list[str]
    anomaly_quantifier: GaussianAnomalyQuantifier


class DataWrapper(BaseModel, metaclass=ABCMeta):

    @property
    @abstractmethod
    def data(self) -> pd.DataFrame: ...

    @property
    def identifiers(self) -> list[str]:
        return list(self.data.columns) + list(self.data.index.names)

    @property
    def identifiers_dtypes(self) -> dict[str, Any]:
        columns_dtypes = self.data.dtypes.to_dict()
        index_dtypes = dict(
            zip(
                self.data.index.names,
                [self.data.index.get_level_values(level).dtype for level in self.data.index.names],
                strict=True,
            )
        )
        return {**columns_dtypes, **index_dtypes}

    @property
    def numerical_identifiers(self) -> list[str]:
        return [
            id_var for id_var, dtype in self.identifiers_dtypes.items() if is_numeric_dtype(dtype)
        ]

    @property
    def numerical_features(self) -> list[str]:
        return [feature for feature in self.data.columns if is_numeric_dtype(self.data[feature])]


class UserSession(Session, metaclass=ABCMeta):

    data_wrapper: DataWrapper

    _data_infilter: pd.Index | None
    _features: list[str] | None
    _anomaly_quantifier: GaussianAnomalyQuantifier | None
    _latent_models: dict[LatentModelType, PCALatentSpace | TSNELatentSpace] | None

    @property
    def pca_data(self) -> pd.DataFrame:
        return self.data_wrapper.data[self.features].dropna()

    @property
    def n_samples(self) -> int:
        return len(self.pca_data)

    @property
    def filtered_data(self) -> pd.DataFrame:
        if (data_infilter := self.data_infilter) is not None:
            return self.data_wrapper.data.loc[data_infilter]
        return self.data_wrapper.data

    @property
    @abstractmethod
    def data_infilter(self) -> pd.Index | None: ...

    @data_infilter.setter
    @abstractmethod
    def data_infilter(self, value: Any) -> None: ...

    @property
    @abstractmethod
    def features(self) -> list[str]: ...

    @features.setter
    @abstractmethod
    def features(self, value: Any) -> None: ...

    @property
    @abstractmethod
    def anomaly_quantifier(self) -> GaussianAnomalyQuantifier | None: ...

    @anomaly_quantifier.setter
    @abstractmethod
    def anomaly_quantifier(self, value: Any) -> None: ...

    @property
    @abstractmethod
    def models(self) -> dict[LatentModelType, PCALatentSpace | TSNELatentSpace]: ...

    @models.setter
    @abstractmethod
    def models(self, value: Any) -> None: ...


class SessionType(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
