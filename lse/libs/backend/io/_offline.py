import io
from typing import Self

import pandas as pd
from mltoolbox.anomaly_detection import GaussianAnomalyQuantifier
from mltoolbox.latent_space import PCALatentSpace, TSNELatentSpace

from lse.libs.data_models import LatentModelType
from lse.libs.exceptions import DataProcessingError

from ._online import OnlineDataWrapper
from .data_models import DataWrapper, UserSession, UserSessionParameters


class OfflineDataWrapper(DataWrapper):
    raw_data: io.BytesIO

    @classmethod
    def init_from_session_id(cls, session_id: str) -> Self:
        _data = OnlineDataWrapper(session_id=session_id).get_data_buffer()

        return cls(raw_data=_data)

    @property
    def data(self) -> pd.DataFrame:
        return pd.read_feather(self.raw_data)


class OfflineUserSession(UserSession):
    """A class to store offline usersession data."""

    data_wrapper: OfflineDataWrapper

    def __init__(self, session_id: str, user_session_parameters: UserSessionParameters) -> None:
        super().__init__(
            session_id=session_id,
            data_wrapper=OfflineDataWrapper.init_from_session_id(session_id=session_id),
        )
        self._initialize_keys(user_session_parameters)

    def _initialize_keys(self, user_session_parameters: UserSessionParameters) -> None:

        self._latent_models = user_session_parameters.latent_models
        self._data_infilter = user_session_parameters.data_infilter
        self._features = user_session_parameters.features
        self._anomaly_quantifier = user_session_parameters.anomaly_quantifier

    @property
    def data_infilter(self) -> pd.Index | None:
        return self._data_infilter

    @data_infilter.setter
    def data_infilter(self, value: pd.Index) -> None:
        self._data_infilter = value

    @property
    def features(self) -> list[str]:
        if (features := self._features) is None:
            return self.data_wrapper.numerical_features
        return features

    @features.setter
    def features(self, value: list[str]) -> None:
        if len(value) == 0:
            raise DataProcessingError("No features selected, check your features selection")
        self._features = value

    @property
    def anomaly_quantifier(self) -> GaussianAnomalyQuantifier | None:
        return self._anomaly_quantifier

    @anomaly_quantifier.setter
    def anomaly_quantifier(self, value: GaussianAnomalyQuantifier) -> None:
        self._anomaly_quantifier = value

    @property
    def models(self) -> dict[LatentModelType, PCALatentSpace | TSNELatentSpace]:
        if (latent_models := self._latent_models) is None:
            return {}
        return latent_models

    @models.setter
    def models(self, value: dict[LatentModelType, PCALatentSpace | TSNELatentSpace]) -> None:
        self._latent_models = value
