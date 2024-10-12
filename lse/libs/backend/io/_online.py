import io
import logging
from abc import ABCMeta, abstractmethod
from functools import cached_property, partial
from typing import Any

import dill as pickle
import pandas as pd
from mltoolbox.anomaly_detection import GaussianAnomalyQuantifier
from mltoolbox.latent_space import PCALatentSpace, TSNELatentSpace

from lse.libs.data_models import LatentModelType
from lse.libs.exceptions import DataProcessingError
from lse.utils.database import get_redis_client

from ._data_processing import _process_data
from .data_models import (
    DataWrapper,
    RedisKeyGroupType,
    RedisKeyType,
    Session,
    UserSession,
)
from .utils import FEATHER_COMPRESSION

REDIS_KEY_EXPIRE_TIME = 3600


logger = logging.getLogger()


class RedisStorage(Session, metaclass=ABCMeta):
    _REDIS_SUFFIX: RedisKeyGroupType

    def get_redis_key_base(self) -> str:
        return "_".join((self.session_id, self._REDIS_SUFFIX))

    @property
    @abstractmethod
    def redis_key(self) -> Any: ...

    def get_key_value(self) -> Any:
        return get_redis_client().get(self.redis_key)


class RedisKeyManager(RedisStorage):
    _REDIS_SUFFIX = RedisKeyGroupType.USER

    key_name: RedisKeyType

    @property
    def redis_key(self) -> str:
        return "_".join((self.get_redis_key_base(), self.key_name))

    @property
    def cached_data(self) -> Any | None:
        if (buffer := self.get_key_value()) is not None:
            return pickle.loads(buffer)

        logger.debug(f"No session data for {self.redis_key} found in database")
        return None

    def cache_data(self, value: Any) -> None:
        get_redis_client().set(self.redis_key, pickle.dumps(value), ex=REDIS_KEY_EXPIRE_TIME)


class OnlineDataWrapper(DataWrapper, RedisStorage):
    """A class to store and retrieve data in/from a Redis Database."""

    _REDIS_SUFFIX = RedisKeyGroupType.DATA

    def __init__(self, session_id: str) -> None:
        super().__init__(
            session_id=session_id,
        )

    @property
    def redis_key(self) -> str:
        return self.get_redis_key_base()

    @cached_property
    def data(self) -> pd.DataFrame:
        return pd.read_feather(self.get_data_buffer())

    def get_data_buffer(self) -> io.BytesIO:
        return io.BytesIO(self.get_key_value())

    def cache_data(self, data: pd.DataFrame) -> None:
        with io.BytesIO() as buffer:
            data.to_feather(buffer, compression=FEATHER_COMPRESSION)
            buffer.seek(0)
            get_redis_client().set(self.redis_key, buffer.read(), ex=REDIS_KEY_EXPIRE_TIME)


class OnlineUserSession(UserSession):
    """A class to store and retrieve user session data in/from a Redis Database."""

    data_wrapper: OnlineDataWrapper
    key_manager: dict[RedisKeyType, RedisKeyManager]

    def __init__(self, session_id: str) -> None:
        super().__init__(
            session_id=session_id,
            data_wrapper=OnlineDataWrapper(session_id=session_id),
            key_manager=self._get_key_manager(session_id=session_id),
        )

    def _get_key_manager(self, session_id: str) -> dict[RedisKeyType, RedisKeyManager]:
        partial_key_manager = partial(RedisKeyManager, session_id=session_id)
        return {member: partial_key_manager(key_name=member.value) for member in RedisKeyType}

    def reset(self, key_type: RedisKeyGroupType) -> None:
        cursor = 0
        while True:
            cursor, keys = get_redis_client().scan(cursor=cursor, match=f"{'_'.join((self.session_id, key_type))}*")  # type: ignore
            if keys:
                logger.debug(f"Removing the following keys from the cache database:\n{keys}")
                get_redis_client().delete(*keys)
            if cursor == 0:  # no more keys to delete
                break

    @property
    def data_infilter(self) -> pd.Index | None:
        return self.key_manager[RedisKeyType.DATA_INFILTER].cached_data

    @data_infilter.setter
    def data_infilter(self, value: pd.Index) -> None:
        self.key_manager[RedisKeyType.DATA_INFILTER].cache_data(value)

    @property
    def features(self) -> list[str]:
        if (features := self.key_manager[RedisKeyType.FEATURES].cached_data) is None:
            return self.data_wrapper.numerical_features
        return features

    @features.setter
    def features(self, value: list[str]) -> None:
        if len(value) == 0:
            raise DataProcessingError("No features selected, check your features selection")
        self.key_manager[RedisKeyType.FEATURES].cache_data(value)

    @property
    def anomaly_quantifier(self) -> GaussianAnomalyQuantifier | None:
        return self.key_manager[RedisKeyType.ANOMALY_QUANTIFIER].cached_data

    @anomaly_quantifier.setter
    def anomaly_quantifier(self, value: GaussianAnomalyQuantifier) -> None:
        self.key_manager[RedisKeyType.ANOMALY_QUANTIFIER].cache_data(value)

    @property
    def models(self) -> dict[LatentModelType, PCALatentSpace | TSNELatentSpace]:
        if (latent_models := self.key_manager[RedisKeyType.LATENT_MODELS].cached_data) is None:
            return {}
        return latent_models

    @models.setter
    def models(self, value: dict[LatentModelType, PCALatentSpace | TSNELatentSpace]) -> None:
        self.key_manager[RedisKeyType.LATENT_MODELS].cache_data(value)


def process_and_cache_data(session_id: str, data: pd.DataFrame) -> None:
    """Processes and cache the data."""
    user_session = OnlineUserSession(session_id=session_id)
    user_session.reset(key_type=RedisKeyGroupType.USER)
    user_session.data_wrapper.cache_data(_process_data(data))
