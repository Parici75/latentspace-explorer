import fakeredis
import numpy as np
import pandas as pd
import pytest
from mltoolbox.anomaly_detection import GaussianAnomalyQuantifier
from pandas.api.types import is_numeric_dtype

from lse.libs.backend import AppBackend, process_and_cache_data
from lse.libs.backend.io import OfflineUserSession, OnlineUserSession
from lse.libs.backend.io._data_processing import _process_data
from lse.libs.backend.io._online import OnlineDataWrapper, RedisKeyManager
from lse.libs.backend.io.data_models import UserSessionParameters
from lse.libs.data_models import VariableReference

SAMPLE_DATA = pd.DataFrame(
    {"Feature1": [1, 2, 3], "Feature2": [4, 5, 6], 3: [7, 8, 9], "id": ["a", "b", "c"]}
)
SAMPLE_DATA_WITH_NAN = pd.concat(
    (
        SAMPLE_DATA,
        pd.DataFrame.from_records([{key: np.nan for key in SAMPLE_DATA.columns}]),
    )
)
SESSION_ID = "aa-bb"
MOCK_USER_DATA = "data_from_cache"


@pytest.fixture()
def mock_data_wrapper(mocker):
    mocker.patch(
        "lse.libs.backend.io._online.OnlineDataWrapper.data",
        return_value=_process_data(SAMPLE_DATA),
        new_callable=mocker.PropertyMock,
    )
    return mocker


@pytest.fixture()
def mock_nan_data_wrapper(mocker):
    mocker.patch(
        "lse.libs.backend.io._online.OnlineDataWrapper.data",
        return_value=_process_data(SAMPLE_DATA_WITH_NAN),
        new_callable=mocker.PropertyMock,
    )
    return mocker


@pytest.fixture(autouse=True)
def mock_redis_database(mocker):
    mocker.patch("lse.libs.backend.io._online.get_redis_client", return_value=fakeredis.FakeRedis())
    return mocker


@pytest.fixture()
def mock_user_session_cached_key(mocker):
    mocker.patch(
        "lse.libs.backend.io._online.RedisKeyManager.cached_data",
        return_value=MOCK_USER_DATA,
        new_callable=mocker.PropertyMock,
    )
    return mocker


def test_data_processing():
    processed_data = _process_data(SAMPLE_DATA)
    assert "Feature_3" in processed_data.columns
    assert VariableReference.UNIQUE_ID in processed_data.index.names


def test_data_caching(mock_redis_database):
    process_and_cache_data(session_id=SESSION_ID, data=SAMPLE_DATA)


class TestOnlineDataWrapper:
    def test_data_processing(self, mock_data_wrapper):
        data_wrapper = OnlineDataWrapper(session_id=SESSION_ID)
        assert (data_wrapper.data.values == SAMPLE_DATA.values).all()

    def test_numerical_identifiers_property(self, mock_data_wrapper):
        processed_data = _process_data(SAMPLE_DATA)
        data_wrapper = OnlineDataWrapper(session_id=SESSION_ID)
        assert data_wrapper.numerical_identifiers == [
            col for col in processed_data.columns if is_numeric_dtype(processed_data[col])
        ] + [VariableReference.UNIQUE_ID]


class TestUserDataManager:
    def test_redis_key(self):
        with pytest.raises(ValueError):
            user_data_manager = RedisKeyManager(session_id=SESSION_ID, key_name="test")

        user_data_manager = RedisKeyManager(session_id=SESSION_ID, key_name="features")
        assert user_data_manager.redis_key == "_".join((SESSION_ID, "user", "features"))

    def test_cached_data(self, mock_user_session_cached_key):
        assert (
            RedisKeyManager(session_id=SESSION_ID, key_name="features").cached_data
            == MOCK_USER_DATA
        )

    def test_cache_data(self):
        mock_data = "mock_data"
        user_data_manager = RedisKeyManager(session_id=SESSION_ID, key_name="features")
        user_data_manager.cache_data(mock_data)
        assert user_data_manager.cached_data == mock_data


class TestOnlineUserSession:

    def test_features(self, mock_data_wrapper):
        user_session = OnlineUserSession(session_id=SESSION_ID)
        assert "id" not in user_session.features

    def test_filter_property(self, mock_data_wrapper):
        user_session = OnlineUserSession(session_id=SESSION_ID)
        assert (
            user_session.filtered_data.values == SAMPLE_DATA.values
        ).all()  # It should be the same as sample_data initially
        # Test the filtered_data property after setting data_infilter
        user_session.data_infilter = _process_data(SAMPLE_DATA).index[
            1:
        ]  # Filter out the first row
        assert (user_session.filtered_data.values == SAMPLE_DATA.iloc[1:].values).all()

    def test_nan_processing(self, mock_nan_data_wrapper):
        user_session = OnlineUserSession(session_id=SESSION_ID)
        assert len(user_session.data_wrapper.data) == len(SAMPLE_DATA_WITH_NAN)
        assert len(user_session.pca_data) == len(SAMPLE_DATA)


class TestOfflineUserSession:

    def test_user_session_parameter(self):
        user_session_parameters = UserSessionParameters(
            latent_models={},
            data_infilter=pd.Index([], name="index"),
            features=["Feature_1", "Feature_2"],
            anomaly_quantifier=GaussianAnomalyQuantifier.initialize(),
        )
        assert user_session_parameters.latent_models == {}
        assert (user_session_parameters.data_infilter == pd.Index([], name="index")).all()
        assert user_session_parameters.features == ["Feature_1", "Feature_2"]
        assert isinstance(user_session_parameters.anomaly_quantifier, GaussianAnomalyQuantifier)

        with pytest.raises(ValueError):
            UserSessionParameters(
                latent_models=None,
            )

    def test_init(self):
        user_session = OfflineUserSession(
            session_id=SESSION_ID,
            user_session_parameters=UserSessionParameters(
                latent_models={},
                data_infilter=pd.Index([], name="index"),
                features=["Feature_1", "Feature_2"],
                anomaly_quantifier=GaussianAnomalyQuantifier.initialize(),
            ),
        )
        assert user_session.models == {}


class TestAppBackend:
    def test_gini(self):
        x_pure = np.array([1, 1, 1])
        assert AppBackend.gini(x_pure) == 0

        x_unpure = np.array([1, 2, 3])
        expected_results = 1 - np.sum(3 * (1 / 3) ** 2)
        assert AppBackend.gini(x_unpure) == expected_results

        x_halfpure = np.array([1, 1, 2, 2])
        expected_results = 1 - np.sum(2 * ((1 / 2) ** 2))
        assert AppBackend.gini(x_halfpure) == expected_results
