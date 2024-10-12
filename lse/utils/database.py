"""
Redis Database Singleton
"""

import logging
from functools import partial

import redis
from redis import Redis

from .constants import DECODE_RESPONSES, REDIS_HOST, REDIS_PORT, REDIS_URI
from .exceptions import RedisConnectionError

logger = logging.getLogger()

_REDIS_CLIENT: Redis | None = None

REDIS_KEY_EXPIRE_DELAY = 10


def init_redis_client(reset: bool = False) -> None:
    global _REDIS_CLIENT  # noqa: PLW0603
    if _REDIS_CLIENT is None or reset:
        _redis_client = partial(Redis, health_check_interval=30)
        if REDIS_URI is not None:
            _REDIS_CLIENT = _redis_client().from_url(REDIS_URI)
        else:
            _REDIS_CLIENT = _redis_client(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=DECODE_RESPONSES,
            )
        try:
            _REDIS_CLIENT.ping()
            logger.debug(f"Connection {_REDIS_CLIENT.connection_pool} OK")
        except Exception as exc:
            raise RedisConnectionError(
                "Could not connect to Redis database on host, check your server"
            ) from exc


def _check_client_initialized() -> None:
    if _REDIS_CLIENT is None:
        raise ValueError("Redis client is not initialized, call `init_redis_client()` first")


def get_redis_client() -> redis.Redis:
    _check_client_initialized()
    return _REDIS_CLIENT  # type: ignore
