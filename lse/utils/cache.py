"""
Flask Cache manager singleton pattern
"""

import logging
from collections.abc import Callable
from typing import Any, Self

import dash
from flask_caching import Cache

from lse.utils.cache_config import CACHE_CONFIG

logger = logging.getLogger()


class FlaskCacheManager:
    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self.cache: Cache | None = None
        self.memoized_functions: dict[str, Callable[..., Any]] = {}

    def init_app_cache(self, app: dash.Dash) -> None:
        if self.cache is None:
            logger.debug(f"Initializing cache for {app.__class__.__name__} app...")
            self.cache = Cache(app.server, config=CACHE_CONFIG)
        logger.info("Cache is already initialized.")

    def memoize_function(self, fct: Callable[..., Any]) -> Callable[..., Any]:
        if (fct_name := fct.__name__) not in self.memoized_functions:
            logger.debug(f"Setting memoized cache for function {fct_name}")
            self.memoized_functions[fct_name] = self.cache.memoize()(fct)  # type: ignore

        return self.memoized_functions[fct_name]

    def flush_cache(self, *fcts: Callable[..., Any]) -> None:
        for fct in fcts:
            if (fct_name := fct.__name__) in self.memoized_functions:
                logger.info(f"Flushing memoize cache for function {fct_name}")
                self.cache.delete_memoized(self.memoized_functions[fct_name])  # type: ignore
            else:
                logger.error(f"No memoized cache found for function {fct_name}")
