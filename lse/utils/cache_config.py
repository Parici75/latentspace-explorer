from .constants import REDIS_HOST, REDIS_PORT, REDIS_URI

redis_uri = f"{REDIS_URI}" if REDIS_URI is not None else f"redis://{REDIS_HOST}:{REDIS_PORT}"

CACHE_CONFIG = {
    "CACHE_TYPE": "redis",
    "CACHE_REDIS_URL": f"{redis_uri}",
}
