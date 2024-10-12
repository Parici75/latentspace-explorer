import os

from dotenv import load_dotenv

from .strtobool import strtobool

load_dotenv(override=False)

MODEL_EXPORT_DIRECTORY = os.environ.get("MODEL_EXPORT_PATH", os.environ["TMPDIR"])
if MODEL_EXPORT_DIRECTORY is None:
    raise FileNotFoundError(
        "Could not find the path of temporary directory, check your environment variable configuration"
    )

ASSETS_PATH = os.environ.get("ASSETS_PATH", "./libs/assets")

# Redis Cache
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = os.environ.get("REDIS_PORT", 6379)
REDIS_URI = os.environ.get("REDIS_URI", None)
DECODE_RESPONSES = strtobool(os.environ.get("DECODE_RESPONSES", False))
