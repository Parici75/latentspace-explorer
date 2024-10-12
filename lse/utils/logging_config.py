import logging
import logging.config

LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s: %(message)s"


def configure_logging(log_level: int = logging.INFO) -> None:
    logging.basicConfig(level=log_level, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
