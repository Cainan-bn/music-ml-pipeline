
import logging
import sys
from typing import Any


class StructuredFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:
            base = (
            f"timestamp={self.formatTime(record, self.datefmt)} "
            f"level={record.levelname} "
            f"module={record.module} "
            f"message=\"{record.getMessage()}\""
        )
        if record.exc_info:
            base += f" exception=\"{self.formatException(record.exc_info)}\""
        return base


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)