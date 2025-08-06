#
from enum import Enum
import logging


class LogTypeEnum(Enum):
    """
    Log Enum for
    2021-7-5
    """
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
