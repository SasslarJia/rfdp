#
from enum import Enum


class AjcnEnum(Enum):
    """
    Adjacent rule for searching the neighborhood
    2020-9-29
    """
    Grt = 0
    GrtEq = 1
    Sml = 2
    SmlEq = 3
