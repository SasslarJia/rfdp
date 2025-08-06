#
from enum import Enum
from RfdpLog.PrntLog_RFDP import prnt_log


#
class DimEnum(Enum):
    """
    Dimension of Operator
    """
    Dim1 = 1
    Dim2 = 2
    Dim3 = 3

    @staticmethod
    def get_dim(iput):
        if len(iput.shape) == 1:
            return DimEnum.Dim1
        elif len(iput.shape) == 2:
            return DimEnum.Dim2
        elif len(iput.shape) == 3:
            return DimEnum.Dim3
        else:
            prnt_log("The shape of input Matrix does not meet the request of RFDP!")
            return None

    def chk_dim(self, iput) -> bool:
        return len(iput.shape) == self.value
