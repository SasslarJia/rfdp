#
from MODEL.Sol_RFDP.ItrtSol import ItrtSol
from MODEL.Asst_RFDP.OperAsst import OperAsst
#
from RfdpLog.PrntLog_RFDP import prnt_log


class ItrtLinGrw(ItrtSol):
    """ [Iteration solution of RFDP]
    Number of items grows linearly in the iterative process:
    x(1) = init, x(t+1) = x(t) + init.
    2020-12-26
    """

    def __init__(self, lin_num: int = 2000):
        """
        :param lin_num:
        2020-10-27
        """
        super(ItrtLinGrw, self).__init__()
        self._lin_num = lin_num

    def _get_itrtnum(self):
        """
        :return:
        2020-12-26
        """
        return self._lin_num

    def _itrt_proc(self, pwr_mul, post_mul, asst_rela: OperAsst):
        """
        [Refer to the Abstract Method]
        x(1) = init; x(t+1) = x(t) + init
        2020-11-15
        """
        # Initialize X(1) = pwr×post + post
        tmp_rslts = asst_rela.mul_3d(pwr_mul, post_mul) + post_mul
        # X(t+1) = pwr × X(t) + post
        for itrt_idx in range(self._get_itrtnum()):
            #
            tmp_rslts = asst_rela.mul_3d(pwr_mul, tmp_rslts) + post_mul
            #
            if itrt_idx % 50 == 0:
                print(str(itrt_idx) + 'th iteration is finished!')
        #
        return tmp_rslts
