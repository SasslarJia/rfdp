#
from MODEL.Sol_RFDP.ItrtSol import ItrtSol
from MODEL.Asst_RFDP.OperAsst import OperAsst
#
from RfdpLog.PrntLog_RFDP import prnt_log


class ItrtExpGrw(ItrtSol):
    """ [Iteration solution of RFDP]
    Number of items grows exponentially in the iterative process:
    X(1) = I + Pwr, Y(1) = Pwr^2; X(t+1) = [Y(t) + I]×X(t), Y(t+1) = Y(t)^2.
    2020-12-26
    """

    def __init__(self, exp_num: int = 15):
        """
        :param exp_num:
        2020-10-27
        """
        super(ItrtExpGrw, self).__init__()
        self._exp_num = exp_num

    def _get_itrtnum(self):
        """
        :return:
        2020-12-26
        """
        return self._exp_num

    def _itrt_proc(self, pwr_mul, post_mul, asst_rela: OperAsst):
        """
        [Refer to the Abstract Method]
        X(1) = I + Pwr, Y(1) = Pwr^2; X(t+1) = [Y(t) + I]×X(t), Y(t+1) = Y(t)^2.
        2020-11-15
        """
        qry_num = pwr_mul.shape[0]
        ele_num = pwr_mul.shape[1]
        mat_eye = asst_rela.repl(pwr_mul == 0, 0, asst_rela.expand_strt(asst_rela.eye(ele_num), qry_num))
        # Initialize X(1) = I + Pwr
        mat_x = mat_eye + pwr_mul
        mat_y = asst_rela.mul_3d(pwr_mul, pwr_mul)
        # X(t+1) = pwr×X(t) + post
        for itrt_idx in range(self._get_itrtnum()):
            #
            mat_x = asst_rela.mul_3d(mat_y + mat_eye, mat_x)
            #
            if itrt_idx != self._get_itrtnum() - 1:
                mat_y = asst_rela.mul_3d(mat_y, mat_y)
        #
        tmp_rslts = asst_rela.mul_3d(mat_x, post_mul)
        #
        return tmp_rslts
