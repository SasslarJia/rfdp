#
from MODEL.Sol_RFDP.SolProc import SolProc
from MODEL.Asst_RFDP.OperAsst import OperAsst


class AdpSol(SolProc):
    """
    2020-12-11
    """

    def __init__(self, asst_rela: OperAsst):
        """ Select the solution process adaptively
        :param asst_rela: Omit
        2020-10-27
        """
        super(SolProc, self).__init__(asst_rela)

    def _clac_rslt(self, d_cols, w_mats, coef_cols, cnst_cols, fctr_parm: float,
                   is_sim: bool, asst_rela: OperAsst):
        """
        :param d_cols:
        :param w_mats:
        :param coef_cols:
        :param cnst_cols:
        :param fctr_parm:
        :param is_sim:
        :param asst_rela:
        :return:
        2020-12-13
        """
        pass