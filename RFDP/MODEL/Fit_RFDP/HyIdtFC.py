#
from MODEL.Fit_RFDP.HyFC import HyFC
from MODEL.Asst_RFDP.OperAsst import OperAsst


class HyIdtFC(HyFC):
    """ L2 terms with identical weights in Hybrid fitting constraint
    2021-6-28
    """

    def __init__(self, fctr_parm: float = 0.1):
        """
        [Refer to the Abstract Method]
        2020-10-27
        """
        super(HyIdtFC, self).__init__(fctr_parm)

    def _get_coef(self, qry: list, e_row, incl_qry: bool, asst_rela: OperAsst):
        """
        :param qry:
        :param e_row:
        :param incl_qry:
        :param asst_rela:
        :return:
        """
        # Init coefficient
        tmp_coef = asst_rela.repl(e_row > 0, 1, 0)
        #
        if not incl_qry:
            tmp_coef[0, qry] = 0
        #
        return tmp_coef

    def _get_cnst(self, qry: list, e_row, incl_qry: bool, asst_rela: OperAsst):
        """
        :param qry:
        :param e_row:
        :param incl_qry:
        :param asst_rela:
        :return:
        """
        # Remove self-similarity
        tmp_sum = e_row.sum() - e_row[0, qry].sum()
        #
        tmp_cnst = asst_rela.repl(e_row > 0, tmp_sum / e_row, 1)
        #
        if not incl_qry:
            tmp_cnst[0, qry] = 0
        #
        return tmp_cnst
