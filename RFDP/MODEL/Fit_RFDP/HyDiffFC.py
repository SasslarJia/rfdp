#
from MODEL.Fit_RFDP.HyFC import HyFC
from MODEL.Asst_RFDP.OperAsst import OperAsst


class HyDiffFC(HyFC):
    """ L2 terms with different weights in Hybrid fitting constraint
    2021-6-28
    """

    def __init__(self, fctr_parm: float = 0.1):
        """
        [Refer to the Abstract Method]
        2020-10-27
        """
        super(HyDiffFC, self).__init__(fctr_parm)

    def _get_coef(self, qry: list, e_row, incl_qry: bool, asst_rela: OperAsst):
        """
        :param qry:
        :param e_row:
        :param incl_qry:
        :param asst_rela:
        :return:
        """
        # Remove self-similarity
        tmp_sum = e_row.sum() - e_row[0, qry].sum()
        # Init coefficient
        tmp_coef = e_row / (tmp_sum if tmp_sum > 0 else 1)
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
        #
        tmp_cnst = asst_rela.ones(e_row.shape)
        #
        if not incl_qry:
            tmp_cnst[0, qry] = 0
        #
        return tmp_cnst
