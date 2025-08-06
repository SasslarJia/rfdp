#
from abc import ABCMeta
from abc import abstractmethod as ABCFctn
from MODEL.Fit_RFDP.FitCstr import FitCstr
from MODEL.Asst_RFDP.OperAsst import OperAsst


class HyFC(FitCstr):
    """ ‖X-A‖^2 + |Y-∞|, where ‖•‖ is L2 norm and |•| is L1 norm
    2020-10-28
    """

    def __init__(self, fctr_parm):
        """
        [Refer to the Abstract Method]
        2020-10-27
        """
        super(HyFC, self).__init__(fctr_parm)

    def _xfrm_iput(self, sub_w, asst_rela: OperAsst):
        """
        [Refer to the Abstract Method]
        2020-10-27
        """
        return sub_w

    # <editor-fold desc="Check whether the learned result is similarity">
    def _check_sim(self) -> bool:
        """
        [Refer to the Abstract Method]
        :return:
        2020-12-20
        """
        return False
    # </editor-fold>

    def _get_fit(self, sub_w, qries_insub: list, incl_qry: bool, asst_rela: OperAsst):
        """
        [Refer to the Abstract Method]
        2020-10-29
        """
        # Introduce temporary variable
        smpl_num = sub_w.shape[0]
        qry_num = len(qries_insub)
        # Diagonal elements in factor matrix is either 1 or 0,
        # where 1 for those belong to kNN of the query and 0 for the rests.
        coef_cols = asst_rela.zeros([qry_num, smpl_num, 1])
        # Initialize constant matrix
        cnst_cols = asst_rela.zeros([qry_num, smpl_num, 1])
        # Update each column in 'coef_cols' and 'cnst_cols'
        for ele_idx in range(qry_num):
            # Get indexes of query
            qry = qries_insub[ele_idx]
            # Sum the rows with query indexes to one row and normalize the result
            e_row = asst_rela.sum2lbo(sub_w[qry, :]) / len(qry)
            # Set the values of each column in coefficient matrix
            coef_cols[ele_idx, :, 0] = self._get_coef(qry, e_row, incl_qry, asst_rela)
            # Set the values of each column in constant matrix
            cnst_cols[ele_idx, :, 0] = self._get_cnst(qry, e_row, incl_qry, asst_rela)

            # # Set the values of each column in constant
            # coef_cols[ele_idx, :, 0] = asst_rela.repl(e_row > 0, e_col, 0.0)
            # # Zero setting for the elements related to the query
            # if not incl_qry:
            #     coef_cols[ele_idx, qry, 0] = 0
            # #
            # cnst_cols[ele_idx, qry, 0] = 0
        #
        return coef_cols, cnst_cols

    @ABCFctn
    def _get_coef(self, qry: list, e_row, incl_qry: bool, asst_rela: OperAsst):
        """ Get coefficient column
        :param qry:
        :param e_row:
        :param incl_qry:
        :param asst_rela:
        :return:
        """
        pass

    @ABCFctn
    def _get_cnst(self, qry: list, e_row, incl_qry: bool, asst_rela: OperAsst):
        """ Get constant column
        :param qry:
        :param e_row:
        :param incl_qry:
        :param asst_rela:
        :return:
        """
        pass
