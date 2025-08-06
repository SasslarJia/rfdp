#
from MODEL.Fit_RFDP.FitCstr import FitCstr
from MODEL.Asst_RFDP.OperAsst import OperAsst
from RfdpLog.PrntLog_RFDP import prnt_log


class OazFC(FitCstr):
    """ ‖X-Y‖^2 with 1-0 (One-and-zero, Oaz) as predefined values, where ‖•‖ is L2 norm
    2020-10-28
    """

    def __init__(self, fctr_parm: float = 1.0):
        """
        [Refer to the Abstract Method]
        2020-10-27
        """
        super(OazFC, self).__init__(fctr_parm)

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
        return True
    # </editor-fold>

    def _get_fit(self, sub_w, qries_insub: list, incl_qry: bool, asst_rela: OperAsst):
        """
        [Refer to the Abstract Method]
        2020-10-29
        """
        #
        if not incl_qry:
            prnt_log("Without self-similarity/self-distance, One-and-zero fitting constraint cannot carried out!")
            return None
        # Introduce temporary variable
        smpl_num = sub_w.shape[0]
        qry_num = len(qries_insub)
        # Factor matrix is always I for each query, expressed as column-vector [1,...,1]'.
        coef_cols = asst_rela.ones([1, smpl_num, 1])
        # Get constant expressed as columns
        cnst_cols = asst_rela.zeros([qry_num, smpl_num, 1])
        # Replace with the value of 1 at the position of input query
        for ele_idx in range(qry_num):
            cnst_cols[ele_idx, qries_insub[ele_idx], 0] = 1
        #
        return coef_cols, cnst_cols
