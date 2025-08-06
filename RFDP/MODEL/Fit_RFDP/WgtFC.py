#
from MODEL.Fit_RFDP.FitCstr import FitCstr
from MODEL.Asst_RFDP.OperAsst import OperAsst


class WgtFC(FitCstr):
    """ ‖X-Y‖^2 with column-vector of W as predefined values, where ‖•‖ is L2 norm
    2020-10-28
    """

    def __init__(self, fctr_parm: float = 1.0):
        """
        :param fctr_parm:
        2020-10-27
        """
        super(WgtFC, self).__init__(fctr_parm)

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
        # Introduce temporary variable
        smpl_num = sub_w.shape[0]
        qry_num = len(qries_insub)
        # Factor matrix is always I for each query, expressed as column-vector [1,...,1]'.
        coef_cols = asst_rela.ones([1 if incl_qry else qry_num, smpl_num, 1])
        # Get constant expressed as columns
        cnst_cols = asst_rela.zeros([qry_num, smpl_num, 1])
        # Update each column in 'cnst_cols' and 'coef_cols' if needed
        for ele_idx in range(qry_num):
            # Get indexes of query
            qry = qries_insub[ele_idx]
            # Zero setting for the elements related to the query
            if not incl_qry:
                coef_cols[ele_idx, qry, 0] = 0
            # Update the values in each column of cnst_cols with the average weight
            cnst_cols[ele_idx, :, 0] = asst_rela.sum2last(sub_w[:, qry], False) / qry_num
            cnst_cols[ele_idx, qry, 0] = 1  # 1 if incl_qry else 0
        #
        return coef_cols, cnst_cols
        # </editor-fold>
