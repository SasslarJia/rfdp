#
from MODEL.Smth_RFDP.SmthCstr import SmthCstr
from MODEL.Asst_RFDP.OperAsst import OperAsst


class InclangSC(SmthCstr):
    """ Smoothness constraint reflects 'included angle'
    2021-2-24
    """

    def __init__(self, is_norm: bool = False, slp_fctr: float = 2000.0):
        """
        :param is_norm:
        2021-2-24
        """
        #
        super(InclangSC, self).__init__(is_norm)
        #
        self._slp_fctr = slp_fctr

    def _xfrm_iput(self, sub_w, asst_rela: OperAsst):
        """
        [Refer to the Abstract Method]
        2021-2-24
        """
        #
        return sub_w

    def _get_lapl(self, iput_w, qries_insub: list, incl_qry: bool, asst_rela: OperAsst):
        """
        [Refer to the Abstract Method]
        2021-2-25
        """
        #
        smpl_num = iput_w.shape[0]
        qry_num = len(qries_insub)
        #
        tmp_mat = asst_rela.zeros([qry_num, smpl_num])
        #
        for qry_idx in range(qry_num):
            # Get indexes of query
            tmp_mat[qry_idx, qries_insub[qry_idx]] = 1
        #
        w_cols = asst_rela.mul_3d(tmp_mat, iput_w) * self._slp_fctr
        dnmtr_cols = 1 / asst_rela.pow(asst_rela.pow(w_cols, 2) + 1, 0.5)
        #
        nmrtr_mats = asst_rela.mul_3d(asst_rela.expand_dim(w_cols, 2), asst_rela.expand_dim(w_cols, 1)) + iput_w
        dnmtr_mats = asst_rela.mul_3d(asst_rela.expand_dim(dnmtr_cols, 2), asst_rela.expand_dim(dnmtr_cols, 1))
        #
        return nmrtr_mats * dnmtr_mats
