#
from MODEL.Aff_RFDP.AffXfrm import AffXfrm
from MODEL.Asst_RFDP.OperAsst import OperAsst
from MODEL.Asst_RFDP.OperAsstNumpy import OperAsstNumpy
from MODEL.Asst_RFDP.OperAsstTorch import OperAsstTorch


class Dist2Aff(AffXfrm):
    """
    Dist2Aff class used to , inherit from AffXfrm
    2020-9-29
    """

    # def __init__(self, k_num: int, gamma: float = 0.4, asst_rela: Asst_RFDP = AsstRelaNumpy()):
    def __init__(self, k_num: int, gamma: float = 0.4, asst_rela: OperAsst = OperAsstTorch()):
        #
        super(Dist2Aff, self).__init__(asst_rela)
        # Initialize Global Variables
        self.__k_num, self.__gamma = k_num, gamma

    def _calc_oputmat(self, iput_mat):
        #
        tmp_k = self.__k_num + 1
        #
        wt_val = self.__gamma / (2 * self.__k_num)
        #
        top_vals, _ = self._asst_rela.topk_2d(iput_mat, tmp_k, rvrs=False)
        #
        expan_mat = self._asst_rela.repeat_2d(self._asst_rela.sum2lbo(top_vals[1:tmp_k, :]), self._ele_num)
        # expan_mat = self._asst_rela.repeat_2d(top_vals[1:tmp_k, :].sum(0)[:, None], self._ele_num)
        #
        delta_mat = 1 / (wt_val * (expan_mat + expan_mat.transpose(1, 0)))
        #
        sim_mat = self._asst_rela.exp(-self._asst_rela.pow(iput_mat, 2.0) * self._asst_rela.pow(delta_mat, 2.0))
        #
        return sim_mat
