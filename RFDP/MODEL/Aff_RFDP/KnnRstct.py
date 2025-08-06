#
from MODEL.Enum_RFDP.KnnEnum import KnnEnum
from MODEL.Aff_RFDP.AffXfrm import AffXfrm
from MODEL.Asst_RFDP.OperAsst import OperAsst
from MODEL.Asst_RFDP.OperAsstTorch import OperAsstTorch
from RfdpLog.PrntLog_RFDP import prnt_log


class KnnRstct(AffXfrm):
    """
    KnnRstct class used to generate KNN matrix, inherit from AffXfrmTorch
    2020-9-29
    """

    # def __init__(self, k_num: int, knn_rule: KnnRule = KnnRule.MUTUAL, asst_rela: Asst_RFDP = AsstRelaNumpy()):
    def __init__(self, k_num: int, knn_rule: KnnEnum = KnnEnum.MUTUAL, asst_rela: OperAsst = OperAsstTorch()):
        #
        super(KnnRstct, self).__init__(asst_rela)
        #
        self.__k_num = k_num
        #
        self.__knn_rule = knn_rule

    def _calc_oputmat(self, iput_mat):
        #
        if self.__k_num > self._ele_num:
            prnt_log.error('K value is greater than the matrix dim!')
        #
        k_num = self.__k_num + 1  # Without self: BUG in torch: top_vals[:, k_num]
        #
        top_vals, _ = self._asst_rela.topk_2d(iput_mat, k_num, rvrs=True)
        kth_vals = top_vals[k_num - 1, :][:, None]
        #
        fir_flg = iput_mat >= kth_vals
        flg_mat = self._asst_rela.cvrt2float(fir_flg * fir_flg.transpose(1, 0))
        #
        if self.__knn_rule == KnnEnum.MUTUAL:
            return flg_mat * iput_mat
        else:
            return flg_mat * iput_mat
