#
import matplotlib.pyplot as plt
from RfdpLog.PrntLog_RFDP import prnt_log


class EvalPrcsRcal(object):
    """
    Plot Precision-Recall curve
    2020-12-13
    """

    @staticmethod
    def plot(rslt_lst: list, ref_lst: list, dply_str: str):
        """
        :param rslt_lst:
        :param ref_lst:
        :param dply_str:
        :return:
        """
        #
        prcs_lst, rcal_lst = EvalPrcsRcal.__eval(rslt_lst, ref_lst)
        #
        plt.plot(rcal_lst, prcs_lst, dply_str)

    @staticmethod
    def dply(x_max: float = None, x_min: float = None, y_max: float = None, y_min: float = None):
        #
        plt.xlim(xmax=x_max, xmin=x_min)
        plt.ylim(ymax=y_max, ymin=y_min)
        #
        plt.show()

    @staticmethod
    def __eval(rslt_lst: list, ref_lst: list) -> (list, list):
        """
        :param rslt_lst:
        :param ref_lst:
        :return:
        2020-12-23
        """
        #
        if not len(rslt_lst) == len(ref_lst):
            prnt_log("The lengths of result-list and reference-list are not same!")
        #
        grp_num = len(rslt_lst)
        #
        corr_lst = list()
        tot_lst = list()
        for grp_idx in range(grp_num):
            #
            tmp_rslt = rslt_lst[grp_idx]
            tmp_ref = ref_lst[grp_idx].rslt_ids
            #
            if not len(tmp_rslt) == len(tmp_ref):
                prnt_log("The lengths of the " + str(grp_idx) + "th result and the "
                         + str(grp_idx) + "th reference are not same!")
            #
            if len(corr_lst) < len(tmp_rslt):
                corr_lst.extend([0.0] * (len(tmp_rslt) - len(corr_lst)))
            for rslt_idx in range(len(tmp_rslt)):
                if tmp_rslt[rslt_idx] in tmp_ref:
                    corr_lst[rslt_idx] += 1.0
            #
            if len(tot_lst) < len(tmp_rslt):
                tot_lst.extend([0.0] * (len(tmp_rslt) - len(tot_lst)))
            for rslt_idx in range(len(tmp_rslt)):
                tot_lst[rslt_idx] += 1.0
        #
        tot_num = sum(tot_lst)
        prcs_lst = [sum(corr_lst[:idx + 1]) / sum(tot_lst[:idx + 1]) for idx in range(len(corr_lst))]
        rcal_lst = [sum(corr_lst[:idx + 1]) / tot_num for idx in range(len(corr_lst))]
        #
        return prcs_lst, rcal_lst
