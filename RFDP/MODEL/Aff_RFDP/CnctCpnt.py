#
import sys
#
from MODEL.Asst_RFDP.OperAsst import OperAsst
from MODEL.Asst_RFDP.OperAsstTorch import OperAsstTorch

sys.setrecursionlimit(10000)


class CnctCpnt(object):
    """
    Class used ot generate connect components
    2020-2-17
    """

    def __init__(self, asst_rela: OperAsst = OperAsstTorch()):
        #
        self._asst_rela = asst_rela

    def __call__(self, edges_lst: list, smpl_num: int) -> (list, list):
        """ Execute functions
        :param edges_lst:
        :return:
        2020-2-17
        """
        # Get vector with connected component indexes
        return CnctCpnt.__cnct_cpnts(edges_lst, smpl_num)

    @staticmethod
    def __cnct_cpnts(edges_lst: list, smpl_num: int) -> list:
        """ Label elements with indexes of connected components
        :param edges_lst: List with edges
        :param smpl_num:
        :return: 1) Final connected component indexes for different elements
                 2) List of elements in different connected components
        2020-10-6
        """
        # Initialize connected component index vector
        cc_ids = [-1] * smpl_num
        # Current connected component index
        cur_cc = -1
        # Depth-first search
        for ele_idx in range(smpl_num):
            # Whether eleId is labeled
            if cc_ids[ele_idx] == -1:
                # Initialize current connected component index
                cur_cc = cur_cc + 1
                cc_ids[ele_idx] = cur_cc
                # Label element with connected component index
                cc_ids = CnctCpnt.__recu_cc(cc_ids, edges_lst, ele_idx, cur_cc)
        #
        return cc_ids

    @staticmethod
    def __recu_cc(cc_ids, top_lst, strt_idx, cur_cc) -> list:
        """ Recursion function for deep-first traversal
        :param cc_ids: Initial connected component indexes for different elements
        :param strt_idx: Reference element
        :param top_lst: List with tops
        :param cur_cc: Current connected component index
        :return: Updated connected component indexes for different elements
        """
        #
        for end_idx in top_lst[strt_idx]:
            if cc_ids[end_idx] == -1:
                cc_ids[end_idx] = cur_cc
                cc_ids = CnctCpnt.__recu_cc(cc_ids, top_lst, end_idx, cur_cc)
        #
        return cc_ids

    # @staticmethod
    # def __cnct_lst(top_ids, cnct_mat, asst_rela: Asst_RFDP) -> list:
    #     #
    #     top_lst = list()
    #     #
    #     for idx in range(top_ids.shape[0]):
    #         tmp_flg = cnct_mat[idx, top_ids[idx, :]] > 0
    #         tmp_lst = list(set(asst_rela.cvrt2lst(top_ids[idx, tmp_flg])))
    #         tmp_lst.remove(idx)
    #         top_lst.append(tmp_lst)
    #     #
    #     return top_lst

