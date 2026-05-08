from MODEL.Asst_RFDP.OperAsst import OperAsst
from MODEL.Asst_RFDP.OperAsstTorch import OperAsstTorch


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
        cc_ids = [-1] * smpl_num
        cur_cc = -1
        for ele_idx in range(smpl_num):
            if cc_ids[ele_idx] != -1:
                continue
            cur_cc += 1
            stack = [ele_idx]
            cc_ids[ele_idx] = cur_cc
            while len(stack) > 0:
                strt_idx = stack.pop()
                for end_idx in edges_lst[strt_idx]:
                    if cc_ids[end_idx] == -1:
                        cc_ids[end_idx] = cur_cc
                        stack.append(end_idx)
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
