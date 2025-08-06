#
import copy
#
from MODEL.Asst_RFDP.ProcAsst import ProcAsst
from MODEL.Frmwk_RFDP.RfdpSuplExtn import RfdpSuplExtn


class ItrtQryExtn(object):
    """ Iterative Query Extension Based on RFDP
    Re-ranking by iteratively extend each query
    2021-5-26
    """

    def __init__(self, rfdp: RfdpSuplExtn, rerank_num: int):
        """
        Iterative Query Extension Based on RFDP
        :param rfdp: Input RFDP with supplement extension
        :param rerank_num: Number of samples expected to be re-ranked
        2021-5-26
        """
        self._rfdp = rfdp
        self._rerank_num = rerank_num

    def __call__(self, iput_qries: list, step_size: int = 5) -> list:
        """
        Iteratively Query Extension by RFDP
        :param iput_qries: List with input queries
        :param step_size: Step size for each iteration
        2021-5-26
        """
        # Check the length of each query
        if not ProcAsst.meas_qry(iput_qries, self._rerank_num):
            return None
        # Copy the list of input queries
        oput_lst = copy.deepcopy(iput_qries)
        idx_lst = list(range(len(iput_qries)))
        #
        while len(idx_lst) > 0:
            # Construct temporary query list
            strt_qries = [oput_lst[qry_idx] for qry_idx in idx_lst]
            # Get temporary re-ranking result by RFDP
            tmp_rslts = self._rfdp(strt_qries, step_size)
            #
            for idx in range(len(idx_lst) - 1, -1, -1):
                #
                if len(tmp_rslts[idx]) >= self._rerank_num:
                    oput_lst[idx_lst[idx]] = tmp_rslts[idx][:self._rerank_num]
                    idx_lst.pop(idx)
                else:
                    oput_lst[idx_lst[idx]] = tmp_rslts[idx]
        #
        return oput_lst

    @staticmethod
    def _chk_qries(iput_qries: list, rerank_num: int) -> list:
        """ Check whether the length of query is less than the reranking number
        :param iput_qries: Input query list
        :param rerank_num: Number of elements expected to be reranked
        :return: Output list
        2021-7-9
        """
        #
        idx_lst = list()
        # Check length of each query and add the insufficient ones to list
        for qry_idx in range(len(iput_qries)):
            if len(iput_qries[qry_idx]) < rerank_num:
                idx_lst.append(qry_idx)
        #
        return idx_lst
