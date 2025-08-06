#
from MODEL.Iput_RFDP.Qry2Blk import Qry2Blk
from MODEL.Iput_RFDP.MdlIput import MdlIput
from MODEL.Asst_RFDP.OperAsst import OperAsst
from MODEL.Asst_RFDP.OperAsstTorch import OperAsstTorch
from RfdpLog.PrntLog_RFDP import prnt_log


class IputWhl(MdlIput):
    """
    Diffusion process on the whole connected component(s)
    2020-12-29
    """

    def __init__(self, smth_w, fit_w=None, full_w=None, asst_rela: OperAsst = OperAsstTorch()):
        """
        :param smth_w:
        :param fit_w:
        :param full_w:
        :param asst_rela:
        2020-12-29
        """
        super(IputWhl, self).__init__(smth_w, fit_w, full_w, asst_rela)

    def _stl_qries(self, qry_lst: list) -> list:
        """
        [Refer to the Abstract Method]
        2020-10-9
        """
        # create the list of Qry2Blk
        q2b_list = list()
        #
        for qry_idx in range(len(qry_lst)):
            #
            qry = qry_lst[qry_idx]
            # Get indexes of connected components
            qry_ccs = list(set([self._cnct_ids[idx] for idx in qry]))
            qry_ccs.sort()
            # Check whether there exists Qry2Blk
            tmp_q2b = MdlIput._exst_lst(q2b_list, qry_ccs)
            # Add query to q2b_list
            if tmp_q2b is not None:
                # Add qry into Qry2Blk
                tmp_q2b.append(qry_idx)
            else:
                # Initialize Qry2Blk and add to list
                q2b_list.append(Qry2Blk(qry_ccs, qry_idx))
        #
        return q2b_list

    def get_eles(self, rela_ids: list):
        """
        [Refer to the Abstract Method]
        2020-10-9
        """
        tmp_eles = list()
        for cc_id in rela_ids:
            tmp_eles.extend([ele for ele, tmp in enumerate(self._cnct_ids) if tmp == cc_id])
        #
        return tmp_eles
