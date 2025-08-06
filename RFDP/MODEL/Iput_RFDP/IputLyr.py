#
from MODEL.Iput_RFDP.Qry2Blk import Qry2Blk
from MODEL.Iput_RFDP.MdlIput import MdlIput
from MODEL.Asst_RFDP.OperAsst import OperAsst
from MODEL.Asst_RFDP.OperAsstTorch import OperAsstTorch
from RfdpLog.PrntLog_RFDP import prnt_log


class IputLyr(MdlIput):
    """
    Diffusion process on the certain layers of the query
    2020-12-29
    """

    def __init__(self, smth_w, fit_w=None, full_w=None,
                 asst_rela: OperAsst = OperAsstTorch(), kp_lyrs: int = 2):
        """
        :param smth_w:
        :param fit_w:
        :param full_w:
        :param asst_rela:
        :param kp_lyrs:
        2020-12-29
        """
        super(IputLyr, self).__init__(smth_w, fit_w, full_w, asst_rela)
        #
        if kp_lyrs < 1:
            prnt_log("The kept layers should be larger than 1!")
        elif kp_lyrs > 3:
            prnt_log("The kept layers should be less than or equal to 3!")
        else:
            self._kp_lyrs = kp_lyrs

    def _stl_qries(self, qry_lst: list) -> (list, list):
        """
        [Refer to the Abstract Method]
        2020-10-9
        """
        # Initialize the list with elements of block
        q2b_list = list()
        # Check all the queries
        for qry_idx in range(len(qry_lst)):
            # Get elements indexes on last but one layer
            pres_lyr = qry_lst[qry_idx]
            for lyr_idx in range(self._kp_lyrs - 1):
                # Initialize the list of next layer
                nxt_lyr = list()
                # Add all elements connected each node to list
                for tmp_node in pres_lyr:
                    nxt_lyr.extend(self._edges_lst[tmp_node])
                #
                pres_lyr = list(set(nxt_lyr))
            # Sort the list
            pres_lyr.sort()
            # Check whether there exists Qry2Blk
            tmp_q2b = MdlIput._exst_lst(q2b_list, pres_lyr)
            # Add query to q2b_list
            if tmp_q2b is not None:
                # Add qry into Qry2Blk
                tmp_q2b.append(qry_idx)
            else:
                # Initialize Qry2Blk and add to list
                q2b_list.append(Qry2Blk(pres_lyr, qry_idx))
        #
        return q2b_list

    def get_eles(self, rela_ids: list):
        """
        [Refer to the Abstract Method]
        2020-10-9
        """
        tmp_eles = list()
        for tmp_idx in rela_ids:
            tmp_eles.extend(self._edges_lst[tmp_idx])
        #
        tmp_eles = list(set(tmp_eles))
        tmp_eles.sort()
        #
        return tmp_eles
