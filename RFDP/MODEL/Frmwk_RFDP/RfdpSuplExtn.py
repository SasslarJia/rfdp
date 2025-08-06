from MODEL.Frmwk_RFDP.Rfdp import Rfdp
from MODEL.Asst_RFDP.OperAsst import OperAsst
from MODEL.Iput_RFDP.MdlIput import MdlIput
from MODEL.Smth_RFDP.SmthCstr import SmthCstr
from MODEL.Smth_RFDP.PairwiseSC import PairwiseSC
from MODEL.Fit_RFDP.FitCstr import FitCstr
from MODEL.Fit_RFDP.HyDiffFC import HyDiffFC
from MODEL.Sol_RFDP.SolProc import SolProc
from MODEL.Sol_RFDP.ItrtLinGrw import ItrtLinGrw
from RfdpLog.PrntLog_RFDP import prnt_log


class RfdpSuplExtn(Rfdp):
    """
    Supplement the query with several elements from the outcome of RFDP
    2021-6-25
    """

    def __init__(self, mdl_iput: MdlIput, smth_cstr: SmthCstr = PairwiseSC(),
                 fit_cstr: FitCstr = HyDiffFC(), sol_proc: SolProc = ItrtLinGrw(),
                 incl_qry: bool = True):
        """
        [Refer to the Abstract Method]
        2021-6-25
        """
        super(RfdpSuplExtn, self).__init__(mdl_iput, smth_cstr, fit_cstr, sol_proc, incl_qry)

    def _init_parm(self, extn_num: int):
        """
        [Refer to the Abstract Method]
        :return:
        2021-6-28
        """
        if extn_num < 0:
            prnt_log("Extension number should be greater than or equal to 0!")
        # Set extent number
        self._extn_num = extn_num

    def _rfdp_len(self, qry_len: int, sub_len: int) -> int:
        """
        [Refer to the Abstract Method]
        :return:
        2021-6-28
        """
        #
        if self._extn_num == 0:
            return sub_len
        else:
            return sub_len if qry_len + self._extn_num > sub_len else qry_len + self._extn_num

    def _extra_len(self, qry_len: int, rfdp_len: int) -> int:
        """
        [Refer to the Abstract Method]
        :return:
        2021-6-28
        """
        return self._extn_num if qry_len >= rfdp_len else 0
