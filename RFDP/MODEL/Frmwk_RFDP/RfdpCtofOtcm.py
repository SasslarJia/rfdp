from RfdpLog.PrntLog_RFDP import prnt_log
from MODEL.Frmwk_RFDP.Rfdp import Rfdp
from MODEL.Asst_RFDP.OperAsst import OperAsst
from MODEL.Iput_RFDP.MdlIput import MdlIput
from MODEL.Smth_RFDP.SmthCstr import SmthCstr
from MODEL.Smth_RFDP.PairwiseSC import PairwiseSC
from MODEL.Fit_RFDP.FitCstr import FitCstr
from MODEL.Fit_RFDP.HyDiffFC import HyDiffFC
from MODEL.Sol_RFDP.SolProc import SolProc
from MODEL.Sol_RFDP.ItrtLinGrw import ItrtLinGrw


class RfdpCtofOtcm(Rfdp):
    """
    Cutoff the outcome of RFDP as the re-ranking result
    2021-6-25
    """

    def __init__(self, mdl_iput: MdlIput, smth_cstr: SmthCstr = PairwiseSC(),
                 fit_cstr: FitCstr = HyDiffFC(), sol_proc: SolProc = ItrtLinGrw(),
                 incl_qry: bool = True):
        """
        [Refer to the Abstract Method]
        2021-6-25
        """
        super(RfdpCtofOtcm, self).__init__(mdl_iput, smth_cstr, fit_cstr, sol_proc, incl_qry)

    def _init_parm(self, ctof_num: int):
        """
        [Refer to the Abstract Method]
        :return:
        2021-6-28
        """
        #
        if ctof_num <= 0:
            prnt_log("Cutoff number should be greater than 0!")
        # Set cutoff number
        self._ctof_num = ctof_num

    def _rfdp_len(self, qry_len: int, sub_len: int) -> int:
        """
        [Refer to the Abstract Method]
        :return:
        2021-6-28
        """
        return self._ctof_num if self._ctof_num <= sub_len else sub_len

    def _extra_len(self, qry_len: int, rfdp_len: int) -> int:
        """
        [Refer to the Abstract Method]
        :return:
        2021-6-28
        """
        return 0 if rfdp_len >= self._ctof_num else self._ctof_num - rfdp_len
