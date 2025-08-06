#
from abc import ABCMeta
from abc import abstractmethod as ABCFctn
from MODEL.Asst_RFDP.ProcAsst import ProcAsst
from MODEL.Asst_RFDP.OperAsst import OperAsst
from MODEL.Iput_RFDP.MdlIput import MdlIput
from MODEL.Smth_RFDP.SmthCstr import SmthCstr
from MODEL.Fit_RFDP.FitCstr import FitCstr
from MODEL.Sol_RFDP.SolProc import SolProc
# from MODEL.Smth_RFDP.PairwiseSC import PairwiseSC
# from MODEL.Fit_RFDP.HybrdFC import HybrdFC
# from MODEL.Sol_RFDP.ItrtLinGrw import ItrtLinGrw
from RfdpLog.PrntLog_RFDP import prnt_log
import operator


class Rfdp(metaclass=ABCMeta):
    """
    Regularization Intg_Frmwk of Diffusion Process (RFDP)
    2020-10-9
    """

    def __init__(self, mdl_iput: MdlIput, smth_cstr: SmthCstr, fit_cstr: FitCstr,
                 sol_proc: SolProc, incl_qry: bool = True):
        """
        :param mdl_iput: The input of the model
        :param smth_cstr: Input smoothness constraint
        :param fit_cstr: Input fitting constraint
        :param sol_proc: Solving process
        :param incl_qry: Whether construct objective function with query
        :return:
        2020-10-9
        """
        # Set parameters
        self._mdl_iput = mdl_iput
        self._smth_cstr = smth_cstr
        self._fit_cstr = fit_cstr
        self._sol_proc = sol_proc
        self._incl_qry = incl_qry

    def __call__(self, iput_qries: list, iput_len: int) -> list:
        """
        :param iput_qries: Input list of queries
        :param iput_len: Number related to the output
        :return:
        2020-11-28
        """
        # Init parameters
        self._init_parm(iput_len)
        #
        dist_qries, dist_poses = ProcAsst.find_idcl(iput_qries)
        # Get re-rank results
        return ProcAsst.cvrtbk_idcl(self._rerank(dist_qries), iput_qries, dist_poses)

    def _rerank(self, qry_lst: list) -> list:
        """
        :param qry_lst: Input list of queries
        :return:
        2021-6-25
        """
        # Carry out MdlIput with input queries
        q2b_lst = self._mdl_iput(qry_lst, self._incl_qry)
        # Initialize the output result
        rslt_dict = dict()
        # Solve each sub-problem
        for q2b in q2b_lst:
            # <editor-fold desc="Determine query indexes">
            # Get subset-element indexes in the dataset
            sub_ids = self._mdl_iput.get_eles(q2b.rela_ids)
            # Subset of queries sharing the same connected components
            ptl_qries = [qry_lst[ele] for ele in q2b.qids_lst]
            # Get indexes of queries in the subset
            qries_insub = Rfdp._qries_reidx(sub_ids, ptl_qries)
            # </editor-fold>
            # <editor-fold desc="Solution Process">
            # Get  L=D-W RFDP with smoothness constraint
            d_cols, w_mats = self._smth_cstr(self._mdl_iput, sub_ids, qries_insub)
            # Carry out RFDP with fitting constraint
            coef_cols, cnst_cols = self._fit_cstr(self._mdl_iput, sub_ids, qries_insub)
            # Get the list of RFDP solution
            sol_insub = self._sol_proc(qries_insub, d_cols, w_mats, coef_cols, cnst_cols,
                                       self._fit_cstr.fctr_parm, self._fit_cstr.is_sim,
                                       self._mdl_iput.asst_rela)
            # Get RFDP results
            rfdp_rslts = self._sol2rank(ptl_qries, sol_insub, sub_ids)
            # Get final results by adding extra elements from the full matrix
            fnl_rslts = self._cplt_oput(ptl_qries, rfdp_rslts, self._mdl_iput.ful_w,
                                       self._fit_cstr.is_sim, self._mdl_iput.asst_rela)
            #
            for idx in range(len(q2b.qids_lst)):
                rslt_dict[q2b.qids_lst[idx]] = fnl_rslts[idx]
        #
        tuple_lst = sorted(rslt_dict.items(), key=operator.itemgetter(0))
        return [tmp[1] for tmp in tuple_lst]

    # <editor-fold desc="Choose the re-ranking result from the solution of RFDP">
    def _sol2rank(self, ptl_qries: list, sol_insub: list, sub_ids: list) -> list:
        """ Choose the re-ranking result from the solution of RFDP
        :param ptl_qries: List of queries
        :param sol_insub: Solution of RFDP
        :param sub_ids: List of sub-indexes
        :return:
        2021-6-27
        """
        # Initialize the list of output result
        rfdp_rslts = list()
        #
        for idx in range(len(sol_insub)):
            #
            tmp_sol = sol_insub[idx]
            tmp_len = self._rfdp_len(len(ptl_qries[idx]), len(sub_ids))
            #
            tmp_rslt = [sub_ids[sub_idx] for sub_idx in tmp_sol[:tmp_len]]
            # Add the rests to the end
            rfdp_rslts.append(tmp_rslt)
        #
        return rfdp_rslts

    @ABCFctn
    def _init_parm(self, iput_num: int):
        """ Initialize input parameters
        :return:
        """
        pass

    @ABCFctn
    def _rfdp_len(self, qry_len: int, sub_len: int) -> int:
        """ Length of result for RFDP
        :param qry_len: Length of query
        :return:
        2021-6-28
        """
        pass

    @ABCFctn
    def _extra_len(self, qry_len: int, rfdp_len: int) -> int:
        """ Leength of final result
        :param qry_len: Length of query
        :param rfdp_len: Length of RFDP result
        :return:
        2021-6-28
        """
        pass

    # <editor-fold desc="Update the indexes of queries in the subset">
    @staticmethod
    def _qries_reidx(ids_lst: list, qries_lst: list) -> list:
        """ Update qry according to 'ids_lst'
        :param ids_lst:
        :param qries_lst:
        :return:
        2020-12-10
        """
        # Get updated indexes of query in the given element list
        qries_insub = list()
        for lst in qries_lst:
            qries_insub.append([ids_lst.index(tmp) for tmp in lst])
        #
        return qries_insub
    # </editor-fold>

    # <editor-fold desc="Organize the re-ranking results">
    def _cplt_oput(self, qries_lst: list, rfdp_rslts: list, ful_w, is_sim: bool,
                   asst_rela: OperAsst) -> list:
        """ Complete the final re-ranking results
        :param qries_lst:
        :param rfdp_rslts:
        :param ful_w:
        :param is_sim:
        :param asst_rela: Omit
        :return:
        """
        #
        for qry_idx in range(len(qries_lst)):
            #
            qry = qries_lst[qry_idx]
            tmp_rfdp_len = len(rfdp_rslts[qry_idx])
            #
            tmp_extra_len = self._extra_len(len(qry), tmp_rfdp_len)
            #
            if tmp_extra_len > 0:
                #
                tmp_vec = asst_rela.sum2lbo(ful_w[qry, :])
                tmp_vec[0, rfdp_rslts[qry_idx]] = 0.0 if is_sim else float('inf')
                _, rest_ids = asst_rela.topk_2d(tmp_vec, tmp_extra_len, dim=1, rvrs=is_sim)
                #
                rfdp_rslts[qry_idx].extend(asst_rela.cvrt2lst(rest_ids[0, :])[:tmp_extra_len])
        #
        return rfdp_rslts
    # </editor-fold>
