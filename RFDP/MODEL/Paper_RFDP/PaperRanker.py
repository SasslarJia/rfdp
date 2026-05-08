#
import numpy as np
from MODEL.Asst_RFDP.ProcAsst import ProcAsst
from MODEL.Paper_RFDP.PaperUtils import build_paper_problem
from MODEL.Paper_RFDP.PaperUtils import rank_by_similarity
from MODEL.Paper_RFDP.PaperUtils import to_numpy
from MODEL.Paper_RFDP.PaperSolvers import GmfptClosedForm
from MODEL.Paper_RFDP.PaperSolvers import HyRdpClosedForm
from MODEL.Paper_RFDP.PaperSolvers import HyRdpIterative


class _PaperRankerBase(object):
    """Common ranking logic for the paper-accurate implementations."""

    def __init__(self, mdl_iput, incl_qry: bool = False):
        self._mdl_iput = mdl_iput
        self._incl_qry = incl_qry
        self._graph_w = to_numpy(mdl_iput.smth_w)
        self._full_w = to_numpy(mdl_iput.ful_w if mdl_iput.ful_w is not None else mdl_iput.smth_w)
        self._smpl_num = self._graph_w.shape[0]

    def __call__(self, iput_qries: list, iput_len: int) -> list:
        dist_qries, dist_poses = ProcAsst.find_idcl(iput_qries)
        q2b_lst = self._mdl_iput(dist_qries, self._incl_qry)
        rslt_dict = dict()
        for q2b in q2b_lst:
            sub_ids = self._mdl_iput.get_eles(q2b.rela_ids)
            for tmp_qid in q2b.qids_lst:
                rslt_dict[tmp_qid] = self.solve_query(dist_qries[tmp_qid], iput_len, sub_ids=sub_ids)
        ret_lst = [rslt_dict[idx] for idx in sorted(rslt_dict.keys())]
        return ProcAsst.cvrtbk_idcl(ret_lst, iput_qries, dist_poses)

    def solve_query(self, query_ids: list, iput_len: int, sub_ids: list = None, weight_mat=None) -> list:
        if sub_ids is None:
            sub_ids = self._default_subset(query_ids)
        rank_ids, _, _ = self.rank_subset(query_ids, sub_ids, weight_mat)
        return self._complete_output(query_ids, rank_ids, iput_len)

    def rank_subset(self, query_ids: list, sub_ids: list, weight_mat=None) -> (list, object, np.ndarray):
        weight_mat = self._graph_w if weight_mat is None else to_numpy(weight_mat)
        problem = build_paper_problem(weight_mat, sub_ids, query_ids)
        if problem.empty_unlabeled():
            return list(query_ids), problem, np.zeros(0, dtype=float)
        ul_scores = self._solve_problem(problem)
        rank_pairs = list(zip(problem.unlabeled_ids, ul_scores.tolist()))
        rank_pairs.sort(key=lambda tmp: (tmp[1], tmp[0]))
        rank_ids = list(query_ids)
        rank_ids.extend([tmp[0] for tmp in rank_pairs])
        return rank_ids, problem, ul_scores

    def _default_subset(self, query_ids: list) -> list:
        q2b = self._mdl_iput([query_ids], self._incl_qry)[0]
        return self._mdl_iput.get_eles(q2b.rela_ids)

    def _complete_output(self, query_ids: list, rank_ids: list, iput_len: int) -> list:
        ret_ids = list(rank_ids[:iput_len])
        if len(ret_ids) >= iput_len:
            return ret_ids
        excl_ids = list(dict.fromkeys(query_ids + ret_ids))
        rest_ids = rank_by_similarity(self._full_w, query_ids, excl_ids,
                                      top_num=iput_len - len(ret_ids))
        ret_ids.extend(rest_ids)
        return ret_ids

    @property
    def graph_w(self):
        return self._graph_w

    @property
    def full_w(self):
        return self._full_w


class HyRdpPaper(_PaperRankerBase):
    """Paper-accurate HyRDP ranker."""

    def __init__(self, mdl_iput, solver=None, alpha: float = 5.0, beta: float = 25.0,
                 incl_qry: bool = False):
        super(HyRdpPaper, self).__init__(mdl_iput, incl_qry)
        self._solver = HyRdpIterative() if solver is None else solver
        self._alpha = alpha
        self._beta = beta

    def _solve_problem(self, problem) -> np.ndarray:
        return self._solver.solve(problem, self._alpha, self._beta)

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def solver(self):
        return self._solver


class GmfptPaper(_PaperRankerBase):
    """Paper-accurate GMFPT ranker."""

    def __init__(self, mdl_iput, solver=None, incl_qry: bool = False):
        super(GmfptPaper, self).__init__(mdl_iput, incl_qry)
        self._solver = GmfptClosedForm() if solver is None else solver

    def _solve_problem(self, problem) -> np.ndarray:
        return self._solver.solve(problem)

    @property
    def solver(self):
        return self._solver

