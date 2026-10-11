#
import numpy as np
from MODEL.Asst_RFDP.ProcAsst import ProcAsst
from MODEL.Paper_RFDP.PaperUtils import build_paper_problem
from MODEL.Paper_RFDP.PaperUtils import rank_by_similarity
from MODEL.Paper_RFDP.PaperUtils import select_local_subgraph
from MODEL.Paper_RFDP.PaperUtils import to_numpy
from MODEL.Paper_RFDP.PaperSolvers import GmfptClosedForm
from MODEL.Paper_RFDP.PaperSolvers import HyRdpClosedForm
from MODEL.Paper_RFDP.PaperSolvers import HyRdpIterative


class _PaperRankerBase(object):
    """Common ranking logic for the paper-accurate implementations."""

    def __init__(self, mdl_iput, incl_qry: bool = False, local_topk: int = None,
                 fallback_similarity=None):
        self._mdl_iput = mdl_iput
        self._incl_qry = incl_qry
        self._graph_w = to_numpy(mdl_iput.smth_w)
        self._full_w = to_numpy(mdl_iput.ful_w if mdl_iput.ful_w is not None else mdl_iput.smth_w)
        self._smpl_num = self._graph_w.shape[0]
        self._local_topk = local_topk
        self._fallback_similarity = fallback_similarity
        self.diagnostics = {'solver_calls': 0, 'dropped_unreachable_nodes': 0,
                            'queries_with_unreachable_nodes': 0,
                            'fallback_count': 0, 'fallback_queries': [],
                            'max_relative_residual': 0.0}

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
        sub_ids = self._limit_subset(query_ids, sub_ids)
        try:
            rank_ids, _, _ = self.rank_subset(query_ids, sub_ids, weight_mat)
        except np.linalg.LinAlgError as exc:
            # A failed diffusion is recorded and explicitly falls back to the
            # input metric; never rank a singular or non-finite solver output.
            self.diagnostics['fallback_count'] += 1
            self.diagnostics['fallback_queries'].append(
                {'query_ids': list(query_ids), 'reason': str(exc)})
            rank_ids = list(query_ids)
        return self._complete_output(query_ids, rank_ids, iput_len)

    def rank_subset(self, query_ids: list, sub_ids: list, weight_mat=None) -> (list, object, np.ndarray):
        weight_mat = self._graph_w if weight_mat is None else to_numpy(weight_mat)
        problem = build_paper_problem(weight_mat, sub_ids, query_ids)
        removed = len(sub_ids) - len(problem.sub_ids)
        self.diagnostics['dropped_unreachable_nodes'] += removed
        self.diagnostics['queries_with_unreachable_nodes'] += int(removed > 0)
        if problem.empty_unlabeled():
            return list(query_ids), problem, np.zeros(0, dtype=float)
        self.diagnostics['solver_calls'] += 1
        ul_scores = self._solve_problem(problem)
        if not np.isfinite(ul_scores).all():
            raise np.linalg.LinAlgError('Non-finite diffusion scores')
        residual = getattr(self._solver, 'last_relative_residual', 0.0)
        self.diagnostics['max_relative_residual'] = max(
            self.diagnostics['max_relative_residual'], float(residual))
        rank_pairs = list(zip(problem.unlabeled_ids, ul_scores.tolist()))
        rank_pairs.sort(key=lambda tmp: (tmp[1], tmp[0]))
        rank_ids = list(query_ids)
        rank_ids.extend([tmp[0] for tmp in rank_pairs])
        return rank_ids, problem, ul_scores

    def _default_subset(self, query_ids: list) -> list:
        q2b = self._mdl_iput([query_ids], self._incl_qry)[0]
        return self._mdl_iput.get_eles(q2b.rela_ids)

    def _limit_subset(self, query_ids: list, sub_ids: list) -> list:
        sub_ids = list(dict.fromkeys(int(tmp_id) for tmp_id in sub_ids))
        if self._local_topk is None or len(sub_ids) <= self._local_topk:
            return sub_ids
        return select_local_subgraph(self._full_w, query_ids, self._local_topk,
                                     allowed_ids=sub_ids, include_ids=query_ids)

    def _complete_output(self, query_ids: list, rank_ids: list, iput_len: int) -> list:
        ret_ids = list(rank_ids[:iput_len])
        if len(ret_ids) >= iput_len:
            return ret_ids
        excl_ids = list(dict.fromkeys(query_ids + ret_ids))
        rest_ids = self.rank_input(query_ids, excl_ids,
                                   top_num=iput_len - len(ret_ids))
        ret_ids.extend(rest_ids)
        return ret_ids

    def rank_input(self, query_ids: list, excl_ids: list, top_num: int = None) -> list:
        """Original-metric tail; affinity is legacy fallback when not supplied."""
        if self._fallback_similarity is None:
            return rank_by_similarity(self._full_w, query_ids, excl_ids, top_num)
        scores = np.asarray(self._fallback_similarity(query_ids), dtype=float).reshape(-1).copy()
        if scores.shape != (self._smpl_num,) or not np.isfinite(scores).all():
            raise ValueError('Original-metric callback returned invalid similarities')
        scores[list(excl_ids)] = -np.inf
        ids = np.arange(self._smpl_num)
        order = np.lexsort((ids, -scores))
        result = order[np.isfinite(scores[order])].tolist()
        return result if top_num is None else result[:top_num]

    @property
    def graph_w(self):
        return self._graph_w

    @property
    def full_w(self):
        return self._full_w

    @property
    def local_topk(self):
        return self._local_topk


class HyRdpPaper(_PaperRankerBase):
    """HyRDP ranker; a fixed local_topk pool is an approximation."""

    def __init__(self, mdl_iput, solver=None, alpha: float = 5.0, beta: float = 25.0,
                 incl_qry: bool = False, local_topk: int = None, fallback_similarity=None):
        super(HyRdpPaper, self).__init__(mdl_iput, incl_qry, local_topk, fallback_similarity)
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

    def __init__(self, mdl_iput, solver=None, incl_qry: bool = False, local_topk: int = None, fallback_similarity=None):
        super(GmfptPaper, self).__init__(mdl_iput, incl_qry, local_topk, fallback_similarity)
        self._solver = GmfptClosedForm() if solver is None else solver

    def _solve_problem(self, problem) -> np.ndarray:
        return self._solver.solve(problem)

    @property
    def solver(self):
        return self._solver
