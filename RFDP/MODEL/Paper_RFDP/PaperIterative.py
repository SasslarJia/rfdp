#
from MODEL.Asst_RFDP.ProcAsst import ProcAsst
from MODEL.Paper_RFDP.PaperUtils import add_temporary_connections
from MODEL.Paper_RFDP.PaperUtils import build_shell_subset
from MODEL.Paper_RFDP.PaperUtils import rank_by_similarity


class _PaperIterativeRanker(object):
    """Shared implementation for I-HyRDP and I-GMFPT."""

    def __init__(self, base_ranker, nq: int, lambda_step: int = 5):
        self._base_ranker = base_ranker
        self._nq = nq
        self._lambda_step = lambda_step

    def __call__(self, iput_qries: list) -> list:
        if not ProcAsst.meas_qry(iput_qries, self._nq):
            return None
        ret_lst = list()
        for qry in iput_qries:
            ret_lst.append(self._solve_one(qry))
        return ret_lst

    def _solve_one(self, query_ids: list) -> list:
        labeled = list(query_ids)
        while len(labeled) < self._nq:
            work_w = self._prepare_iteration_graph(labeled)
            _, _, s_ids = build_shell_subset(work_w, labeled)
            if len(s_ids) <= len(labeled):
                break
            try:
                rank_ids, _, _ = self._base_ranker.rank_subset(labeled, s_ids, work_w)
            except Exception:
                break
            new_ids = [tmp_id for tmp_id in rank_ids if tmp_id not in labeled][:self._lambda_step]
            if len(new_ids) == 0:
                break
            labeled.extend(new_ids)
        if len(labeled) < self._nq:
            excl_ids = list(dict.fromkeys(labeled))
            rest_ids = rank_by_similarity(self._base_ranker.full_w, query_ids, excl_ids,
                                          top_num=self._nq - len(labeled))
            labeled.extend(rest_ids)
        return labeled[:self._nq]

    def _prepare_iteration_graph(self, labeled: list):
        base_w = self._base_ranker.graph_w
        _, _, s_ids = build_shell_subset(base_w, labeled)
        work_w, _ = add_temporary_connections(base_w, self._base_ranker.full_w, s_ids,
                                              self._lambda_step)
        return work_w


class IHyRdpPaper(_PaperIterativeRanker):
    pass


class IGmfptPaper(_PaperIterativeRanker):
    pass
