#
import numpy as np
from MODEL.Asst_RFDP.ProcAsst import ProcAsst
from MODEL.Paper_RFDP.PaperUtils import add_temporary_connections
from MODEL.Paper_RFDP.PaperUtils import build_shell_subset
from MODEL.Paper_RFDP.PaperUtils import rank_by_similarity
from MODEL.Paper_RFDP.PaperUtils import select_local_subgraph
from MODEL.Paper_RFDP.PaperUtils import to_numpy


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
        if self._base_ranker.local_topk is not None:
            return self._solve_one_local(query_ids)
        labeled = list(query_ids)
        while len(labeled) < self._nq:
            work_w = self._prepare_iteration_graph(labeled)
            _, _, s_ids = build_shell_subset(work_w, labeled)
            s_ids = self._limit_iteration_subset(query_ids, labeled, s_ids)
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

    def _solve_one_local(self, query_ids: list) -> list:
        pool_num = max(self._base_ranker.local_topk, len(query_ids))
        cand_ids = select_local_subgraph(self._base_ranker.full_w, query_ids, pool_num,
                                         include_ids=query_ids)
        if len(cand_ids) == 0:
            return list(query_ids)
        cand_pos = {tmp_id: idx for idx, tmp_id in enumerate(cand_ids)}
        base_local = to_numpy(self._base_ranker.graph_w)[np.ix_(cand_ids, cand_ids)]
        full_local = to_numpy(self._base_ranker.full_w)[np.ix_(cand_ids, cand_ids)]
        query_local = [cand_pos[tmp_id] for tmp_id in query_ids if tmp_id in cand_pos]
        labeled_local = list(query_local)
        target_local = min(self._nq, len(cand_ids))
        while len(labeled_local) < target_local:
            work_local = self._prepare_iteration_graph_local(base_local, full_local, labeled_local)
            _, _, s_local = build_shell_subset(work_local, labeled_local)
            if len(s_local) <= len(labeled_local):
                break
            try:
                rank_local, _, _ = self._base_ranker.rank_subset(labeled_local, s_local, work_local)
            except Exception:
                break
            new_local = [tmp_id for tmp_id in rank_local if tmp_id not in labeled_local][:self._lambda_step]
            if len(new_local) == 0:
                break
            labeled_local.extend(new_local)
        if len(labeled_local) < target_local:
            excl_local = list(dict.fromkeys(labeled_local))
            rest_local = rank_by_similarity(full_local, query_local, excl_local,
                                            top_num=target_local - len(labeled_local))
            labeled_local.extend(rest_local)
        ret_ids = [cand_ids[tmp_id] for tmp_id in labeled_local[:target_local]]
        if len(ret_ids) < self._nq:
            excl_ids = list(dict.fromkeys(ret_ids))
            rest_ids = rank_by_similarity(self._base_ranker.full_w, query_ids, excl_ids,
                                          top_num=self._nq - len(ret_ids))
            ret_ids.extend(rest_ids)
        return ret_ids[:self._nq]

    def _prepare_iteration_graph(self, labeled: list):
        base_w = self._base_ranker.graph_w
        _, _, s_ids = build_shell_subset(base_w, labeled)
        work_w, _ = add_temporary_connections(base_w, self._base_ranker.full_w, s_ids,
                                              self._lambda_step)
        return work_w

    def _prepare_iteration_graph_local(self, base_local, full_local, labeled_local: list):
        _, _, s_local = build_shell_subset(base_local, labeled_local)
        work_local, _ = add_temporary_connections(base_local, full_local, s_local,
                                                  self._lambda_step)
        return work_local

    def _limit_iteration_subset(self, query_ids: list, labeled: list, s_ids: list) -> list:
        local_topk = self._base_ranker.local_topk
        if local_topk is None or len(s_ids) <= local_topk:
            return s_ids
        seed_ids = list(dict.fromkeys(query_ids + labeled))
        return select_local_subgraph(self._base_ranker.full_w, seed_ids, local_topk,
                                     allowed_ids=s_ids, include_ids=seed_ids)


class IHyRdpPaper(_PaperIterativeRanker):
    pass


class IGmfptPaper(_PaperIterativeRanker):
    pass
