#
import numpy as np
from RfdpLog.PrntLog_RFDP import prnt_log
from MODEL.Paper_RFDP.PaperProblem import PaperProblem


def to_numpy(iput_mat) -> np.ndarray:
    """Convert numpy / torch-like matrices to numpy arrays."""
    if isinstance(iput_mat, np.ndarray):
        if np.issubdtype(iput_mat.dtype, np.floating):
            return iput_mat
        return iput_mat.astype(np.float32, copy=False)
    if hasattr(iput_mat, 'detach'):
        tmp_mat = iput_mat.detach().cpu().numpy()
        if np.issubdtype(tmp_mat.dtype, np.floating):
            return tmp_mat
        return tmp_mat.astype(np.float32, copy=False)
    if hasattr(iput_mat, 'cpu') and hasattr(iput_mat, 'numpy'):
        tmp_mat = iput_mat.cpu().numpy()
        if np.issubdtype(tmp_mat.dtype, np.floating):
            return tmp_mat
        return tmp_mat.astype(np.float32, copy=False)
    tmp_mat = np.asarray(iput_mat)
    if np.issubdtype(tmp_mat.dtype, np.floating):
        return tmp_mat
    return tmp_mat.astype(np.float32, copy=False)


def safe_divide(numr, dnmtr):
    numr = np.asarray(numr, dtype=float)
    dnmtr = np.asarray(dnmtr, dtype=float)
    return np.divide(numr, dnmtr, out=np.zeros_like(numr, dtype=float), where=dnmtr != 0)


def row_normalize(iput_mat, row_sums=None) -> np.ndarray:
    iput_mat = np.asarray(iput_mat, dtype=float)
    if row_sums is None:
        row_sums = iput_mat.sum(axis=1)
    row_sums = np.asarray(row_sums, dtype=float).reshape(-1)
    return safe_divide(iput_mat, row_sums[:, None])


def solve_linear(coef_mat: np.ndarray, rhs_mat):
    coef_mat = np.asarray(coef_mat, dtype=float)
    rhs_mat = np.asarray(rhs_mat, dtype=float)
    try:
        return np.linalg.solve(coef_mat, rhs_mat)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(coef_mat, rhs_mat, rcond=None)[0]


def connected_components(weight_mat) -> list:
    weight_mat = to_numpy(weight_mat)
    node_num = weight_mat.shape[0]
    vis_flg = [False] * node_num
    comp_lst = list()
    for node_idx in range(node_num):
        if vis_flg[node_idx]:
            continue
        stack = [node_idx]
        vis_flg[node_idx] = True
        comp = list()
        while len(stack) > 0:
            tmp_idx = stack.pop()
            comp.append(tmp_idx)
            nbrs = np.flatnonzero(weight_mat[tmp_idx, :] > 0)
            for nbr in nbrs:
                if not vis_flg[nbr]:
                    vis_flg[nbr] = True
                    stack.append(int(nbr))
        comp.sort()
        comp_lst.append(comp)
    return comp_lst


def build_shell_subset(weight_mat, labeled_ids: list) -> (list, list, list):
    """Build S_t = L_t + A_t + B_t on the given graph."""
    weight_mat = to_numpy(weight_mat)
    l_set = set(labeled_ids)
    if len(l_set) == 0:
        return list(), list(), list()
    a_ids = set()
    for node_idx in labeled_ids:
        a_ids.update(np.flatnonzero(weight_mat[node_idx, :] > 0).tolist())
    a_ids.difference_update(l_set)
    b_ids = set()
    for node_idx in a_ids:
        b_ids.update(np.flatnonzero(weight_mat[node_idx, :] > 0).tolist())
    b_ids.difference_update(l_set)
    b_ids.difference_update(a_ids)
    s_ids = sorted(l_set.union(a_ids).union(b_ids))
    return sorted(a_ids), sorted(b_ids), s_ids


def build_paper_problem(weight_mat, sub_ids: list, query_ids: list) -> PaperProblem:
    """Construct the local problem using the exact variable split in the paper."""
    sub_ids = list(sub_ids)
    query_ids = list(query_ids)
    if len(sub_ids) == 0:
        prnt_log.error("Subset cannot be empty")
    if not set(query_ids).issubset(set(sub_ids)):
        prnt_log.error("Query should be contained in the subset")
    sub_w = to_numpy(weight_mat)[np.ix_(sub_ids, sub_ids)]
    qry_local = [sub_ids.index(tmp_id) for tmp_id in query_ids]
    qry_set = set(qry_local)
    unl_local = [idx for idx in range(len(sub_ids)) if idx not in qry_set]
    direct_vals = sub_w[:, qry_local].sum(axis=1) if len(qry_local) > 0 else np.zeros(len(sub_ids))
    a_local = [idx for idx in unl_local if direct_vals[idx] > 0]
    b_local = [idx for idx in unl_local if direct_vals[idx] <= 0]
    perm_local = a_local + b_local + qry_local
    perm_ids = [sub_ids[idx] for idx in perm_local]
    perm_w = sub_w[np.ix_(perm_local, perm_local)]
    unl_num = len(unl_local)
    w_uu = perm_w[:unl_num, :unl_num]
    w_ul = perm_w[:unl_num, unl_num:]
    d_u = perm_w[:unl_num, :].sum(axis=1)
    h_u = w_uu.sum(axis=1)
    e_u = w_ul.sum(axis=1)
    p_uu = row_normalize(w_uu, d_u)
    a_ids = [sub_ids[idx] for idx in a_local]
    b_ids = [sub_ids[idx] for idx in b_local]
    unlabeled_ids = a_ids + b_ids
    return PaperProblem(query_ids=query_ids, labeled_ids=list(query_ids),
                        unlabeled_ids=unlabeled_ids, a_ids=a_ids, b_ids=b_ids,
                        w_uu=w_uu, w_ul=w_ul, h_u=h_u, d_u=d_u, e_u=e_u, p_uu=p_uu,
                        sub_ids=sub_ids, is_connected_to_l=e_u > 0, perm_ids=perm_ids,
                        w_sub=sub_w)


def rank_by_similarity(full_w, query_ids: list, excl_ids: list, top_num: int = None) -> list:
    full_w = to_numpy(full_w)
    tmp_vec = full_w[query_ids, :].sum(axis=0)
    tmp_vec = np.asarray(tmp_vec, dtype=float).reshape(-1)
    if len(excl_ids) > 0:
        tmp_vec[excl_ids] = -float('inf')
    sort_ids = np.argsort(-tmp_vec)
    ret_ids = [int(tmp_id) for tmp_id in sort_ids if np.isfinite(tmp_vec[tmp_id])]
    return ret_ids if top_num is None else ret_ids[:top_num]


def select_local_subgraph(full_w, seed_ids: list, max_nodes: int, allowed_ids: list = None,
                          include_ids: list = None) -> list:
    """Pick a small local candidate pool around the seed nodes."""
    full_w = to_numpy(full_w)
    seed_ids = list(dict.fromkeys(int(tmp_id) for tmp_id in seed_ids))
    if len(seed_ids) == 0:
        return list()
    if include_ids is None:
        include_ids = list(seed_ids)
    else:
        include_ids = list(dict.fromkeys(int(tmp_id) for tmp_id in include_ids))
    max_nodes = max(int(max_nodes), len(include_ids))
    node_num = full_w.shape[0]
    if allowed_ids is None:
        cand_ids = np.arange(node_num, dtype=int)
    else:
        cand_ids = np.asarray(list(dict.fromkeys(int(tmp_id) for tmp_id in allowed_ids)), dtype=int)
    score_vec = np.asarray(full_w[seed_ids, :].sum(axis=0), dtype=float).reshape(-1)
    cand_scores = score_vec[cand_ids]
    sort_ids = np.lexsort((cand_ids, -cand_scores))
    ret_ids = list()
    used_ids = set()
    allowed_set = set(cand_ids.tolist())
    for tmp_id in include_ids:
        if tmp_id in allowed_set and tmp_id not in used_ids:
            ret_ids.append(tmp_id)
            used_ids.add(tmp_id)
    for tmp_id in cand_ids[sort_ids].tolist():
        if tmp_id in used_ids:
            continue
        ret_ids.append(tmp_id)
        used_ids.add(tmp_id)
        if len(ret_ids) >= max_nodes:
            break
    return ret_ids[:max_nodes]


def add_temporary_connections(base_w, full_w, active_ids: list, bridge_num: int):
    """Add paper-style temporary connections for the selected components."""
    base_w = to_numpy(base_w)
    full_w = to_numpy(full_w)
    if bridge_num <= 0 or len(active_ids) == 0:
        return np.array(base_w, copy=True), list()
    work_w = np.array(base_w, copy=True)
    full_p = row_normalize(full_w)
    comps = connected_components(base_w)
    act_set = set(active_ids)
    comp_in = [comp for comp in comps if len(act_set.intersection(comp)) > 0]
    comp_out = [comp for comp in comps if len(act_set.intersection(comp)) == 0]
    comp_pairs = list()
    act_lst = sorted(act_set)
    for comp in comp_out:
        tmp_blk = full_p[np.ix_(comp, act_lst)]
        if tmp_blk.size == 0:
            continue
        tmp_idx = int(np.argmax(tmp_blk))
        row_idx, col_idx = np.unravel_index(tmp_idx, tmp_blk.shape)
        comp_pairs.append((float(tmp_blk[row_idx, col_idx]), comp[row_idx], act_lst[col_idx]))
    comp_pairs.sort(key=lambda tmp: tmp[0], reverse=True)
    chosen = list()
    for _, fir_id, sec_id in comp_pairs[:bridge_num]:
        work_w[fir_id, sec_id] = full_w[fir_id, sec_id]
        work_w[sec_id, fir_id] = full_w[sec_id, fir_id]
        chosen.append((fir_id, sec_id))
    return work_w, chosen
