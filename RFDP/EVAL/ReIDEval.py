#
import numpy as np


def evaluate_reid_rankings(rankings: list, query_ids: list, query_pids, query_camids,
                           gallery_pids, gallery_camids, gallery_offset: int,
                           max_rank: int = 50, return_per_query: bool = False) -> dict:
    """Evaluate ReID CMC and mAP from global-id rankings.

    The input rankings may contain query images and gallery images. Only gallery
    ids are evaluated. Market1501-style junk entries are removed:
    same pid/same camera and gallery pid == -1.
    """
    query_pids = np.asarray(query_pids).reshape(-1)
    query_camids = np.asarray(query_camids).reshape(-1)
    gallery_pids = np.asarray(gallery_pids).reshape(-1)
    gallery_camids = np.asarray(gallery_camids).reshape(-1)
    gallery_num = gallery_pids.shape[0]
    max_rank = min(int(max_rank), gallery_num)
    if max_rank <= 0:
        raise ValueError('max_rank should be positive.')
    if len(rankings) != len(query_ids):
        raise ValueError('rankings and query_ids should have the same length.')

    all_cmc = list()
    all_ap = list()
    all_inp = list()
    per_query = list()
    valid_q = 0
    for qry_pos, ranking in enumerate(rankings):
        q_pid = query_pids[query_ids[qry_pos]]
        q_cam = query_camids[query_ids[qry_pos]]
        gal_rank = _global_to_gallery_rank(ranking, gallery_offset, gallery_num)
        cmc, ap, inp, valid, num_rel, retrieved_rel = _evaluate_details(
            gal_rank, q_pid, q_cam, gallery_pids, gallery_camids, max_rank)
        if return_per_query:
            per_query.append({
                'query_id': int(query_ids[qry_pos]), 'pid': int(q_pid), 'camid': int(q_cam),
                'valid': bool(valid), 'AP': ap, 'rank1': float(cmc[0]) if valid else None,
                'INP': inp, 'positive_count': num_rel, 'retrieved_positive_count': retrieved_rel,
            })
        if not valid:
            continue
        all_cmc.append(cmc)
        all_ap.append(ap)
        all_inp.append(inp)
        valid_q += 1

    if valid_q == 0:
        raise RuntimeError('No valid query. Check labels/camera ids and gallery content.')
    cmc = np.asarray(all_cmc, dtype=float).sum(axis=0) / valid_q
    mean_ap = float(np.mean(all_ap))
    metrics = {
        'mAP': mean_ap,
        'mINP': float(np.mean(all_inp)),
        'valid_queries': int(valid_q),
        'cmc': cmc.tolist(),
    }
    for rank in [1, 5, 10, 20]:
        if rank <= max_rank:
            metrics['rank-' + str(rank)] = float(cmc[rank - 1])
    if return_per_query:
        metrics['per_query'] = per_query
    return metrics


def _global_to_gallery_rank(ranking: list, gallery_offset: int, gallery_num: int) -> np.ndarray:
    seen = set()
    gal_ids = list()
    upper = gallery_offset + gallery_num
    for tmp_id in ranking:
        tmp_id = int(tmp_id)
        if tmp_id in seen:
            continue
        seen.add(tmp_id)
        if gallery_offset <= tmp_id < upper:
            gal_ids.append(tmp_id - gallery_offset)
    return np.asarray(gal_ids, dtype=int)


def _evaluate_one(gal_rank: np.ndarray, q_pid: int, q_cam: int, gallery_pids: np.ndarray,
                  gallery_camids: np.ndarray, max_rank: int):
    """Keep the historical private helper's three-value return contract."""
    cmc, ap, _, valid, _, _ = _evaluate_details(
        gal_rank, q_pid, q_cam, gallery_pids, gallery_camids, max_rank)
    return cmc, ap, valid


def _evaluate_details(gal_rank, q_pid, q_cam, gallery_pids, gallery_camids, max_rank):
    # Count all eligible positives in the complete gallery, not only returned hits.
    num_rel = int(np.sum((gallery_pids == q_pid) & (gallery_camids != q_cam)
                         & (gallery_pids != -1)))
    if num_rel == 0:
        return None, None, None, False, 0, 0
    order_pids = gallery_pids[gal_rank]
    order_camids = gallery_camids[gal_rank]
    remove = (order_pids == -1) | ((order_pids == q_pid) & (order_camids == q_cam))
    matches = (order_pids[~remove] == q_pid).astype(np.int32)
    retrieved_rel = int(matches.sum())
    if retrieved_rel == 0:
        return np.zeros(max_rank, dtype=float), 0.0, 0.0, True, num_rel, 0

    cumulative = matches.cumsum()
    cmc = np.minimum(cumulative, 1)
    if cmc.shape[0] < max_rank:
        cmc = np.pad(cmc, (0, max_rank - cmc.shape[0]), constant_values=cmc[-1])
    precision = cumulative / (np.arange(matches.shape[0]) + 1.0)
    ap = float((precision * matches).sum() / num_rel)
    # A missing positive has no finite last-positive rank; conservatively INP=0.
    # mINP is meaningful for complete rankings and must not be inflated by top-k.
    last_positive_rank = int(np.flatnonzero(matches)[-1]) + 1
    inp = float(num_rel / last_positive_rank) if retrieved_rel == num_rel else 0.0
    return cmc[:max_rank].astype(float), ap, inp, True, num_rel, retrieved_rel
