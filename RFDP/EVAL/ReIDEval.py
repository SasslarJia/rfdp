#
import numpy as np


def evaluate_reid_rankings(rankings: list, query_ids: list, query_pids, query_camids,
                           gallery_pids, gallery_camids, gallery_offset: int,
                           max_rank: int = 50) -> dict:
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
    valid_q = 0
    for qry_pos, ranking in enumerate(rankings):
        q_pid = query_pids[query_ids[qry_pos]]
        q_cam = query_camids[query_ids[qry_pos]]
        gal_rank = _global_to_gallery_rank(ranking, gallery_offset, gallery_num)
        cmc, ap, valid = _evaluate_one(gal_rank, q_pid, q_cam, gallery_pids,
                                       gallery_camids, max_rank)
        if not valid:
            continue
        all_cmc.append(cmc)
        all_ap.append(ap)
        valid_q += 1

    if valid_q == 0:
        raise RuntimeError('No valid query. Check labels/camera ids and gallery content.')
    cmc = np.asarray(all_cmc, dtype=float).sum(axis=0) / valid_q
    mean_ap = float(np.mean(all_ap))
    metrics = {
        'mAP': mean_ap,
        'valid_queries': int(valid_q),
        'cmc': cmc.tolist(),
    }
    for rank in [1, 5, 10, 20]:
        if rank <= max_rank:
            metrics['rank-' + str(rank)] = float(cmc[rank - 1])
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
    num_rel = int(np.sum((gallery_pids == q_pid) & (gallery_camids != q_cam)))
    if num_rel == 0:
        return None, None, False
    order_pids = gallery_pids[gal_rank]
    order_camids = gallery_camids[gal_rank]
    remove = (order_pids == -1) | ((order_pids == q_pid) & (order_camids == q_cam))
    keep = np.invert(remove)
    matches = (order_pids[keep] == q_pid).astype(np.int32)
    if not np.any(matches):
        return None, None, False

    cmc = matches.cumsum()
    cmc[cmc > 1] = 1
    if cmc.shape[0] < max_rank:
        pad_val = cmc[-1] if cmc.shape[0] > 0 else 0
        cmc = np.pad(cmc, (0, max_rank - cmc.shape[0]), constant_values=pad_val)
    cmc = cmc[:max_rank]

    tmp_cmc = matches.cumsum()
    precision = tmp_cmc / (np.arange(matches.shape[0]) + 1.0)
    ap = float((precision * matches).sum() / num_rel)
    return cmc.astype(float), ap, True
