#
import argparse
import hashlib
import subprocess
import json
import os
import sys

import numpy as np

#
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from EVAL.ReIDEval import evaluate_reid_rankings
from MODEL.Aff_RFDP.Dist2Aff import Dist2Aff
from MODEL.Aff_RFDP.KnnRstct import KnnRstct
from MODEL.Asst_RFDP.OperAsstNumpy import OperAsstNumpy
from MODEL.Iput_RFDP.IputWhl import IputWhl
from MODEL.Paper_RFDP.PaperIterative import IGmfptPaper
from MODEL.Paper_RFDP.PaperIterative import IHyRdpPaper
from MODEL.Paper_RFDP.PaperRanker import GmfptPaper
from MODEL.Paper_RFDP.PaperRanker import HyRdpPaper
from MODEL.Paper_RFDP.PaperSolvers import HyRdpClosedForm
from MODEL.Paper_RFDP.PaperSolvers import HyRdpIterative


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run paper-accurate RFDP re-ranking on ReID feature .mat files.'
    )
    parser.add_argument('--mat-path', type=str, default='RFDP/pytorch_result.mat',
                        help='Feature .mat path with query/gallery features, labels, and cameras.')
    parser.add_argument('--query-feature-key', type=str, default='query_f')
    parser.add_argument('--gallery-feature-key', type=str, default='gallery_f')
    parser.add_argument('--query-label-key', type=str, default='query_label')
    parser.add_argument('--gallery-label-key', type=str, default='gallery_label')
    parser.add_argument('--query-cam-key', type=str, default='query_cam')
    parser.add_argument('--gallery-cam-key', type=str, default='gallery_cam')
    parser.add_argument('--cache-path', type=str, default=None,
                        help='Optional .npz cache for aff_mat/knn_aff.')
    parser.add_argument('--save-cache', type=str, default=None,
                        help='Optional .npz path to save aff_mat/knn_aff.')
    parser.add_argument('--cache-aff-key', type=str, default='aff_mat')
    parser.add_argument('--cache-knn-key', type=str, default='knn_aff')
    parser.add_argument('--graph-mode', type=str, default='dense',
                        choices=['dense', 'sparse_topk'],
                        help='dense keeps the original full-matrix graph; sparse_topk builds a top-k sparse approximation.')
    parser.add_argument('--sparse-full-topk', type=int, default=500,
                        help='Neighbors retained per row in aff_mat when graph-mode=sparse_topk.')
    parser.add_argument('--k-num', type=int, default=20,
                        help='K for distance-to-affinity and KNN graph construction.')
    parser.add_argument('--gamma', type=float, default=0.4,
                        help='Gamma for Dist2Aff.')
    parser.add_argument('--method', type=str, default='ihyrdp',
                        choices=['baseline', 'affinity', 'hyrdp', 'gmfpt', 'ihyrdp', 'igmfpt'])
    parser.add_argument('--hyrdp-solver', type=str, default='iterative',
                        choices=['iterative', 'closed'])
    parser.add_argument('--alpha', type=float, default=5.0)
    parser.add_argument('--beta', type=float, default=25.0)
    parser.add_argument('--topk', type=int, default=None,
                        help='RFDP output length per query. Defaults to all samples.')
    parser.add_argument('--eval-max-rank', type=int, default=50,
                        help='Maximum CMC rank to report/store.')
    parser.add_argument('--lambda-step', type=int, default=5)
    parser.add_argument('--local-topk', type=int, default=None,
                        help='Limit each solver call to a local candidate pool.')
    parser.add_argument('--query-limit', type=int, default=None,
                        help='Only evaluate the first N query images.')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Feature distance batch size when building the full matrix.')
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float32', 'float64'],
                        help='Matrix dtype for distances/affinities.')
    parser.add_argument('--save-json', type=str, default=None,
                        help='Optional output JSON with metrics and rankings.')
    parser.add_argument('--save-rankings', type=str, default=None,
                        help='Optional .npy path to save equal-length rankings as int32, without pickle.')
    parser.add_argument('--save-per-query-json', type=str, default=None,
                        help='Optional JSON with query AP, Rank-1, INP, identity, and camera.')
    parser.add_argument('--preview', type=int, default=3)
    return parser.parse_args()


def load_mat(path: str) -> dict:
    ext = os.path.splitext(path)[1].lower()
    if ext != '.mat':
        raise ValueError('--mat-path should point to a .mat file.')
    try:
        import scipy.io as scio
    except ImportError as exc:
        raise ImportError('Loading .mat files requires scipy.') from exc
    return scio.loadmat(path)


def get_required(data: dict, key: str):
    if key not in data:
        valid_keys = [tmp_key for tmp_key in data.keys() if not tmp_key.startswith('__')]
        raise KeyError("Key '" + key + "' not found. Available keys: " + str(valid_keys))
    return data[key]


def load_reid_data(args):
    data = load_mat(args.mat_path)
    query_f = np.asarray(get_required(data, args.query_feature_key), dtype=float)
    gallery_f = np.asarray(get_required(data, args.gallery_feature_key), dtype=float)
    query_pids = np.asarray(get_required(data, args.query_label_key)).reshape(-1).astype(int)
    gallery_pids = np.asarray(get_required(data, args.gallery_label_key)).reshape(-1).astype(int)
    query_camids = np.asarray(get_required(data, args.query_cam_key)).reshape(-1).astype(int)
    gallery_camids = np.asarray(get_required(data, args.gallery_cam_key)).reshape(-1).astype(int)
    if query_f.ndim != 2 or gallery_f.ndim != 2:
        raise ValueError('query/gallery features should be 2D matrices.')
    if query_f.shape[1] != gallery_f.shape[1]:
        raise ValueError('query/gallery feature dimensions do not match.')
    if query_f.shape[0] != query_pids.shape[0] or query_f.shape[0] != query_camids.shape[0]:
        raise ValueError('query feature/label/camera counts do not match.')
    if gallery_f.shape[0] != gallery_pids.shape[0] or gallery_f.shape[0] != gallery_camids.shape[0]:
        raise ValueError('gallery feature/label/camera counts do not match.')
    return query_f, gallery_f, query_pids, query_camids, gallery_pids, gallery_camids


def l2_normalize(feats: np.ndarray) -> np.ndarray:
    feats = np.asarray(feats, dtype=float)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return feats / norms


def pairwise_l2_distance(feats: np.ndarray, batch_size: int, dtype) -> np.ndarray:
    feats = l2_normalize(feats).astype(dtype, copy=False)
    sample_num = feats.shape[0]
    dist_mat = np.empty((sample_num, sample_num), dtype=dtype)
    feat_t = feats.T
    for start in range(0, sample_num, batch_size):
        end = min(start + batch_size, sample_num)
        sim = feats[start:end].dot(feat_t)
        block = np.maximum(2.0 - 2.0 * sim, 0.0)
        np.sqrt(block, out=block)
        dist_mat[start:end] = block.astype(dtype, copy=False)
        print('Distance rows', str(end) + '/' + str(sample_num))
    np.fill_diagonal(dist_mat, 0.0)
    return dist_mat


def batch_topk_l2(feats: np.ndarray, batch_size: int, dtype, top_num: int):
    feats = l2_normalize(feats).astype(dtype, copy=False)
    sample_num = feats.shape[0]
    top_num = min(int(top_num), sample_num)
    top_dists = np.empty((sample_num, top_num), dtype=dtype)
    top_ids = np.empty((sample_num, top_num), dtype=np.int32)
    feat_t = feats.T
    for start in range(0, sample_num, batch_size):
        end = min(start + batch_size, sample_num)
        sim = feats[start:end].dot(feat_t)
        dist = np.maximum(2.0 - 2.0 * sim, 0.0)
        np.sqrt(dist, out=dist)
        part_ids = np.argpartition(dist, top_num - 1, axis=1)[:, :top_num]
        part_vals = np.take_along_axis(dist, part_ids, axis=1)
        sort_ids = np.argsort(part_vals, axis=1)
        top_ids[start:end] = np.take_along_axis(part_ids, sort_ids, axis=1).astype(np.int32)
        top_dists[start:end] = np.take_along_axis(part_vals, sort_ids, axis=1).astype(dtype, copy=False)
        print('TopK distance rows', str(end) + '/' + str(sample_num))
    return top_dists, top_ids


def build_sparse_topk_graph(args, all_feats: np.ndarray):
    sample_num = all_feats.shape[0]
    if args.k_num <= 0 or args.k_num >= sample_num:
        raise ValueError('k-num should be in [1, sample_num - 1].')
    if args.sparse_full_topk < args.k_num:
        raise ValueError('sparse-full-topk should be >= k-num.')
    dtype = np.float32 if args.dtype == 'float32' else np.float64
    full_top = min(args.sparse_full_topk + 1, sample_num)
    top_dists, top_ids = batch_topk_l2(all_feats, args.batch_size, dtype, full_top)
    scale_num = min(args.k_num + 1, full_top)
    local_sums = top_dists[:, 1:scale_num].sum(axis=1)
    wt_val = args.gamma / (2.0 * args.k_num)
    aff_mat = np.zeros((sample_num, sample_num), dtype=dtype)
    for row_idx in range(sample_num):
        ids = top_ids[row_idx]
        denom = wt_val * (local_sums[row_idx] + local_sums[ids])
        denom = np.maximum(denom, 1e-12)
        vals = np.exp(-np.square(top_dists[row_idx] / denom)).astype(dtype, copy=False)
        aff_mat[row_idx, ids] = vals
    knn_aff = np.zeros((sample_num, sample_num), dtype=dtype)
    directed_edges = set()
    knn_top = min(args.k_num + 1, full_top)
    for row_idx in range(sample_num):
        for col_idx in top_ids[row_idx, :knn_top].tolist():
            directed_edges.add(row_idx * sample_num + int(col_idx))
    for row_idx in range(sample_num):
        for col_idx in top_ids[row_idx, :knn_top].tolist():
            col_idx = int(col_idx)
            if col_idx * sample_num + row_idx in directed_edges:
                knn_aff[row_idx, col_idx] = aff_mat[row_idx, col_idx]
    if args.save_cache is not None:
        np.savez_compressed(args.save_cache, aff_mat=aff_mat, knn_aff=knn_aff,
                            graph_mode=args.graph_mode,
                            sparse_full_topk=args.sparse_full_topk,
                            k_num=args.k_num, gamma=args.gamma, dtype=args.dtype)
    return aff_mat, knn_aff


def load_cached_graph(args, sample_num: int):
    if args.cache_path is None:
        return None
    cache_data = np.load(args.cache_path)
    if args.cache_aff_key not in cache_data or args.cache_knn_key not in cache_data:
        raise KeyError('Cache should contain ' + args.cache_aff_key + ' and ' + args.cache_knn_key)
    aff_mat = np.asarray(cache_data[args.cache_aff_key])
    knn_aff = np.asarray(cache_data[args.cache_knn_key])
    if aff_mat.shape != (sample_num, sample_num) or knn_aff.shape != (sample_num, sample_num):
        raise ValueError('Cached graph shape does not match feature count.')
    return aff_mat, knn_aff


def build_or_load_graph(args, all_feats: np.ndarray):
    sample_num = all_feats.shape[0]
    cached = load_cached_graph(args, sample_num)
    if cached is not None:
        return cached
    if args.graph_mode == 'sparse_topk':
        return build_sparse_topk_graph(args, all_feats)
    if args.k_num <= 0 or args.k_num >= sample_num:
        raise ValueError('k-num should be in [1, sample_num - 1].')
    dtype = np.float32 if args.dtype == 'float32' else np.float64
    dist_mat = pairwise_l2_distance(all_feats, args.batch_size, dtype)
    asst = OperAsstNumpy()
    aff_mat = Dist2Aff(args.k_num, gamma=args.gamma, asst_rela=asst)(dist_mat)
    knn_aff = KnnRstct(args.k_num, asst_rela=asst)(aff_mat)
    aff_mat = np.asarray(aff_mat, dtype=dtype)
    knn_aff = np.asarray(knn_aff, dtype=dtype)
    if args.save_cache is not None:
        np.savez_compressed(args.save_cache, aff_mat=aff_mat, knn_aff=knn_aff)
    return aff_mat, knn_aff


def build_ranker(args, mdl_iput, fallback_similarity=None):
    if args.hyrdp_solver == 'closed':
        hyrdp_solver = HyRdpClosedForm()
    else:
        hyrdp_solver = HyRdpIterative()
    if args.method in ['hyrdp', 'ihyrdp']:
        base_ranker = HyRdpPaper(mdl_iput=mdl_iput, solver=hyrdp_solver,
                                 alpha=args.alpha, beta=args.beta,
                                 incl_qry=False, local_topk=args.local_topk,
                                 fallback_similarity=fallback_similarity)
    else:
        base_ranker = GmfptPaper(mdl_iput=mdl_iput, incl_qry=False,
                                 local_topk=args.local_topk,
                                 fallback_similarity=fallback_similarity)
    if args.method == 'hyrdp' or args.method == 'gmfpt':
        return base_ranker
    if args.topk is None or args.topk <= 0:
        raise ValueError('Iterative methods require topk > 0.')
    return IHyRdpPaper(base_ranker, args.topk, args.lambda_step) \
        if args.method == 'ihyrdp' else IGmfptPaper(base_ranker, args.topk, args.lambda_step)


def resolve_used_query_num(query_limit, query_num: int) -> int:
    if query_limit is None:
        return query_num
    used_query_num = min(query_limit, query_num)
    if used_query_num <= 0:
        raise ValueError('query-limit should be positive when provided.')
    return used_query_num


def run_ranker(args, ranker, query_num: int, sample_num: int):
    used_query_num = resolve_used_query_num(args.query_limit, query_num)
    queries = [[idx] for idx in range(used_query_num)]
    if args.topk is None:
        args.topk = sample_num
    args.topk = min(args.topk, sample_num)
    if args.method in ['hyrdp', 'gmfpt']:
        rankings = ranker(queries, args.topk)
    else:
        rankings = ranker(queries)
    return queries, rankings


def run_baseline(args, query_f: np.ndarray, gallery_f: np.ndarray):
    """Rank gallery images directly by normalized feature similarity."""
    query_num = query_f.shape[0]
    gallery_num = gallery_f.shape[0]
    used_query_num = resolve_used_query_num(args.query_limit, query_num)
    dtype = np.float32 if args.dtype == 'float32' else np.float64
    query_norm = l2_normalize(query_f[:used_query_num]).astype(dtype, copy=False)
    gallery_norm = l2_normalize(gallery_f).astype(dtype, copy=False)
    gallery_t = gallery_norm.T
    queries = [[idx] for idx in range(used_query_num)]
    rankings = list()
    for start in range(0, used_query_num, args.batch_size):
        end = min(start + args.batch_size, used_query_num)
        sim = query_norm[start:end].dot(gallery_t)
        order = np.argsort(-sim, axis=1)
        for row in order:
            rankings.append((row + query_num).astype(int).tolist())
        print('Baseline query rows', str(end) + '/' + str(used_query_num))
    args.topk = gallery_num
    return queries, rankings


def run_affinity(args, aff_mat: np.ndarray, query_num: int):
    """Isolate adaptive-kernel conversion from diffusion, using the same graph cache."""
    used_query_num = resolve_used_query_num(args.query_limit, query_num)
    length = aff_mat.shape[0] if args.topk is None else min(args.topk, aff_mat.shape[0])
    if length <= 0:
        raise ValueError('topk should be positive.')
    queries = [[idx] for idx in range(used_query_num)]
    rankings = [np.argsort(-aff_mat[idx], kind='stable')[:length].astype(np.int32).tolist()
                for idx in range(used_query_num)]
    args.topk = length
    return queries, rankings


def original_similarity_fallback(all_feats: np.ndarray):
    """Return original cosine scores without storing another dense distance matrix."""
    normalized = l2_normalize(all_feats).astype(np.float32, copy=False)

    def similarity(query_ids):
        scores = normalized[np.asarray(query_ids, dtype=int)].dot(normalized.T)
        return np.mean(scores, axis=0)

    return similarity


def collect_run_metadata(args):
    digest = hashlib.sha256()
    with open(args.mat_path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    try:
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo_root,
                                       text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo_root,
                                            text=True, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        head, dirty = None, None
    return {'feature_sha256': digest.hexdigest(), 'code_git_head': head,
            'code_git_dirty': dirty,
            'evaluation_protocol': 'Market1501: remove pid=-1 and same pid/same camera; retain pid=0'}


def print_summary(args, query_num: int, gallery_num: int, metrics: dict, rankings: list):
    print('Method:', args.method)
    print('Queries evaluated:', metrics['valid_queries'], '/', len(rankings))
    print('Gallery:', gallery_num)
    print('mAP:', round(metrics['mAP'] * 100.0, 2))
    if 'mINP' in metrics:
        print('mINP:', round(metrics['mINP'] * 100.0, 2))
    for key in ['rank-1', 'rank-5', 'rank-10', 'rank-20']:
        if key in metrics:
            print(key + ':', round(metrics[key] * 100.0, 2))
    for idx in range(min(args.preview, len(rankings))):
        print('Query', idx, 'ranking preview:', rankings[idx][:20])


def save_outputs(args, metrics: dict, rankings: list, query_num: int, gallery_num: int):
    if args.save_json is not None:
        out = {
            'method': args.method,
            'mat_path': args.mat_path,
            'query_num': query_num,
            'gallery_num': gallery_num,
            'topk': args.topk,
            'eval_max_rank': args.eval_max_rank,
            'query_limit': args.query_limit,
            'batch_size': args.batch_size,
            'dtype': args.dtype,
            'cache_path': args.cache_path,
            'save_cache': args.save_cache,
            'graph_mode': args.graph_mode,
            'sparse_full_topk': args.sparse_full_topk,
            'k_num': args.k_num,
            'gamma': args.gamma,
            'alpha': args.alpha,
            'beta': args.beta,
            'lambda_step': args.lambda_step,
            'local_topk': args.local_topk,
            'metrics': {key: value for key, value in metrics.items() if key != 'per_query'},
            'run_metadata': collect_run_metadata(args),
        }
        if getattr(args, 'experiment_diagnostics', None) is not None:
            out['solver_diagnostics'] = args.experiment_diagnostics
        if args.method == 'baseline':
            out['distance'] = 'cosine_on_l2_normalized_features'
        with open(args.save_json, 'w', encoding='utf-8') as file_obj:
            json.dump(out, file_obj, indent=2)
    if getattr(args, 'save_per_query_json', None) is not None:
        if 'per_query' not in metrics:
            raise ValueError('Per-query output requires evaluate_reid_rankings(return_per_query=True).')
        with open(args.save_per_query_json, 'w', encoding='utf-8') as file_obj:
            json.dump({'method': args.method, 'queries': metrics['per_query']}, file_obj, indent=2)
    if args.save_rankings is not None:
        lengths = {len(row) for row in rankings}
        if len(lengths) > 1:
            raise ValueError('save-rankings requires equal-length rows; object/pickle output is disabled.')
        np.save(args.save_rankings, np.asarray(rankings, dtype=np.int32), allow_pickle=False)


def main():
    args = parse_args()
    query_f, gallery_f, query_pids, query_camids, gallery_pids, gallery_camids = load_reid_data(args)
    query_num = query_f.shape[0]
    gallery_num = gallery_f.shape[0]
    if args.method == 'baseline':
        query_wrappers, rankings = run_baseline(args, query_f, gallery_f)
        query_ids = [tmp[0] for tmp in query_wrappers]
        metrics = evaluate_reid_rankings(rankings=rankings, query_ids=query_ids,
                                         query_pids=query_pids, query_camids=query_camids,
                                         gallery_pids=gallery_pids, gallery_camids=gallery_camids,
                                         gallery_offset=query_num, max_rank=args.eval_max_rank,
                                         return_per_query=args.save_per_query_json is not None)
        print_summary(args, query_num, gallery_num, metrics, rankings)
        save_outputs(args, metrics, rankings, query_num, gallery_num)
        return
    all_feats = np.vstack([query_f, gallery_f])
    if args.topk is None:
        args.topk = all_feats.shape[0]
    args.topk = min(args.topk, all_feats.shape[0])
    aff_mat, knn_aff = build_or_load_graph(args, all_feats)
    if args.method == 'affinity':
        query_wrappers, rankings = run_affinity(args, aff_mat, query_num)
    else:
        mdl_iput = IputWhl(smth_w=knn_aff, fit_w=knn_aff, full_w=aff_mat,
                           asst_rela=OperAsstNumpy())
        fallback_similarity = original_similarity_fallback(all_feats)
        ranker = build_ranker(args, mdl_iput, fallback_similarity=fallback_similarity)
        query_wrappers, rankings = run_ranker(args, ranker, query_num, all_feats.shape[0])
    query_ids = [tmp[0] for tmp in query_wrappers]
    metrics = evaluate_reid_rankings(rankings=rankings, query_ids=query_ids,
                                     query_pids=query_pids, query_camids=query_camids,
                                     gallery_pids=gallery_pids, gallery_camids=gallery_camids,
                                     gallery_offset=query_num, max_rank=args.eval_max_rank,
                                     return_per_query=args.save_per_query_json is not None)
    print_summary(args, query_num, gallery_num, metrics, rankings)
    save_outputs(args, metrics, rankings, query_num, gallery_num)


if __name__ == '__main__':
    main()
