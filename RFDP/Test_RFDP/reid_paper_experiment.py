#
import argparse
import json
import os
import sys
from typing import Optional

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
    parser.add_argument('--k-num', type=int, default=20,
                        help='K for distance-to-affinity and KNN graph construction.')
    parser.add_argument('--gamma', type=float, default=0.4,
                        help='Gamma for Dist2Aff.')
    parser.add_argument('--method', type=str, default='ihyrdp',
                        choices=['hyrdp', 'gmfpt', 'ihyrdp', 'igmfpt'])
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
                        help='Optional .npy path to save rankings as an object array.')
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


def build_ranker(args, mdl_iput):
    if args.hyrdp_solver == 'closed':
        hyrdp_solver = HyRdpClosedForm()
    else:
        hyrdp_solver = HyRdpIterative()
    if args.method in ['hyrdp', 'ihyrdp']:
        base_ranker = HyRdpPaper(mdl_iput=mdl_iput, solver=hyrdp_solver,
                                 alpha=args.alpha, beta=args.beta,
                                 incl_qry=False, local_topk=args.local_topk)
    else:
        base_ranker = GmfptPaper(mdl_iput=mdl_iput, incl_qry=False,
                                 local_topk=args.local_topk)
    if args.method == 'hyrdp' or args.method == 'gmfpt':
        return base_ranker
    if args.topk is None or args.topk <= 0:
        raise ValueError('Iterative methods require topk > 0.')
    return IHyRdpPaper(base_ranker, args.topk, args.lambda_step) \
        if args.method == 'ihyrdp' else IGmfptPaper(base_ranker, args.topk, args.lambda_step)


def run_ranker(args, ranker, query_num: int, sample_num: int):
    if args.query_limit is None:
        used_query_num = query_num
    else:
        used_query_num = min(args.query_limit, query_num)
    queries = [[idx] for idx in range(used_query_num)]
    if args.topk is None:
        args.topk = sample_num
    args.topk = min(args.topk, sample_num)
    if args.method in ['hyrdp', 'gmfpt']:
        rankings = ranker(queries, args.topk)
    else:
        rankings = ranker(queries)
    return queries, rankings


def print_summary(args, query_num: int, gallery_num: int, metrics: dict, rankings: list):
    print('Method:', args.method)
    print('Queries evaluated:', metrics['valid_queries'], '/', len(rankings))
    print('Gallery:', gallery_num)
    print('mAP:', round(metrics['mAP'] * 100.0, 2))
    for key in ['rank-1', 'rank-5', 'rank-10', 'rank-20']:
        if key in metrics:
            print(key + ':', round(metrics[key] * 100.0, 2))
    for idx in range(min(args.preview, len(rankings))):
        print('Query', idx, 'ranking preview:', rankings[idx][:20])


def save_outputs(args, metrics: dict, rankings: list):
    if args.save_json is not None:
        out = {
            'method': args.method,
            'topk': args.topk,
            'eval_max_rank': args.eval_max_rank,
            'k_num': args.k_num,
            'gamma': args.gamma,
            'alpha': args.alpha,
            'beta': args.beta,
            'lambda_step': args.lambda_step,
            'local_topk': args.local_topk,
            'metrics': metrics,
        }
        with open(args.save_json, 'w', encoding='utf-8') as file_obj:
            json.dump(out, file_obj, indent=2)
    if args.save_rankings is not None:
        np.save(args.save_rankings, np.asarray(rankings, dtype=object), allow_pickle=True)


def main():
    args = parse_args()
    query_f, gallery_f, query_pids, query_camids, gallery_pids, gallery_camids = load_reid_data(args)
    all_feats = np.vstack([query_f, gallery_f])
    query_num = query_f.shape[0]
    gallery_num = gallery_f.shape[0]
    if args.topk is None:
        args.topk = all_feats.shape[0]
    args.topk = min(args.topk, all_feats.shape[0])
    aff_mat, knn_aff = build_or_load_graph(args, all_feats)
    mdl_iput = IputWhl(smth_w=knn_aff, fit_w=knn_aff, full_w=aff_mat,
                       asst_rela=OperAsstNumpy())
    ranker = build_ranker(args, mdl_iput)
    query_wrappers, rankings = run_ranker(args, ranker, query_num, all_feats.shape[0])
    query_ids = [tmp[0] for tmp in query_wrappers]
    metrics = evaluate_reid_rankings(rankings=rankings, query_ids=query_ids,
                                     query_pids=query_pids, query_camids=query_camids,
                                     gallery_pids=gallery_pids, gallery_camids=gallery_camids,
                                     gallery_offset=query_num, max_rank=args.eval_max_rank)
    print_summary(args, query_num, gallery_num, metrics, rankings)
    save_outputs(args, metrics, rankings)


if __name__ == '__main__':
    main()
