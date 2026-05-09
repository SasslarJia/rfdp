#
import argparse
import json
import os
import sys
from typing import Optional

import numpy as np

#
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from EVAL.RefRslt import RefRslt
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
        description='End-to-end experiment runner for the paper-accurate RFDP path.'
    )
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to the input matrix file. Supports .mat/.npy/.npz.')
    parser.add_argument('--data-key', type=str, default='all_dists',
                        help='Key used when loading .mat/.npz input matrices.')
    parser.add_argument('--input-kind', type=str, default='distance',
                        choices=['distance', 'affinity', 'knn_affinity'],
                        help='Interpretation of the input matrix.')
    parser.add_argument('--label-path', type=str, default=None,
                        help='Optional label file for evaluation. Supports .mat/.npy/.npz/.json/.txt.')
    parser.add_argument('--label-key', type=str, default='labels',
                        help='Key used when loading .mat/.npz labels.')
    parser.add_argument('--cls-num', type=int, default=None,
                        help='Number of classes for uniform class splits.')
    parser.add_argument('--ele-num', type=int, default=None,
                        help='Number of samples per class for uniform class splits.')
    parser.add_argument('--query-limit', type=int, default=None,
                        help='Only evaluate the first N singleton queries.')
    parser.add_argument('--k-num', type=int, default=20,
                        help='K used by Dist2Aff and KnnRstct for real datasets.')
    parser.add_argument('--gamma', type=float, default=0.4,
                        help='Gamma used by Dist2Aff when input-kind=distance.')
    parser.add_argument('--method', type=str, default='ihyrdp',
                        choices=['hyrdp', 'gmfpt', 'ihyrdp', 'igmfpt'],
                        help='Paper method to run.')
    parser.add_argument('--hyrdp-solver', type=str, default='iterative',
                        choices=['iterative', 'closed'],
                        help='Solver used inside HyRdpPaper.')
    parser.add_argument('--alpha', type=float, default=5.0,
                        help='Alpha in HyRDP.')
    parser.add_argument('--beta', type=float, default=25.0,
                        help='Beta in HyRDP.')
    parser.add_argument('--topk', type=int, default=None,
                        help='Output length. For iterative methods this is used as nq.')
    parser.add_argument('--lambda-step', type=int, default=5,
                        help='Number of samples added per iterative round.')
    parser.add_argument('--local-topk', type=int, default=None,
                        help='Limit each paper solver call to a local candidate pool of this size.')
    parser.add_argument('--incl-qry', action='store_true',
                        help='Whether the query itself participates in local graph learning.')
    parser.add_argument('--preview', type=int, default=5,
                        help='Number of queries to preview in stdout.')
    parser.add_argument('--show-plot', action='store_true',
                        help='Display a precision-recall plot when references are available.')
    parser.add_argument('--save-plot', type=str, default=None,
                        help='Optional path to save the precision-recall curve.')
    parser.add_argument('--save-json', type=str, default=None,
                        help='Optional path to save rankings and metrics as JSON.')
    return parser.parse_args()


def load_array(path: str, key: Optional[str] = None) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path)
    if ext == '.npz':
        data = np.load(path)
        sel_key = choose_key(data, key)
        return data[sel_key]
    if ext == '.mat':
        try:
            import scipy.io as scio
        except ImportError as exc:
            raise ImportError('Loading .mat files requires scipy.') from exc
        data = scio.loadmat(path)
        sel_key = choose_key(data, key)
        return data[sel_key]
    if ext == '.txt':
        return np.loadtxt(path)
    if ext == '.json':
        with open(path, 'r', encoding='utf-8') as file_obj:
            return np.asarray(json.load(file_obj))
    raise ValueError('Unsupported file format: ' + ext)


def choose_key(data, key: Optional[str]) -> str:
    if key is not None and key in data:
        return key
    valid_keys = [tmp_key for tmp_key in data.keys() if not tmp_key.startswith('__')]
    if len(valid_keys) == 1:
        return valid_keys[0]
    if key is None:
        raise KeyError('Missing key for multi-array file. Available keys: ' + str(valid_keys))
    raise KeyError("Key '" + str(key) + "' not found. Available keys: " + str(valid_keys))


def build_synthetic_case():
    smth_w = np.array([
        [1.0, 0.9, 0.0, 0.0, 0.0],
        [0.9, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.85, 0.2],
        [0.0, 0.0, 0.85, 1.0, 0.3],
        [0.0, 0.0, 0.2, 0.3, 1.0]
    ], dtype=float)
    full_w = np.array([
        [1.0, 0.9, 0.10, 0.75, 0.20],
        [0.9, 1.0, 0.80, 0.20, 0.10],
        [0.10, 0.80, 1.0, 0.85, 0.2],
        [0.75, 0.20, 0.85, 1.0, 0.3],
        [0.20, 0.10, 0.2, 0.3, 1.0]
    ], dtype=float)
    labels = np.array([0, 0, 0, 1, 1], dtype=int)
    return smth_w, full_w, labels


def ensure_square(iput_mat: np.ndarray):
    if iput_mat.ndim != 2 or iput_mat.shape[0] != iput_mat.shape[1]:
        raise ValueError('The input matrix must be square.')


def build_graph_inputs(args):
    if args.data_path is None:
        smth_w, full_w, labels = build_synthetic_case()
        return smth_w, full_w, labels, 'synthetic'
    raw_mat = np.asarray(load_array(args.data_path, args.data_key), dtype=float)
    ensure_square(raw_mat)
    if raw_mat.shape[0] <= 1:
        raise ValueError('The input matrix must contain at least two samples.')
    if args.k_num <= 0 or args.k_num >= raw_mat.shape[0]:
        raise ValueError('k-num should be in [1, n-1] for real datasets.')
    asst = OperAsstNumpy()
    if args.input_kind == 'distance':
        full_w = Dist2Aff(args.k_num, gamma=args.gamma, asst_rela=asst)(raw_mat)
        smth_w = KnnRstct(args.k_num, asst_rela=asst)(full_w)
    elif args.input_kind == 'affinity':
        full_w = raw_mat
        smth_w = KnnRstct(args.k_num, asst_rela=asst)(full_w)
    else:
        smth_w = raw_mat
        full_w = raw_mat
    labels = load_labels(args, smth_w.shape[0])
    return np.asarray(smth_w, dtype=float), np.asarray(full_w, dtype=float), labels, 'dataset'


def load_labels(args, sample_num: int):
    if args.label_path is not None:
        labels = np.asarray(load_array(args.label_path, args.label_key)).reshape(-1)
        if labels.shape[0] != sample_num:
            raise ValueError('Label count does not match matrix size.')
        return labels.astype(int)
    if args.cls_num is not None and args.ele_num is not None:
        if args.cls_num * args.ele_num != sample_num:
            raise ValueError('cls-num * ele-num must match the matrix size.')
        return np.repeat(np.arange(args.cls_num, dtype=int), args.ele_num)
    return None


def build_queries(sample_num: int, query_limit: Optional[int] = None):
    queries = [[idx] for idx in range(sample_num)]
    if query_limit is not None:
        if query_limit <= 0:
            raise ValueError('query-limit should be greater than 0.')
        return queries[:query_limit]
    return queries


def build_refs_from_labels(queries: list, labels: np.ndarray):
    label_dict = dict()
    for idx, tmp_lb in enumerate(labels.tolist()):
        label_dict.setdefault(int(tmp_lb), list()).append(idx)
    ref_lst = list()
    for qry in queries:
        ref_lst.append(RefRslt(label_dict[int(labels[qry[0]])]))
    return ref_lst


def build_ranker(args, mdl_iput):
    if args.hyrdp_solver == 'closed':
        hyrdp_solver = HyRdpClosedForm()
    else:
        hyrdp_solver = HyRdpIterative()
    if args.method in ['hyrdp', 'ihyrdp']:
        base_ranker = HyRdpPaper(mdl_iput=mdl_iput, solver=hyrdp_solver,
                                 alpha=args.alpha, beta=args.beta,
                                 incl_qry=args.incl_qry, local_topk=args.local_topk)
    else:
        base_ranker = GmfptPaper(mdl_iput=mdl_iput, incl_qry=args.incl_qry,
                                 local_topk=args.local_topk)
    if args.method == 'hyrdp':
        return base_ranker
    if args.method == 'gmfpt':
        return base_ranker
    nq = args.topk
    if nq is None or nq <= 0:
        raise ValueError('Iterative methods require topk > 0.')
    if args.method == 'ihyrdp':
        return IHyRdpPaper(base_ranker=base_ranker, nq=nq, lambda_step=args.lambda_step)
    return IGmfptPaper(base_ranker=base_ranker, nq=nq, lambda_step=args.lambda_step)


def run_ranker(args, ranker, queries: list, sample_num: int):
    if args.topk is None:
        topk = sample_num
    else:
        topk = min(args.topk, sample_num)
    if args.method in ['hyrdp', 'gmfpt']:
        return ranker(queries, topk), topk
    args.topk = topk
    return ranker(queries), topk


def eval_at_k(rankings: list, refs: list, k: int):
    corr_num = 0.0
    prcs_num = 0.0
    rcal_num = 0.0
    for idx, rank in enumerate(rankings):
        top_ids = rank[:k]
        ref_ids = set(refs[idx].rslt_ids)
        hit_num = sum(1 for tmp_id in top_ids if tmp_id in ref_ids)
        corr_num += hit_num
        prcs_num += hit_num / max(len(top_ids), 1)
        rcal_num += hit_num / max(len(ref_ids), 1)
    qry_num = max(len(rankings), 1)
    return {
        'micro_hits': corr_num,
        'mean_precision': prcs_num / qry_num,
        'mean_recall': rcal_num / qry_num,
    }


def build_pr_curve(rankings: list, refs: list):
    max_len = max(len(tmp) for tmp in rankings)
    prcs_lst = list()
    rcal_lst = list()
    for depth in range(1, max_len + 1):
        tmp_eval = eval_at_k(rankings, refs, depth)
        prcs_lst.append(tmp_eval['mean_precision'])
        rcal_lst.append(tmp_eval['mean_recall'])
    return prcs_lst, rcal_lst


def summarize_metrics(rankings: list, refs: list):
    if refs is None:
        return None
    rank_len = max(len(tmp) for tmp in rankings)
    ks = [tmp_k for tmp_k in [1, 5, 10, rank_len] if tmp_k <= rank_len]
    metrics = dict()
    for tmp_k in ks:
        tmp_eval = eval_at_k(rankings, refs, tmp_k)
        metrics['P@' + str(tmp_k)] = tmp_eval['mean_precision']
        metrics['R@' + str(tmp_k)] = tmp_eval['mean_recall']
    return metrics


def maybe_plot_curve(prcs_lst: list, rcal_lst: list, args):
    if not args.show_plot and args.save_plot is None:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError('Plotting requires matplotlib.') from exc
    plt.figure()
    plt.plot(rcal_lst, prcs_lst, 'bx-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Paper RFDP Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    if args.save_plot is not None:
        plt.savefig(args.save_plot, bbox_inches='tight')
    if args.show_plot:
        plt.show()
    else:
        plt.close()


def print_summary(mode: str, args, sample_num: int, topk: int, rankings: list, metrics, refs):
    print('Mode:', mode)
    print('Method:', args.method)
    print('Samples:', sample_num)
    print('Queries:', len(rankings))
    print('TopK:', topk)
    print('Input kind:', 'synthetic_affinity' if mode == 'synthetic' else args.input_kind)
    if refs is None:
        print('Evaluation: skipped (no labels or class metadata provided)')
    else:
        for key in sorted(metrics.keys()):
            print(key + ':', round(metrics[key], 6))
    prev_num = min(args.preview, len(rankings))
    for idx in range(prev_num):
        print('Query', idx, '->', rankings[idx])


def save_json(args, mode: str, rankings: list, metrics, prcs_lst, rcal_lst):
    if args.save_json is None:
        return
    oput_dict = {
        'mode': mode,
        'method': args.method,
        'input_kind': args.input_kind if mode != 'synthetic' else 'synthetic_affinity',
        'topk': args.topk,
        'alpha': args.alpha,
        'beta': args.beta,
        'lambda_step': args.lambda_step,
        'local_topk': args.local_topk,
        'metrics': metrics,
        'precision_curve': prcs_lst,
        'recall_curve': rcal_lst,
        'rankings': rankings,
    }
    with open(args.save_json, 'w', encoding='utf-8') as file_obj:
        json.dump(oput_dict, file_obj, indent=2)


def main():
    args = parse_args()
    smth_w, full_w, labels, mode = build_graph_inputs(args)
    sample_num = smth_w.shape[0]
    queries = build_queries(sample_num, args.query_limit)
    refs = None if labels is None else build_refs_from_labels(queries, labels)
    if args.topk is None:
        args.topk = sample_num if args.method in ['ihyrdp', 'igmfpt'] else sample_num
    mdl_iput = IputWhl(smth_w=smth_w, fit_w=smth_w, full_w=full_w, asst_rela=OperAsstNumpy())
    ranker = build_ranker(args, mdl_iput)
    rankings, topk = run_ranker(args, ranker, queries, sample_num)
    metrics = summarize_metrics(rankings, refs)
    if refs is None:
        prcs_lst, rcal_lst = None, None
    else:
        prcs_lst, rcal_lst = build_pr_curve(rankings, refs)
        maybe_plot_curve(prcs_lst, rcal_lst, args)
    print_summary(mode, args, sample_num, topk, rankings, metrics, refs)
    save_json(args, mode, rankings, metrics, prcs_lst, rcal_lst)


if __name__ == '__main__':
    main()
