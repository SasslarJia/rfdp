#!/usr/bin/env python3
"""Frozen-feature ECN/GCR comparison with unchanged pinned official source."""
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat, savemat
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE / 'research' / 'classic'
METHODS = ('ecn-rankdist', 'ecn-origdist', 'gcr-global')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def normalize_float32(features):
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Expected finite row feature matrix.')
    return (values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def source_record(method):
    family = 'ecn' if method.startswith('ecn-') else 'gcr'
    record = json.loads((SOURCE_ROOT / 'upstream_metadata.json').read_text())[family]
    filename = 'ecn.py' if family == 'ecn' else 'gcrv.py'
    path = SOURCE_ROOT / family / filename
    if sha256(path) != record['files'][filename]:
        raise RuntimeError('Official source hash mismatch: ' + str(path))
    return family, path, record


@contextlib.contextmanager
def official_import_path(path):
    """Avoid the GCR utils package colliding with unrelated repo packages."""
    old_path = list(sys.path)
    removed = {k: v for k, v in sys.modules.items() if k == 'utils' or k.startswith('utils.')}
    for k in removed:
        del sys.modules[k]
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path[:] = old_path
        for k in list(sys.modules):
            if k == 'utils' or k.startswith('utils.'):
                del sys.modules[k]
        sys.modules.update(removed)


def load_official(method):
    family, source, record = source_record(method)
    name = 'pinned_official_' + family
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    with official_import_path(source.parent):
        spec.loader.exec_module(module)
    return module, source, record


def gcr_config(verbose=False):
    return SimpleNamespace(
        COMMON=SimpleNamespace(VERBOSE=verbose, DATASET='market'),
        PVG=SimpleNamespace(ENABLE_PVG=False),
        GCR=SimpleNamespace(ENABLE_GCR=True, BETA1=0.2, BETA2=0.2,
                            K1=15, K2=3, SCALE=0.0, GAL_ROUND=3,
                            MODE='nonsym', WITH_GPU=False))


def algorithm_distances(method, query, gallery, q_cams, g_cams, module=None):
    if module is None:
        module, _, _ = load_official(method)
    if method.startswith('ecn-'):
        # Official function consumes features and returns gallery x query.
        return module.ECN(query, gallery, k=25, t=3, q=8,
                          method=method.split('-', 1)[1]).T
    # Identity labels are deliberately zero: official global GCR must never see PIDs.
    q_labels = np.column_stack((np.zeros(len(query), dtype=np.int64), q_cams))
    g_labels = np.column_stack((np.zeros(len(gallery), dtype=np.int64), g_cams))
    new_q, new_g = module.run_gcr(
        gcr_config(), [query.copy(), q_labels, None, gallery.copy(), g_labels, None])
    # Original features are L2; GCR renormalizes each round. Rank by final cosine.
    return 1.0 - new_q.dot(new_g.T)


def rankings_from_distances(distances, nq, ng):
    distances = np.asarray(distances)
    if distances.shape != (nq, ng) or not np.isfinite(distances).all():
        raise RuntimeError('Nonfinite or wrong-shape official reranker output.')
    return np.argsort(distances, axis=1, kind='stable').astype(np.int32) + nq


def independent_global_gcr(features):
    """Independent dense equation check, not used for actual benchmark runs."""
    x = features.copy()
    for _ in range(3):
        sim = x @ x.T
        order = np.argsort(-sim, axis=1)
        a = np.zeros((len(x), len(x)), dtype=np.float64)
        for i in range(len(x)):
            ids = order[i, :15]
            a[i, ids] = np.exp(sim[i, ids] / .2)
            a[i, i] = np.exp(sim[i, i] / .2)
        row = 1.0 / np.sqrt(a.sum(axis=1))
        col = 1.0 / np.sqrt(a.sum(axis=0))
        x = ((row[:, None] * a) * col[None, :]).dot(x)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def cpu_function_self_check():
    """Run unchanged AST-extracted pure-NumPy official functions without imports.

    This does not claim that sklearn/torch/PIL imports have been validated.
    Full --self-check additionally imports official modules normally.
    """
    rng = np.random.default_rng(20261011)
    x = normalize_float32(rng.normal(size=(43, 16)))
    _, path, _ = source_record('gcr-global')
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('mergesetfeat4', 'run_gcr')]
    namespace = {'np': np, 'time': time}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    cfg = gcr_config()
    labels = np.column_stack((np.zeros(43, dtype=int), np.arange(43) % 6))
    official_q, official_g = namespace['run_gcr'](cfg, [x[:11], labels[:11], None, x[11:], labels[11:], None])
    actual = np.vstack((official_q, official_g))
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, independent_global_gcr(x), rtol=1e-12, atol=1e-12)
    changed = labels.copy(); changed[:, 1] = 0
    q2, g2 = namespace['run_gcr'](cfg, [x[:11], changed[:11], None, x[11:], changed[11:], None])
    np.testing.assert_array_equal(actual, np.vstack((q2, g2)))
    _, path, _ = source_record('ecn-rankdist')
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'rankdist']
    namespace = {'np': np, 'csr_matrix': csr_matrix}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    order = np.argsort(1.0 - x @ x.T, axis=1).astype(np.int32)
    weights = np.maximum(0, 25 - np.argsort(order, axis=1).astype(np.int32))
    np.testing.assert_array_equal(namespace['rankdist'](order, 25), -(weights @ weights.T))
    rankings = rankings_from_distances(1.0 - x[:11] @ x[11:].T, 11, 32)
    np.testing.assert_array_equal(np.sort(rankings, axis=1), np.broadcast_to(np.arange(11, 43), rankings.shape))
    return {'gcr_independent_dense_equations': 'passed',
            'gcr_camera_invariance_scale_zero': 'passed',
            'ecn_rankdist_independent_integer_equations': 'passed',
            'global_id_rankings_dtype': str(rankings.dtype),
            'dependency_imports_validated': False}


def self_check(full=False):
    result = cpu_function_self_check()
    if full:
        rng = np.random.default_rng(20261011)
        x = normalize_float32(rng.normal(size=(43, 16)))
        cams = np.arange(43) % 6
        for method in METHODS:
            module, _, _ = load_official(method)
            distances = algorithm_distances(method, x[:11], x[11:], cams[:11], cams[11:], module)
            rankings = rankings_from_distances(distances, 11, 32)
            np.testing.assert_array_equal(np.sort(rankings, axis=1), np.broadcast_to(np.arange(11, 43), rankings.shape))
            if method == 'gcr-global':
                changed = algorithm_distances(method, x[:11], x[11:], np.zeros(11, dtype=int), np.zeros(32, dtype=int), module)
                np.testing.assert_array_equal(distances, changed)
            result[method + '_official_module_smoke'] = 'passed'
        result['dependency_imports_validated'] = True
    print(json.dumps(result, indent=2))
    return result


def run(args):
    sys.path.insert(0, str(args.repo_root / 'RFDP'))
    from EVAL.ReIDEval import evaluate_reid_rankings
    started = time.perf_counter()
    data = loadmat(args.mat_path)
    q = normalize_float32(data['query_f']); g = normalize_float32(data['gallery_f'])
    q_cams = np.asarray(data['query_cam']).reshape(-1)
    g_cams = np.asarray(data['gallery_cam']).reshape(-1)
    if len(q) != len(q_cams) or len(g) != len(g_cams) or q.shape[1] != g.shape[1]:
        raise ValueError('MAT feature/camera dimensions mismatch.')
    module, source, upstream = load_official(args.method)
    print('Starting', args.method, 'query', len(q), 'gallery', len(g), flush=True)
    t = time.perf_counter()
    distances = algorithm_distances(args.method, q, g, q_cams, g_cams, module)
    rerank_seconds = time.perf_counter() - t
    del q, g
    t = time.perf_counter()
    nq, ng = len(q_cams), len(g_cams)
    rankings = rankings_from_distances(distances, nq, ng)
    del distances
    ranking_seconds = time.perf_counter() - t
    t = time.perf_counter()
    metrics = evaluate_reid_rankings(rankings, list(range(nq)), data['query_label'], q_cams,
                                     data['gallery_label'], g_cams, nq,
                                     max_rank=args.eval_max_rank, return_per_query=True)
    evaluation_seconds = time.perf_counter() - t
    per_query = metrics.pop('per_query')
    try:
        head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.repo_root,text=True,stderr=subprocess.DEVNULL,timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        head = None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    parameters = ({'k':25,'t':3,'q':8,'distance_method':args.method.split('-',1)[1]}
                  if args.method.startswith('ecn-') else
                  {'PVG':False,'scale':0.0,'k1':15,'k2':3,'beta1':.2,'beta2':.2,
                   'rounds':3,'mode':'nonsym','WITH_GPU':False})
    output = {'method':args.method,'query_num':nq,'gallery_num':ng,'metrics':metrics,
              'parameters':parameters,'feature_sha256':sha256(args.mat_path),'code_git_head':head,
              'upstream_commit':upstream['commit'],'upstream_url':upstream['remote'],
              'upstream_source_sha256':sha256(source),'runner_sha256':sha256(__file__),
              'graph_population':'all query + all gallery; pid=-1 retained for graph, removed in evaluation',
              'feature_input':'float64 L2 normalization then float32 row features',
              'algorithm_receives_identity_labels':False,
              'camera_metadata_used':False,
              'ranking_convention':'complete global gallery IDs, query-count offset, int32, stable distance argsort',
              'timings_seconds':{'rerank':rerank_seconds,'ranking':ranking_seconds,
                                 'evaluation':evaluation_seconds,'total_before_save':time.perf_counter()-started},
              'process_max_rss_bytes':int(rss if sys.platform=='darwin' else rss*1024),
              'numpy_version':np.__version__}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/(args.method+'.json')).write_text(json.dumps(output,indent=2)+'\n')
    (args.output_dir/(args.method+'.per-query.json')).write_text(json.dumps({'method':args.method,'queries':per_query},indent=2)+'\n')
    np.save(args.output_dir/(args.method+'.rankings.npy'),rankings,allow_pickle=False)
    print(json.dumps(metrics,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--repo-root',type=Path)
    parser.add_argument('--mat-path',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--eval-max-rank',type=int,default=50)
    parser.add_argument('--self-check',action='store_true',help='Full actual official-module random smoke (requires sklearn,torch,PIL).')
    parser.add_argument('--cpu-function-self-check',action='store_true',help='AST-extracted CPU equation checks; dependency imports are not validated.')
    args = parser.parse_args()
    if args.self_check or args.cpu_function_self_check:
        self_check(full=args.self_check)
        return
    if not all((args.method,args.repo_root,args.mat_path,args.output_dir)):
        parser.error('A benchmark run requires --method, --repo-root, --mat-path, --output-dir.')
    run(args)


if __name__=='__main__':
    main()
