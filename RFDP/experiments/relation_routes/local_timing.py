"""Explicitly authorized one-worker timing on full frozen inputs.

One complete query per route, including depth-500 dynamic discovery and the full
gallery permutation. Graph preparation, scoring and saved-score readback have
separate clocks. No identity labels or population accuracy metrics are loaded.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import gzip
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy import sparse
from MODEL.Paper_RFDP.PaperFullIterative import FullIHyRDP
from experiments.continuous_fitting import certified
from experiments.continuous_fitting.verify import verify_query, read_rows
from experiments.full_reid.common import complete_gallery

ROUTES = ('original', 'csw', 'csw_grouped', 'camera_channel', 'th_target', 'quaternary')
REPO = Path(__file__).resolve().parents[3]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def boundary_matrix(graph, indices, targets, bridges):
    """Effective U-to-L affinity, including only this round's temporary bridges."""
    result = graph[indices][:, list(targets)].tolil()
    row_of = {int(node): i for i, node in enumerate(indices)}
    col_of = {int(node): i for i, node in enumerate(targets)}
    for outside, inside, weight in bridges:
        result[row_of[int(outside)], col_of[int(inside)]] += weight
    return result.tocsr()


def prepare(route, graph, affinity, cameras, cfg):
    start = time.perf_counter()
    metadata = {}
    kernel = None
    source = route
    if route in ('csw', 'csw_grouped'):
        from experiments.reliability.reliability_model import build_support, ReliabilityIHyRDP
        from experiments.stationary_measure.csw_graph import ReadbackKernel
        if route == 'csw_grouped':
            from .csw_grouped import build_grouped_csw_inputs
            base, support, metadata = build_grouped_csw_inputs(
                affinity, cameras, k_intra=cfg['k_intra'], k_inter=cfg['k_inter'])
        else:
            base = graph
            support, metadata = build_support(affinity, cameras, k=20, mode='ordinary')
        support_seconds = time.perf_counter() - start
        core = ReliabilityIHyRDP(base, affinity, cameras, eta=cfg['rho'],
            support=support, lam=cfg['support_lambda'], alpha=5., beta=25., step=5)
        reader = ReadbackKernel(base, affinity, cameras, support, cfg['rho'], cfg['support_lambda'])
        assert (core.w - reader.graph).nnz == 0
        assert np.array_equal(core.full_sums, reader.sums)
        kernel = reader.block
        graph = reader.graph
        sums = reader.sums
        source = 'original'
        metadata = dict(support=metadata, support_preparation_seconds=support_seconds)
    else:
        core = FullIHyRDP(graph, affinity, alpha=5., beta=25., step=5)
        sums = core.full_sums
    return core, graph, sums, kernel, source, metadata, time.perf_counter() - start


def execute(route, query, graph, affinity, features, cameras, cfg, dest):
    dest.mkdir(parents=True, exist_ok=False)
    core, graph, sums, kernel, source, metadata, setup = prepare(route, graph, affinity, cameras, cfg)
    dump(dest/'preparation.json', dict(seconds=setup, metadata=metadata))
    events = []
    stage = 0
    pair_states = dest/'pair-states'
    if source == 'quaternary':
        pair_states.mkdir()

    def solve(w, e, alpha, beta, *, diagnostics, indices, targets, bridges):
        nonlocal stage
        if source == 'original':
            scores, residual, event = certified.solve(w, e, alpha, beta, source='original')
        elif source in ('camera_channel', 'th_target'):
            from .smoothing import solve as equation_solve
            scores, residual, event = equation_solve(w, e, alpha, beta, source=source,
                cameras=cameras[indices], target_count=len(targets), theta=cfg['theta'])
        else:
            from .quaternary import solve_quaternary_sparse
            statistics = {}
            before = time.perf_counter()
            raw, residual, relations = solve_quaternary_sparse(w,
                boundary_matrix(graph, indices, targets, bridges), graph[list(targets)][:, list(targets)],
                diffusion=cfg['diffusion'], diagnostics=statistics, return_relations=True,
                require_rank_certificate=True)
            scores = np.asarray([certified.mp.mpf(float(v)) for v in raw], dtype=object)
            state = pair_states/f'{stage:04d}.npy'
            np.save(state, relations)
            event = dict(statistics, source=source, size=len(e),
                returned_prefix_local=sorted(range(len(raw)), key=lambda i:(raw[i],i))[:5],
                returned_scores_sha256=certified.digest_scores(scores),
                returned_residual_upper=float(residual),
                relation_state_sha256=sha(state), seconds=time.perf_counter()-before)
        events.append(event)
        stage += 1
        # Preserve native event counts without guessing absent statistics.
        diagnostics.update(cg_calls=event.get('cg_calls',int(event.get('candidate_method') == 'cg')),
                           cg_iterations=event.get('cg_iterations', 0),
                           direct_calls=int(event.get('candidate_method') in ('sparse-lu', 'sparse-rank-one-lu')),
                           uncertified_cg=0)
        return scores, residual

    certify_stage = 0

    def certify(w, e, values, event, *, alpha, beta, source, indices, targets, bridges):
        nonlocal certify_stage
        if source == 'original':
            result = certified.certify_saved(w, e, values, alpha, beta, source='original')
        elif source in ('camera_channel', 'th_target'):
            from .smoothing import certify_saved
            result = certify_saved(w, e, values, alpha=alpha, beta=beta, source=source,
                cameras=cameras[indices], target_count=len(targets), theta=cfg['theta'])
            assert result['input_sha256'] == event['input_sha256']
        else:
            from .quaternary import certify_saved
            state = pair_states/f'{certify_stage:04d}.npy'
            assert sha(state) == event['relation_state_sha256']
            result = certify_saved(w, boundary_matrix(graph, indices, targets, bridges),
                graph[list(targets)][:, list(targets)], np.load(state), np.asarray(values,dtype=float),
                diffusion=cfg['diffusion'], require_rank_certificate=True)
            result = dict(result, prefix=sorted(range(len(values)), key=lambda i:(values[i],i))[:5],
                          relative=result['relative_residual'])
        certify_stage += 1
        return result

    core.stage_solver = solve
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    similarity = features @ features[query]
    trace_path = dest/'trace.jsonl.gz'
    seen = 0
    with gzip.open(trace_path, 'wt', encoding='utf-8', compresslevel=6) as stream:
        def capture(row):
            nonlocal seen
            event = None
            if row['scores'] is not None:
                assert len(events) == seen+1
                event = events[seen]
                seen += 1
                expected = np.asarray(row['unlabeled'])[event['returned_prefix_local']][:500-len(row['labeled'])]
                assert expected.tolist() == row['selected']
            else:
                assert len(events) == seen
            out = {k: row[k] for k in ('iteration', 'labeled', 'direct', 'indirect', 'unlabeled',
                                        'selected', 'bridges', 'residual')}
            out['returned_scores'] = None if row['scores'] is None else [certified.packed(v) for v in row['scores']]
            out['event'] = event
            stream.write(json.dumps(out, separators=(',', ':'), allow_nan=False)+'\n')
            if (row['iteration']+1) % 10 == 0:
                print(json.dumps(dict(route=route, query=query, stages=row['iteration']+1,
                    seconds=time.perf_counter()-wall_start)), flush=True)
        prefix, diagnostics = core.rank([query], nq=500, input_similarity=similarity,
                                       numerical_tail=False, trace=capture)
    assert diagnostics['status'] == 'complete_diffusion' and diagnostics['numerical_tail_count'] == 0
    rank = complete_gallery(prefix, similarity, cfg['query_count'])
    np.save(dest/'prefix.npy', prefix)
    np.save(dest/'rankings.npy', rank)
    compute = time.perf_counter() - wall_start
    readback_start = time.perf_counter()
    acceptance = verify_query(graph, affinity, similarity, query, np.load(dest/'prefix.npy'),
        np.load(dest/'rankings.npy'), read_rows(trace_path), source, cfg['query_count'], 500,
        kernel=kernel, full_sums=sums, stage_certifier=certify)
    readback = time.perf_counter() - readback_start
    assert rank.shape == (cfg['gallery_count'],)
    item = dict(route=route, source=source, query=query, setup_seconds=setup,
        compute_seconds=compute, readback_seconds=readback,
        compute_and_readback_seconds=time.perf_counter()-wall_start,
        cpu_seconds=time.process_time()-cpu_start, full_query_accepted=True,
        diagnostics=diagnostics, acceptance=acceptance,
        score_certificate_precisions=sorted(set(event.get('dps', 0) for event in events)),
        source_event_counts={key:sum(bool(e.get(key)) for e in events) for key in ('refinements',)},
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        peak_rss_scope='process-lifetime high-water mark; not per-route memory',
        files_sha256={p.name:sha(p) for p in (dest/'prefix.npy', dest/'rankings.npy', trace_path)},
        preparation=metadata)
    if route == 'original' and cfg.get('baseline_prefix'):
        item['archived_baseline_prefix_identical'] = bool(np.array_equal(prefix, np.load(cfg['baseline_prefix'])))
        assert item['archived_baseline_prefix_identical']
    dump(dest/'acceptance.json', acceptance)
    dump(dest/'timing.json', item)
    return item


def run(config):
    cfg = json.loads(Path(config).read_text())
    routes = cfg['routes']
    if not routes or len(set(routes)) != len(routes) or any(r not in ROUTES for r in routes):
        raise ValueError('Explicit unique supported routes required')
    if cfg['parameters'] != dict(alpha=5, beta=25, step=5, depth=500):
        raise ValueError('This paired protocol fixes the original parameters')
    cache = Path(cfg['cache'])
    expected = cfg['cache_sha256']
    assert set(expected) == {'features.npy', 'affinity.npy', 'graph.npz', 'cameras.npy'}
    for name, digest in expected.items():
        if sha(cache/name) != digest:
            raise ValueError('Frozen input SHA mismatch: '+name)
    output = Path(cfg['output']);output.mkdir(parents=True, exist_ok=False)
    graph = sparse.load_npz(cache/'graph.npz').astype(np.float64).tocsr()
    graph.eliminate_zeros();graph.sort_indices()
    affinity = np.load(cache/'affinity.npy', mmap_mode='r')
    features = np.load(cache/'features.npy', mmap_mode='r')
    cameras = np.load(cache/'cameras.npy')
    n = cfg['query_count']+cfg['gallery_count']
    assert graph.shape == affinity.shape == (n,n) and features.shape[0] == len(cameras) == n
    query = cfg['query_id']
    assert type(query) is int and 0 <= query < cfg['query_count']
    files = list((REPO/'RFDP/experiments/relation_routes').glob('*.py'))
    files += [REPO/p for p in ('RFDP/MODEL/Paper_RFDP/PaperFullIterative.py',
        'RFDP/experiments/continuous_fitting/verify.py', 'RFDP/experiments/continuous_fitting/certified.py',
        'RFDP/experiments/reliability/reliability_model.py', 'RFDP/experiments/stationary_measure/csw_graph.py')]
    report = dict(scope='complete single-query runtime; no population accuracy', config=cfg,
        commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO,text=True).strip(),
        worktree_changes=subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True),
        source_sha256={str(p.relative_to(REPO)):sha(p) for p in files},
        platform=platform.platform(), python=sys.version, numpy=np.__version__,workers=1,blas_threads=1,
        runs=[],failures=[])
    dump(output/'timing.json',report)
    for route in routes:
        try:
            item = execute(route,query,graph,affinity,features,cameras,cfg,output/route)
            reference = 'csw' if route == 'csw_grouped' else 'original'
            base = next((r for r in report['runs'] if r['route'] == reference),None)
            if base:
                item['reference_route'] = reference
                item['paired_compute_ratio'] = item['compute_seconds']/base['compute_seconds']
                item['paired_total_ratio'] = item['compute_and_readback_seconds']/base['compute_and_readback_seconds']
            report['runs'].append(item)
            print(json.dumps({k:v for k,v in item.items() if k not in ('diagnostics','acceptance','preparation','files_sha256')},ensure_ascii=False),flush=True)
        except Exception:
            failure = dict(route=route,error=traceback.format_exc())
            report['failures'].append(failure)
            dump(output/route/'FAILED.json',failure)
            print(json.dumps(failure),flush=True)
        dump(output/'timing.json',report)
    report['status'] = 'completed' if not report['failures'] else 'completed_with_recorded_failures'
    dump(output/'timing.json',report)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    run(parser.parse_args().config)
