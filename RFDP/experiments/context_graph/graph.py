"""Label-free context calibration; original topology, diagonal and dense kernel.

The placebo permutes FINAL undirected weights within original-affinity deciles.
This preserves their exact multiset, unlike permuting Q with a fixed mass scale.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from scipy import sparse
from scipy.special import logsumexp

BUILDER_ID = 'context-calibration-mass-preserving-v1'
SEED = 20261116
BINS = 10


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*2**20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def canonical(graph):
    g = sparse.csr_matrix(graph, dtype=np.float64, copy=True)
    g.sum_duplicates(); g.eliminate_zeros(); g.sort_indices()
    if g.shape[0] != g.shape[1] or not np.isfinite(g.data).all() or np.any(g.data <= 0):
        raise ValueError('A finite positive symmetric weighted graph is required')
    if (g-g.T).nnz or np.any(g.diagonal() <= 0):
        raise ValueError('Preserve original symmetric graph and positive self loops')
    return g


def edges(graph):
    upper = sparse.triu(graph, 1).tocoo()
    order = np.lexsort((upper.col, upper.row))
    return upper.row[order], upper.col[order], upper.data[order]


def assemble(original, i, j, values):
    diagonal=np.arange(original.shape[0])
    g = sparse.csr_matrix((np.r_[values,values,original.diagonal()],
        (np.r_[i,j,diagonal],np.r_[j,i,diagonal])),shape=original.shape)
    g.sort_indices()
    if (not np.array_equal(g.indices,original.indices) or
        not np.array_equal(g.indptr,original.indptr) or
        not np.array_equal(g.diagonal(),original.diagonal()) or
        not np.isfinite(g.data).all() or np.any(g.data<=0)):
        raise ValueError('Calibration changed positive support, diagonal or finiteness')
    return g


def calibrate(graph):
    """One unsupervised rule; no labels, cameras, fitting or retrieval inputs."""
    g = canonical(graph); i,j,a = edges(g)
    b = g.copy(); b.setdiag(0); b.eliminate_zeros(); b.data = np.ones(b.nnz,dtype=np.int32)
    degree = np.asarray(b.sum(axis=1)).ravel()
    common_matrix = (b@b).tocsr()
    common = np.asarray(common_matrix[i,j]).ravel() if len(a) else np.empty(0,dtype=np.int64)
    union = degree[i]+degree[j]-common-2
    if np.any(union<common) or np.any(common<0):raise ValueError('Invalid endpoint-removed counts')
    q = (common+1.)/(union+2.)
    la,lq = np.log(a),np.log(q)
    degenerate = ('no_nonself_edges' if not len(a) else
                  'constant_context' if np.ptp(lq)==0 else
                  'constant_affinity' if np.ptp(la)==0 else None)
    if degenerate:
        kappa,scale,values = 0.,1.,a.copy()
    else:
        kappa = float(np.std(la,ddof=0)/np.std(lq,ddof=0))
        logscale = float(logsumexp(la)-logsumexp(la+kappa*lq))
        scale = float(np.exp(logscale)); values = np.exp(la+kappa*lq+logscale)
    if not np.isfinite(kappa) or not np.isfinite(scale) or scale<=0:
        raise ValueError('Nonfinite calibration; do not clip or silently change the rule')
    result = assemble(g,i,j,values)
    mass = float(a.sum()); newmass = float(values.sum())
    if mass and not np.isclose(newmass,mass,rtol=5e-14,atol=0):raise ValueError('Nonself mass differs')
    info = dict(kappa=kappa,scale=scale,degenerate=degenerate,nodes=g.shape[0],
        undirected_nonself_edges=len(a),nnz=result.nnz,log_affinity_std=float(np.std(la)) if len(a) else 0.,
        log_context_std=float(np.std(lq)) if len(a) else 0.,undirected_nonself_mass_before=mass,
        undirected_nonself_mass_after=newmass,topology_unchanged=True,diagonal_unchanged=True,
        graph_dtype='float64',labels_used=False,cameras_used=False)
    return result,info


def permute_weights(original, calibrated, *, seed=SEED, bins=BINS):
    original,calibrated = canonical(original),canonical(calibrated)
    i,j,a = edges(original); x,y,weights = edges(calibrated)
    if not np.array_equal(i,x) or not np.array_equal(j,y):raise ValueError('Placebo support differs')
    if not np.array_equal(original.diagonal(),calibrated.diagonal()):raise ValueError('Placebo diagonal differs')
    if type(seed) is not int or type(bins) is not int or bins<1:raise ValueError('Invalid fixed permutation')
    cuts = np.quantile(a,np.arange(1,bins)/bins,method='linear') if len(a) else np.array([])
    groups = np.searchsorted(cuts,a,side='right')
    order = np.arange(len(a)); rng = np.random.Generator(np.random.PCG64(seed))
    for group in range(bins):
        ids=np.flatnonzero(groups==group); order[ids]=rng.permutation(ids)
    values=weights[order]
    if not np.array_equal(np.sort(values),np.sort(weights)):raise ValueError('Placebo weight multiset differs')
    result=assemble(original,i,j,values)
    for group in range(bins):
        ids=np.flatnonzero(groups==group)
        if not np.array_equal(np.sort(values[ids]),np.sort(weights[ids])):raise ValueError('Stratum multiset differs')
    return result,dict(seed=seed,bins=bins,generator='PCG64',quantile_method='linear',
        boundary_tie_rule='searchsorted-right',canonical_order='undirected i<j, lexicographic i,j',
        bin_counts=np.bincount(groups,minlength=bins).tolist(),
        exact_global_weight_multiset=True,exact_stratum_weight_multisets=True,
        fraction_edge_positions_changed=float(np.mean(values!=weights)) if len(a) else 0.,
        mechanism='Final calibrated weight placement within original-affinity strata; not a pure Q permutation')


def build_cache(source, destination, *, variant='context', parent=None):
    source,destination=Path(source),Path(destination)
    if variant not in ('context','weight_permutation'):raise ValueError('Unknown fixed graph variant')
    destination.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    try:
        g=sparse.load_npz(source/'graph.npz')
        if variant=='context':
            result,info=calibrate(g);parent_sha=None;permutation=None
        else:
            if parent is None:raise ValueError('Placebo requires the immutable calibrated parent')
            parent=Path(parent);metadata=json.loads((parent/'manifest.json').read_text())
            if metadata['builder_id']!=BUILDER_ID or metadata['variant']!='context':raise ValueError('Wrong parent model')
            if metadata['source_files']['graph.npz']!=sha(source/'graph.npz'):raise ValueError('Parent uses another original graph')
            parent_sha=sha(parent/'graph.npz')
            if parent_sha!=metadata['files']['graph.npz']:raise ValueError('Parent graph changed')
            base=sparse.load_npz(parent/'graph.npz');info=metadata['calibration']
            if info['degenerate']:
                result=canonical(base);permutation=dict(skipped='no informative context adjustment')
            else:result,permutation=permute_weights(g,base)
        for name in ('features.npy','affinity.npy'):
            try:os.link(source/name,destination/name)
            except OSError:shutil.copyfile(source/name,destination/name)
        sparse.save_npz(destination/'graph.npz',result)
        sources={name:sha(source/name) for name in ('features.npy','affinity.npy','graph.npz')}
        files={name:sha(destination/name) for name in ('features.npy','affinity.npy','graph.npz')}
        if any(files[k]!=sources[k] for k in ('features.npy','affinity.npy')):raise ValueError('Frozen Z or complete A changed')
        manifest=dict(status='complete',builder_id=BUILDER_ID,builder_sha256=sha(__file__),
            variant=variant,source_cache=str(source.resolve()),source_files=sources,files=files,
            parent_graph_sha256=parent_sha,calibration=info,permutation=permutation,
            full_affinity='unchanged original dense bridge kernel and row sums',
            parameters=dict(kd=20,ks=20,gamma=.4),seconds=time.monotonic()-started,numpy=np.__version__)
        dump(destination/'manifest.json',manifest)
        return manifest
    except BaseException:
        dump(destination/'FAILED.json',dict(error=traceback.format_exc(),variant=variant))
        raise
