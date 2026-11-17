"""The support is fixed; only the scoring graph changes after five new nodes.

Candidate discovery and the persistent boundary use the original graph from
the outset. Equal positive supports make their frontier identical to the
context graph's for any fixed targets. Early solves use context W and freshly
accumulated context e. At the switch the maintained original e is already the
exact ordered sum for the actual targets, never a scaled context boundary.
"""
from pathlib import Path

import numpy as np
from scipy import sparse

from experiments.continuous_fitting import certified
from experiments.continuous_fitting.original_solver import SOLVER_ID as WARM_ID
from experiments.continuous_fitting.stable_original_solver import SOLVER_ID as STABLE_ID
from experiments.reid_adaptation.runtime import AdaptationSolver, SOLVER_ID, boundary_sha
from experiments.reid_adaptation.schedule import ScheduledIHyRDP
from .protocol import ROUTE, WINDOW


def phase(targets):
    return 'context' if len(targets)-1 < WINDOW else 'original'


def producer_boundary(graph, indices, targets, bridges):
    boundary = np.zeros(graph.shape[0], dtype=np.float64)
    for node in targets:
        start, stop = graph.indptr[node:node+2]
        boundary[graph.indices[start:stop]] += graph.data[start:stop]
    result = boundary[np.asarray(indices, dtype=np.int64)]
    positions = {int(v): i for i, v in enumerate(indices)}
    for outside, inside, weight in bridges:
        if outside in positions: result[positions[outside]] += float(weight)
    return result


def reader_boundary(graph, indices, targets, bridges):
    # Independent column-wise reconstruction; no producer accumulator reused.
    contacts = graph[np.asarray(indices)][:, list(targets)].tocsc()
    result = np.zeros(len(indices), dtype=np.float64)
    for column in range(len(targets)):
        start, stop = contacts.indptr[column:column+2]
        result[contacts.indices[start:stop]] += contacts.data[start:stop]
    for outside, inside, weight in bridges:
        result[np.asarray(indices) == outside] += float(weight)
    return result


class SwitchSolver:
    def __init__(self, original, context):
        self.context = context
        self.solvers = {
            'context': AdaptationSolver(ROUTE, 'early5', context, numerical_solver='stable_lu_certified'),
            'original': AdaptationSolver(ROUTE, 'early5', original, numerical_solver='warm_certified'),
        }

    def reset(self):
        for solver in self.solvers.values(): solver.reset()

    def __call__(self, w, e, alpha, beta, *, diagnostics, indices, targets, bridges):
        current = phase(targets)
        if current == 'context':
            w = self.context[indices][:, indices]
            e = producer_boundary(self.context, indices, targets, bridges)
        values, residual = self.solvers[current](w, e, alpha, beta,
                         diagnostics=diagnostics, indices=indices, targets=targets, bridges=bridges)
        certified.EVENTS[-1].update(scoring_graph=current, context_window_new_nodes=WINDOW,
                                   stage_boundary_sha256=boundary_sha(e))
        return values, residual


def initialize(cfg, context, affinity, compute=True):
    if (cfg['adaptation_route'] != ROUTE or cfg['source'] != 'original' or
        cfg['promotion_schedule'] != 'early5' or cfg['context_window_new_nodes'] != WINDOW or
        cfg['numerical_solver'] != 'phase_certified'):
        raise ValueError('Invalid fixed graph-switch runtime')
    original = sparse.load_npz(Path(cfg['original_cache'])/'graph.npz').astype(np.float64).tocsr()
    context = sparse.csr_matrix(context, dtype=np.float64, copy=True)
    for graph in (original, context):
        graph.eliminate_zeros(); graph.sort_indices()
        if (not graph.has_canonical_format or graph.shape != affinity.shape or
            not np.isfinite(graph.data).all() or np.any(graph.data <= 0) or
            (graph-graph.T).nnz):
            raise ValueError('Invalid symmetric positive graph')
    if (not np.array_equal(original.indptr, context.indptr) or
        not np.array_equal(original.indices, context.indices) or
        not np.array_equal(original.diagonal(), context.diagonal())):
        raise ValueError('Stage switch requires identical support and self-loops')
    sums = np.asarray(affinity.sum(axis=1, dtype=np.float64))

    def certifier(w, e, values, event, *, alpha, beta, source, indices, targets, bridges):
        current = phase(targets)
        assert source == 'original' and event['adaptation_route'] == ROUTE
        assert event['solver_id'] == SOLVER_ID and event['promotion_schedule'] == 'early5'
        assert event['scoring_graph'] == current and event['context_window_new_nodes'] == WINDOW
        assert event['linear_solver_id'] == (STABLE_ID if current == 'context' else WARM_ID)
        if current == 'context':
            w = context[indices][:, indices]
            e = reader_boundary(context, indices, targets, bridges)
        assert event['stage_boundary_sha256'] == boundary_sha(e)
        return certified.certify_saved(w, e, values, alpha=alpha, beta=beta, source='original')

    engine = None
    if compute:
        engine = ScheduledIHyRDP(original, affinity, alpha=cfg['alpha'], beta=cfg['beta'],
                                step=5, promotion_schedule='early5',
                                stage_solver=SwitchSolver(original, context))
    return engine, original, None, sums, certifier
