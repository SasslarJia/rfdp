"""Original equations with explicit graph, admission, or fitting adaptations.

The P3 producer accumulates target rows once. Its reader independently slices
G[U,L] and accumulates columns in admission order. Identical multiplication and
addition order identifies the actual binary boundary used by the certificate;
neither a producer boundary vector nor a tolerance-expanded equation is trusted.
"""
import hashlib

import numpy as np
from scipy import sparse

from experiments.continuous_fitting import certified
from experiments.continuous_fitting.original_solver import WarmOriginalSolver, SOLVER_ID as WARM_SOLVER_ID
from .schedule import ScheduledIHyRDP, promotion_step

ROUTES = ('p1_graph', 'early5', 'camera_boundary', 'p1_early5', 'late5')
SOLVER_ID = 'reid-adaptation-warm-certified-v1'


def boundary_sha(values):
    return hashlib.sha256(np.asarray(values, dtype='<f8').tobytes()).hexdigest()


def boundary_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(count=len(values), positive_count=int(np.count_nonzero(values)),
                sum=float(values.sum(dtype=np.float64)),
                minimum=float(values.min()), maximum=float(values.max()))


class CameraBoundaryAccumulator:
    """Query-local persistent contacts; current bridges are never persisted."""
    def __init__(self, graph, cameras, rho):
        self.graph, self.cameras, self.rho = graph, cameras, rho
        self.reset()

    def reset(self):
        self.targets = ()
        self.boundary = np.zeros(self.graph.shape[0], dtype=np.float64)

    def stage(self, indices, targets, bridges):
        targets = tuple(map(int, targets))
        if targets[:len(self.targets)] != self.targets or len(set(targets)) != len(targets):
            raise ValueError('Target history must extend the current query')
        for target in targets[len(self.targets):]:
            begin, end = self.graph.indptr[target:target + 2]
            neighbors = self.graph.indices[begin:end]
            factors = np.where(self.cameras[neighbors] == self.cameras[target], self.rho, 1.)
            self.boundary[neighbors] += self.graph.data[begin:end] * factors
        self.targets = targets
        ids = np.asarray(indices, dtype=np.int64)
        result = self.boundary[ids].copy()
        positions = {int(node): i for i, node in enumerate(ids)}
        for outside, inside, weight in bridges:
            if outside in positions:
                result[positions[outside]] += float(weight) * (
                    self.rho if self.cameras[outside] == self.cameras[inside] else 1.)
        return result


def rebuild_camera_boundary(graph, cameras, rho, indices, targets, bridges):
    """Stateless read-back: columns of G[U,L], in original target order."""
    ids = np.asarray(indices, dtype=np.int64)
    targets = tuple(map(int, targets))
    contacts = graph[ids][:, list(targets)].tocsc()
    result = np.zeros(len(ids), dtype=np.float64)
    for column, target in enumerate(targets):
        begin, end = contacts.indptr[column:column + 2]
        rows = contacts.indices[begin:end]
        factors = np.where(cameras[ids[rows]] == cameras[target], rho, 1.)
        result[rows] += contacts.data[begin:end] * factors
    positions = {int(node): i for i, node in enumerate(ids)}
    for outside, inside, weight in bridges:
        if outside in positions:
            result[positions[outside]] += float(weight) * (
                rho if cameras[outside] == cameras[inside] else 1.)
    return result


class AdaptationSolver:
    def __init__(self, route, schedule, graph, cameras=None, rho=.5):
        self.route, self.schedule, self.rho = route, schedule, rho
        self.warm = WarmOriginalSolver()
        self.contacts = (CameraBoundaryAccumulator(graph, cameras, rho)
                         if route == 'camera_boundary' else None)

    def reset(self):
        self.warm.reset()
        if self.contacts is not None:
            self.contacts.reset()

    def __call__(self, w, e, alpha, beta, *, diagnostics, indices, targets, bridges):
        fitting = (e if self.contacts is None else
                   self.contacts.stage(indices, targets, bridges))
        if not np.array_equal(np.asarray(e) > 0, np.asarray(fitting) > 0):
            raise ValueError('Positive camera weights must preserve boundary support')
        values, residual = self.warm(w, fitting, alpha, beta, diagnostics=diagnostics,
                                    indices=indices, targets=targets, bridges=bridges)
        event = certified.EVENTS[-1]
        event.update(solver_id=SOLVER_ID, linear_solver_id=WARM_SOLVER_ID,
                     adaptation_route=self.route, promotion_schedule=self.schedule)
        if self.contacts is not None:
            event.update(fitting_rho=self.rho, fitting_boundary_sha256=boundary_sha(fitting),
                         fitting_boundary_summary=boundary_summary(fitting))
        return values, residual


def initialize(cfg, graph, affinity, compute=True):
    """Protocol validation belongs to the caller; bind only this runtime."""
    route = cfg['adaptation_route']
    if route not in ROUTES or cfg['source'] != 'original':
        raise ValueError('Unsupported original-equation adaptation')
    schedule = cfg['promotion_schedule']
    promotion_step(schedule, 0, cfg['step'])
    graph = sparse.csr_matrix(graph, dtype=np.float64, copy=True)
    graph.sort_indices()
    if not graph.has_canonical_format or graph.shape != affinity.shape:
        raise ValueError('Graph coordinates or affinity dimensions disagree')
    if (graph - graph.T).nnz:
        raise ValueError('Adaptation requires the unchanged symmetric graph')
    cameras = None
    rho = float(cfg['boundary_rho'])
    if route == 'camera_boundary':
        cameras = np.load(cfg['camera_ids_file'], allow_pickle=False)
        if cameras.shape != (graph.shape[0],) or cameras.dtype.kind not in 'iu':
            raise ValueError('Camera order/shape differs from the graph')
        if not np.isfinite(rho) or not 0 < rho <= 1:
            raise ValueError('Camera fitting rho must be positive and at most one')
    sums = np.asarray(affinity.sum(axis=1, dtype=np.float64))

    def certifier(w, e, values, event, *, alpha, beta, source, indices, targets, bridges):
        assert source == 'original' and event['adaptation_route'] == route
        assert event['solver_id'] == SOLVER_ID and event['linear_solver_id'] == WARM_SOLVER_ID
        assert event['promotion_schedule'] == schedule
        fitting = e
        if route == 'camera_boundary':
            assert event['fitting_rho'] == rho
            fitting = rebuild_camera_boundary(graph, cameras, rho, indices, targets, bridges)
            assert np.array_equal(np.asarray(e) > 0, fitting > 0)
            assert event['fitting_boundary_sha256'] == boundary_sha(fitting)
            assert event['fitting_boundary_summary'] == boundary_summary(fitting)
        return certified.certify_saved(w, fitting, values, alpha=alpha, beta=beta, source='original')

    engine = None
    if compute:
        solver = AdaptationSolver(route, schedule, graph, cameras, rho)
        engine = ScheduledIHyRDP(graph, affinity, alpha=cfg['alpha'], beta=cfg['beta'],
                                step=cfg['step'], stage_solver=solver,
                                promotion_schedule=schedule)
    return engine, graph, None, sums, certifier
