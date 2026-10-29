"""Assemble a quadratic continuous-fitting hypothesis, not a production ranker.

The stage matrix is the original HyRDP matrix. Only the source changes.
No ranking certificate, deployment, dataset loading or performance claim is made
by this small mathematical implementation. See README.md for the energy.
"""
from typing import NamedTuple

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


class StageSystem(NamedTuple):
    matrix: sparse.csr_matrix
    rhs: np.ndarray
    boundary_probability: np.ndarray


def assemble_system(w_uu, e_u, alpha=5., beta=25., source='continuous'):
    """Return M, b and p with original or continuous fitting on a fixed stage.

    Every connected component must have a positive boundary anchor. Invalid
    inputs are rejected, not regularized. Self loops follow the original degree
    convention. p is a graph transition probability, not identity confidence.
    """
    w = sparse.csr_matrix(w_uu, dtype=np.float64, copy=True)
    w.sum_duplicates()
    w.eliminate_zeros()
    e = np.asarray(e_u, dtype=np.float64)
    if w.shape[0] == 0 or w.shape[0] != w.shape[1] or e.shape != (w.shape[0],):
        raise ValueError('Expected a nonempty square W and one boundary weight per node')
    if (not np.isfinite(w.data).all() or np.any(w.data < 0)
            or not np.isfinite(e).all() or np.any(e < 0)):
        raise ValueError('Weights must be finite and nonnegative')
    if (w - w.T).nnz:
        raise ValueError('W must be symmetric')
    if not np.isfinite([alpha, beta]).all() or min(alpha, beta) <= 0:
        raise ValueError('alpha and beta must be finite and positive')
    h = np.asarray(w.sum(axis=1)).ravel()
    if np.any(h <= 0) or not np.isfinite(h + e).all():
        raise ValueError('Invalid local degrees')
    count, labels = connected_components(w, directed=False)
    if len(np.unique(labels[e > 0])) != count:
        raise ValueError('Every component requires a boundary anchor')
    inv_h = 1. / np.sqrt(h)
    normalized = sparse.diags(inv_h) @ w @ sparse.diags(inv_h)
    # Match the original implementation's arithmetic order as well as its formula.
    matrix = (sparse.diags(1. + alpha * e / (h + e)) - normalized).tocsr()
    p = e / (h + e)
    if source == 'original':
        rhs = np.where(e > 0, alpha, beta)
    elif source == 'continuous':
        rhs = alpha * p + beta * (1. - p)
    else:
        raise ValueError('source must be original or continuous')
    return StageSystem(matrix, rhs, p)
