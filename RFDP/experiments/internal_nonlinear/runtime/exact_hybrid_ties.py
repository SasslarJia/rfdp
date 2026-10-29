"""Certify equal HyRDP responses by exact transposition symmetry, not tolerance."""
import mpmath as mp
import numpy as np


def swap_invariant(w, e, i, j):
    if e[i] != e[j] or w[i, i] != w[j, j]:
        return False
    # Symmetry of W was checked upstream. All coordinates outside {i,j}
    # must match exactly. The shared ij edge is unchanged by a swap.
    ri = {int(k): v for k, v in zip(w[i].indices, w[i].data) if k not in (i, j)}
    rj = {int(k): v for k, v in zip(w[j].indices, w[j].data) if k not in (i, j)}
    return ri == rj


def certify_ties(w, e, alpha, beta, scores, intervals, labels, nc, dps, Block):
    """Return only if every ambiguous front-five comparison has an exact proof.

    PWP'=W and Pe=e imply PMP'=M and Pb=b. Unique SPD solution gives
    P psi=psi. Thus exchanged coordinates are exactly equal. Equal score
    representatives are rechecked against the original equation afterwards.
    """
    n = len(scores)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    order = sorted(range(n), key=lambda i: (scores[i], i))
    pairs = []
    for pos in range(min(5, n-1)):
        i = order[pos]
        for j in order[pos+1:]:
            if intervals[i][1] < intervals[j][0]:
                continue
            if not swap_invariant(w, e, i, j):
                return None
            parent[find(j)] = find(i)
            pairs.append((i, j))
    if not pairs:
        return None
    grouped = {}
    for i, j in pairs:
        for k in (i, j):
            grouped.setdefault(find(k), set()).add(k)
    groups = [sorted(v) for v in grouped.values()]
    values = list(scores)
    for group in groups:
        lo = max(intervals[i][0] for i in group)
        hi = min(intervals[i][1] for i in group)
        if lo > hi:
            return None
        shared = (mp.mpf(lo)+mp.mpf(hi))/2
        for i in group:
            values[i] = shared
    updated = list(intervals)
    max_residual = mp.mpf(0)
    # Recheck all blocks so the accepted residual belongs to returned values.
    for c in range(nc):
        ids = np.flatnonzero(labels == c)
        block = Block(w[ids][:, ids].tocsr(), e[ids], alpha, beta)
        cert = block.certify([values[i] for i in ids], dps)
        if not cert['accepted']:
            return None
        max_residual = max(max_residual, cert['relative'])
        for i, iv in zip(ids, cert['intervals']):
            updated[i] = iv
    order = sorted(range(n), key=lambda i: (values[i], i))
    for pos in range(min(5, n-1)):
        i = order[pos]
        for j in order[pos+1:]:
            if find(i) == find(j) and values[i] == values[j]:
                continue
            if not updated[i][1] < updated[j][0]:
                return None
    scores[:] = values
    intervals[:] = updated
    return dict(groups=groups, max_residual=max_residual,
                certificate='exact-W-e-transposition-symmetry-unique-SPD-response',
                ordering='equal-response-then-existing-local-node-index')
