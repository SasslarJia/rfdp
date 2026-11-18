"""Fixed KR, CA-Jaccard and ECN on full-width distance rows and sparse encodings."""
import numpy as np
from scipy import sparse


def unit_float32(features):
    result = np.empty_like(features)
    for lo in range(0, len(result), 512):
        value = features[lo:lo+512].astype(np.float64)
        result[lo:lo+512] = value / np.linalg.norm(value, axis=1, keepdims=True)
    return result


class DistanceRows:
    """Reproduce the query-query / query-gallery / gallery-gallery block convention."""
    def __init__(self, frozen, queries, method):
        self.qn, self.method = queries, method
        self.x = unit_float32(frozen) if method == "k-reciprocal" else frozen
        if method == "ecn-rankdist":
            # The archived runner passes F directly to sklearn cosine_similarity.
            self.x = np.array(frozen, dtype=np.float32, copy=True)
            norms = np.sqrt(np.einsum("ij,ij->i", self.x, self.x))
            self.x /= norms[:, None]
        if method == "caj":
            self.squares = np.empty(len(self.x), dtype=np.float32)
            for lo in range(0, len(self.x), 512):
                xx = self.x[lo:lo+512]
                self.squares[lo:lo+512] = np.sum(xx*xx, axis=1)

    def rows(self, lo, hi):
        ids = np.arange(lo, hi)
        if self.method == "ecn-rankdist":
            dist = 1. - self.x[lo:hi] @ self.x.T
            np.clip(dist, 0., 2., out=dist)
            dist[np.arange(hi-lo), ids] = 0.
            return dist
        # Both authors take the transpose after column maximum normalization.
        # Cross-split distances always originate in the query-to-gallery block.
        dots = np.ascontiguousarray((self.x @ self.x[lo:hi].T).T)
        nq = max(0, min(hi, self.qn)-lo)
        if nq:
            dots[:nq, self.qn:] = self.x[lo:lo+nq] @ self.x[self.qn:].T
        if self.method == "k-reciprocal":
            dist = np.maximum(2. - 2.*dots, 0.)
            np.sqrt(dist, out=dist)
            dist[np.arange(hi-lo), ids] = 0.
        else:
            dist = np.maximum(self.squares[lo:hi, None]+self.squares[None, :]-2.*dots, 0.)
        np.power(dist, 2, out=dist)
        largest = dist.max(axis=1)
        if np.any(largest == 0):
            raise ValueError("A column has zero maximum in the author distance normalization")
        dist /= largest[:, None]
        return dist

    def ecn_background(self, lo, hi):
        # The ECN author uses orig_dist[gallery, query] for zero rank similarity.
        dist = 1. - (self.x[self.qn:] @ self.x[lo:hi].T).T
        return np.clip(dist, 0., 2.)


def leading_neighbors(distance, n, method, cams, notify, width=64):
    if method == "caj":
        intra = np.empty((n, 6), dtype=np.int32)
        inter = np.empty((n, 21), dtype=np.int32)
    else:
        k = 25 if method == "ecn-rankdist" else 21
        nearest = np.empty((n, k), dtype=np.int32)
    for lo in range(0, n, width):
        hi = min(lo+width, n)
        rows = distance.rows(lo, hi)
        if not np.isfinite(rows).all():
            raise ValueError("Nonfinite neighbor distance")
        if method == "caj":
            same = cams[lo:hi, None] == cams[None, :]
            # Preserve the author's multi-kth argpartition (and its tie handling).
            intra[lo:hi] = np.argpartition(rows + 999.*(~same), range(7), axis=1)[:, :6]
            inter[lo:hi] = np.argpartition(rows + 999.*same, range(22), axis=1)[:, :21]
        else:
            nearest[lo:hi] = np.argsort(rows, axis=1)[:, :k]
        notify("neighbors", hi, n)
    return (intra, inter) if method == "caj" else nearest


def reciprocal(nearest, i, count):
    forward = nearest[i, :count]
    return forward[np.any(nearest[forward, :count] == i, axis=1)]


def csr_rows(indices, values, n):
    ptr = np.r_[0, np.cumsum([len(i) for i in indices], dtype=np.int64)]
    return sparse.csr_matrix((np.concatenate(values), np.concatenate(indices), ptr), shape=(n, n))


def jaccard_encoding(distance, neighbors, n, method, notify, width=64):
    indices, weights = [], []
    for lo in range(0, n, width):
        hi = min(n, lo+width)
        rows = distance.rows(lo, hi)
        for i in range(lo, hi):
            if method == "caj":
                intra, inter = neighbors
                chosen = np.union1d(reciprocal(intra, i, 6), reciprocal(inter, i, 21))
            else:
                base = reciprocal(neighbors, i, 21)
                selected = [base]
                for other in base:
                    half = reciprocal(neighbors, other, 11)
                    if len(np.intersect1d(half, base)) > (2./3)*len(half):
                        selected.append(half)
                chosen = np.unique(np.concatenate(selected))
            w = np.exp(-rows[i-lo, chosen])
            indices.append(chosen)
            weights.append(w/w.sum() if len(w) else w)
        notify("reciprocal_encoding", hi, n)
    if method == "caj":
        expansion = np.concatenate([neighbors[0][:, :3], neighbors[1][:, :3]], axis=1)
    else:
        expansion = neighbors[:, :6]
    qe_cols, qe_values = [], []
    for i, group in enumerate(expansion):
        cols = np.unique(np.concatenate([indices[j] for j in group]))
        total = np.zeros(len(cols), dtype=np.float32)
        for j in group:
            total[np.searchsorted(cols, indices[j])] += weights[j]
        qe_cols.append(cols)
        qe_values.append(total / len(group))
        if i%4096 == 0:
            notify("local_query_expansion", i, n)
    return csr_rows(qe_cols, qe_values, n)


def jaccard_rank_blocks(distance, encoding, qn, width, start=0):
    reverse = encoding.tocsc()
    reverse.sort_indices()
    n = encoding.shape[0]
    for lo in range(start, qn, width):
        hi = min(qn, lo+width)
        rows = distance.rows(lo, hi)
        output = np.empty((hi-lo, n-qn), dtype=np.int32)
        for i in range(lo, hi):
            overlap = np.zeros(n, dtype=np.float32)
            begin, stop = encoding.indptr[i:i+2]
            for j, w in zip(encoding.indices[begin:stop], encoding.data[begin:stop]):
                a, b = reverse.indptr[j:j+2]
                ids = reverse.indices[a:b]
                overlap[ids] += np.minimum(w, reverse.data[a:b])
            shared = overlap[qn:]
            final = (1.-shared/(2.-shared))*.7 + rows[i-lo, qn:]*.3
            output[i-lo] = np.argsort(final, kind="stable") + qn
        yield lo, output


def ecn_rank_blocks(distance, nearest, qn, width, start=0):
    n, k = nearest.shape
    cols = nearest.reshape(-1)
    w = sparse.csr_matrix((np.tile(np.arange(k, 0, -1, dtype=np.int32), n), cols,
                           np.arange(0, n*k+1, k, dtype=np.int64)), shape=(n, n))
    first = nearest[:, 1:4]
    neighbors = np.concatenate([first, nearest[first, 1:9].reshape(n, 24)], axis=1)
    expanded = sparse.csr_matrix((np.ones(neighbors.size, dtype=np.int32), neighbors.ravel(),
                                 np.arange(0, neighbors.size+1, 27, dtype=np.int64)), shape=(n, n))
    # Integer dot products exactly preserve both 27-term neighbor multisets.
    # Repeated neighbors have their original multiplicity.
    expanded = expanded @ w
    gallery_w, gallery_ew = w[qn:].T.tocsr(), expanded[qn:].T.tocsr()
    for lo in range(start, qn, width):
        hi = min(lo+width, qn)
        both = (expanded[lo:hi] @ gallery_w + w[lo:hi] @ gallery_ew).toarray()
        dist = np.where(both != 0, -both.astype(np.float64)/54., distance.ecn_background(lo, hi))
        yield lo, np.argsort(dist, axis=1, kind="stable").astype(np.int32)+qn
