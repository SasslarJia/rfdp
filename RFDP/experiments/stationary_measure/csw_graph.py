"""Frozen CSW inputs and an independent weighted-kernel read-back view.

The producer remains ReliabilityIHyRDP. This view reconstructs its graph and
bridge kernel for saved-score verification; identity labels are never loaded.
"""
import numpy as np
from scipy import sparse
from scipy.io import loadmat


def load_inputs(cfg, full):
    from experiments.reliability.reliability_model import load_support, array_sha256
    data = loadmat(cfg['mat'], variable_names=['query_cam', 'gallery_cam'])
    cameras = np.r_[data['query_cam'].ravel(), data['gallery_cam'].ravel()]
    support, metadata = load_support(cfg['support_file'], expected={
        'mode': cfg['support_mode'], 'k': cfg['k'],
        'cameras_sha256': array_sha256(cameras), 'full_sha256': array_sha256(full)})
    return cameras, support, metadata


class ReadbackKernel:
    def __init__(self, graph, full, cameras, support, rho=.5, lam=.25):
        if not 0 < rho <= 1 or not 0 <= lam < 1:
            raise ValueError('Invalid CSW camera/support weights')
        self.full = full
        self.cameras = np.asarray(cameras)
        cameras = self.cameras
        self.support = support
        if support.nnz == len(cameras)**2 and np.all(support.data == 1):
            lam = 0.
        self.rho, self.lam = rho, lam
        n = len(cameras)
        if graph.shape != (n, n) or full.shape != (n, n) or support.shape != (n, n):
            raise ValueError('CSW input dimensions disagree')
        self.graph = sparse.csr_matrix(graph, dtype=np.float64, copy=True)
        self.graph.sum_duplicates(); self.graph.eliminate_zeros(); self.graph.sort_indices()
        rows = np.repeat(np.arange(n), np.diff(self.graph.indptr))
        columns = self.graph.indices
        self.graph.data[self.cameras[rows] == self.cameras[columns]] *= rho
        if lam:
            q = np.asarray(support[rows, columns]).ravel()
            self.graph.data *= 1-lam+lam*q
        if np.any(self.graph.data <= 0):
            raise ValueError('CSW lost positive graph support')
        # Preserve the historical arithmetic order of fixed full-kernel sums.
        base = np.asarray(full.sum(axis=1, dtype=np.float64))
        if rho != 1:
            for camera in np.unique(cameras):
                ids = np.flatnonzero(cameras == camera)
                for start in range(0, len(ids), 128):
                    selected = ids[start:start+128]
                    base[selected] -= (1-rho)*np.asarray(full[np.ix_(selected, ids)], dtype=np.float64).sum(1)
        if not lam:
            self.sums = base
            return
        correction = np.zeros(n)
        for start in range(0, n, 128):
            stop = min(n, start+128)
            begin, end = support.indptr[start], support.indptr[stop]
            rr = np.repeat(np.arange(start, stop), np.diff(support.indptr[start:stop+1]))
            cc = support.indices[begin:end]
            values = np.asarray(full[rr, cc], dtype=np.float64)
            values *= np.where(cameras[rr] == cameras[cc], rho, 1.)
            values *= support.data[begin:end]
            correction[start:stop] = np.bincount(rr-start, weights=values, minlength=stop-start)
        self.sums = (1-lam)*base+lam*correction
        if not np.isfinite(self.sums).all() or np.any(self.sums <= 0):
            raise ValueError('Invalid CSW full-kernel row sums')

    def block(self, rows, columns):
        rows, columns = np.asarray(rows, dtype=int), np.asarray(columns, dtype=int)
        values = np.asarray(self.full[np.ix_(rows, columns)], dtype=np.float64)
        values *= np.where(self.cameras[rows, None] == self.cameras[columns][None, :], self.rho, 1.)
        if self.lam:
            values *= 1-self.lam+self.lam*self.support[rows][:, columns].toarray()
        return values
