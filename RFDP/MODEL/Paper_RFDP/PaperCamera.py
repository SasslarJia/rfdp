"""Task-specific camera edge prior; original HyRDP objective on the modified graph."""
import numpy as np
from scipy import sparse
from .PaperFullIterative import FullIHyRDP

def camera_graph(graph, cameras, eta=0.5):
    if not 0 < eta <= 1:
        raise ValueError('eta must be in (0,1]')
    w = sparse.csr_matrix(graph, dtype=np.float64).copy()
    cameras = np.asarray(cameras)
    if cameras.ndim != 1 or len(cameras) != w.shape[0]:
        raise ValueError('camera count mismatch')
    rows = np.repeat(np.arange(w.shape[0]), np.diff(w.indptr))
    same = cameras[rows] == cameras[w.indices]
    w.data[same] *= eta
    return w

class CameraIHyRDP(FullIHyRDP):

    def __init__(self, graph, full, cameras, eta=0.5):
        self.cameras = np.asarray(cameras)
        self.eta = eta
        super().__init__(camera_graph(graph, cameras, eta), full)
        if eta != 1:
            for cam in np.unique(self.cameras):
                ids = np.flatnonzero(self.cameras == cam)
                for s in range(0, len(ids), 128):
                    ii = ids[s:s + 128]
                    self.full_sums[ii] -= (1 - eta) * np.asarray(full[np.ix_(ii, ids)], dtype=np.float64).sum(1)
        if not np.isfinite(self.full_sums).all() or np.any(self.full_sums <= 0):
            raise ValueError('Invalid camera-weighted similarity degrees')

    def bridges(self, active_components):
        key = tuple(sorted(active_components))
        if key in self.bridge_cache:
            return self.bridge_cache[key]
        g = np.flatnonzero(np.isin(self.components, key))
        choices = []
        for comp in range(self.nc):
            if comp in active_components:
                continue
            outside = np.flatnonzero(self.components == comp)
            best = None
            for start in range(0, len(outside), 128):
                ids = outside[start:start + 128]
                block = np.asarray(self.full[np.ix_(ids, g)], dtype=np.float64)
                block *= np.where(self.cameras[ids, None] == self.cameras[g][None, :], self.eta, 1.0)
                block /= self.full_sums[ids, None]
                (i, j) = np.unravel_index(np.argmax(block), block.shape)
                value = float(block[i, j])
                candidate = (value, int(ids[i]), int(g[j]), comp)
                if best is None or value > best[0] or (value == best[0] and candidate[1:3] < best[1:3]):
                    best = candidate
            if best is not None and best[0] > 0:
                choices.append(best)
        choices.sort(key=lambda item: (-item[0], item[1], item[2]))
        result = [(i, j, float(self.full[i, j]) * (self.eta if self.cameras[i] == self.cameras[j] else 1.0)) for (_, i, j, _) in choices[:self.step]]
        self.bridge_cache[key] = result
        return result
