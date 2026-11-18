"""Build the complete MSMT17 affinity on one disk-backed triangular array.

Distances are computed in both directions by full-width historical GEMM blocks.
The stored triangle is the FP32 average of both directed distances. Before
discarding the redundant affinity triangle, the builder checks the two halves
of every affinity tile byte for byte via SHA-256.
"""
import argparse
import hashlib
import shutil
import time
from pathlib import Path

import numpy as np

from experiments.reid_adaptation import graph as historical
from .extract import TOTAL, block_hash, read_json, save_json, writer
from .models import SPECS, digest
from .rankings import frozen_input


class PackedSymmetric:
    def __init__(self, values, n, sums=None):
        if values.shape != (n*(n+1)//2,) or values.dtype != np.float32:
            raise ValueError("Invalid packed FP32 triangle")
        self.values = values
        self.shape = (n, n)
        self.ids = np.arange(n, dtype=np.int64)
        self.base = self.ids * (self.ids+1) // 2
        self.row_sums = sums

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, coordinates):
        rows, cols = coordinates
        if isinstance(rows, slice) and rows == slice(None) and isinstance(cols, slice):
            start, end, stride = cols.indices(len(self))
            if stride == 1:
                return self.symmetric_rows(start, end).T
        r, c = np.broadcast_arrays(rows, cols)
        return self.values[self.base[np.maximum(r, c)] + np.minimum(r, c)]

    def symmetric_rows(self, start, end):
        rows = self.ids[start:end]
        result = np.empty((len(rows), len(self)), dtype=np.float32)
        for i, row in enumerate(rows):
            base = self.base[row]
            result[i, :start] = self.values[base:base+start]
        result[:, start:end] = self[rows[:, None], rows]
        for lo in range(end, len(self), 1024):
            hi = min(lo+1024, len(self))
            result[:, lo:hi] = self.values[self.base[lo:hi, None] + rows].T
        return result

    def sum(self, axis, dtype):
        if axis != 1 or np.dtype(dtype) != np.float64 or self.row_sums is None:
            raise ValueError("Only complete FP64 row sums are defined here")
        return self.row_sums


def write_distances(z, triangle, block_rows, notify):
    """Use the same dense row product and keep both operands of (D+D.T)*.5."""
    data, base = triangle.values, triangle.base
    for lo, hi in historical._blocks(len(z), block_rows):
        dots = z[lo:hi].dot(z.T)
        distances = np.maximum(2.0 - 2.0*dots, 0.0)
        del dots
        np.sqrt(distances, out=distances)
        rows = triangle.ids[lo:hi]
        distances[np.arange(hi-lo), rows] = 0.
        for i, row in enumerate(rows):
            dest = data[base[row]:base[row]+row+1]
            dest[:lo] = (dest[:lo] + distances[i, :lo]) * .5
            dest[lo:] = (distances[i, lo:row+1] + distances[:i+1, row]) * .5
        # A future source row still needs this direction's rounded distance.
        for start in range(hi, len(z), 1024):
            end = min(start+1024, len(z))
            data[base[start:end, None] + rows] = distances[:, start:end].T
        notify("distance", hi)
    data.flush()


def transform(triangle, scales, kd, gamma, block_rows, notify):
    n = len(triangle)
    sums = np.empty(n, dtype=np.float64)
    dense_body_hash = hashlib.sha256()
    upper_hash = {}
    for lo, hi in historical._blocks(n, block_rows):
        # At this point every entry touching a not-yet-processed row has its
        # untouched distance in the lower triangle of that or a later row.
        distances = triangle.symmetric_rows(lo, hi)
        values = historical.affinity_block(distances, scales, lo, hi, gamma/(2*kd))
        rows = triangle.ids[lo:hi]
        if not np.all(values[np.arange(hi-lo), rows] == 1):
            raise ValueError("Affinity lost a unit self-loop")
        for start, end in historical._blocks(n, block_rows):
            tile = values[:, start:end]
            if start < lo:
                if block_hash(tile) != upper_hash.pop((start, lo)):
                    raise ValueError(f"Directed affinities differ for blocks {start}, {lo}")
            elif start == lo:
                if not np.array_equal(tile, tile.T):
                    raise ValueError(f"Asymmetric diagonal affinity block at {lo}")
            else:
                upper_hash[(lo, start)] = block_hash(tile.T)
        sums[lo:hi] = values.sum(axis=1, dtype=np.float64)
        dense_body_hash.update(memoryview(values).cast("B"))
        for i, row in enumerate(rows):
            base = triangle.base[row]
            triangle.values[base:base+row+1] = values[i, :row+1]
        notify("affinity", hi)
    if upper_hash or not np.all(np.isfinite(sums) & (sums > 0)):
        raise ValueError("Incomplete affinity symmetry or degree verification")
    triangle.values.flush()
    return sums, dense_body_hash.hexdigest()


def build(frozen, backbone, output, packed_path, block_rows=128):
    with writer(output):
        feature_path, _, binding = frozen_input(frozen, backbone)
        manifest = output / "manifest.json"
        if manifest.is_file():
            done = read_json(manifest)
            if done["input"] != binding or digest(packed_path) != done["packed_sha256"]:
                raise ValueError("Existing graph belongs to different bytes")
            for name, expected in done["files"].items():
                if digest(output/name) != expected:
                    raise ValueError(f"Graph file differs: {name}")
            return
        if packed_path.exists():
            raise FileExistsError(f"Existing unfinished packed graph: {packed_path}")
        started = time.monotonic()
        n = TOTAL
        packed_path.parent.mkdir(parents=True, exist_ok=True)
        footprint = n*(n+1)//2*4 + 4096
        disk = shutil.disk_usage(packed_path.parent)
        if disk.free-footprint < max(10*2**30, disk.total//4):
            raise OSError("Full symmetric graph would cross the storage reserve")
        f = np.load(feature_path, mmap_mode="r")
        z = np.lib.format.open_memmap(output/"features.npy", mode="w+", dtype=np.float32, shape=f.shape)
        for lo, hi in historical._blocks(n, 512):
            part = f[lo:hi].astype(np.float64)
            z[lo:hi] = part / np.linalg.norm(part, axis=1, keepdims=True)
        z.flush()
        data = np.lib.format.open_memmap(packed_path, mode="w+", dtype=np.float32, shape=(n*(n+1)//2,))
        triangle = PackedSymmetric(data, n)

        def notify(phase, done):
            if done == n or (done//block_rows) % 16 == 0:
                save_json(output/"progress.json", dict(phase=phase, completed=done, total=n,
                                                       wall_s=time.monotonic()-started))

        write_distances(z, triangle, block_rows, notify)
        scales = historical._distance_sums(triangle, 20, block_rows,
                                           lambda e: notify(e["phase"], e["completed"]))
        sums, dense_sha = transform(triangle, scales, 20, .4, block_rows, notify)
        np.save(output/"row-sums.npy", sums, allow_pickle=False)
        info = historical._write_mutual_graph(triangle, output/"graph.npz", 20, block_rows,
                                               lambda e: notify(e["phase"], e["completed"]))
        # Independently expand the stored triangle in historical full row order.
        expanded = hashlib.sha256()
        for lo, hi in historical._blocks(n, block_rows):
            rows = triangle.symmetric_rows(lo, hi)
            if not np.array_equal(rows.sum(axis=1, dtype=np.float64), sums[lo:hi]):
                raise ValueError("Stored affinity row degrees changed")
            expanded.update(memoryview(rows).cast("B"))
        if expanded.hexdigest() != dense_sha:
            raise ValueError("Full stored affinity differs from historical row generation")
        files = {name: digest(output/name) for name in ("features.npy", "row-sums.npy", "graph.npz")}
        save_json(manifest, dict(input=binding, n=n, dtype="float32", kd=20, ks=20, gamma=.4,
                                 full_matrix_shape=[n, n], dense_body_sha256=dense_sha,
                                 packed_path=str(packed_path.resolve()), packed_sha256=digest(packed_path),
                                 packed_bytes=packed_path.stat().st_size, files=files, graph=info,
                                 seconds=time.monotonic()-started, block_rows=block_rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--backbone", choices=SPECS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packed", type=Path, required=True)
    args = parser.parse_args()
    build(args.frozen, args.backbone, args.output, args.packed)
