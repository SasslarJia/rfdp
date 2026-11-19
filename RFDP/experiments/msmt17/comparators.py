"""One registered configuration per MSMT17 comparator, with complete gallery readback."""
import argparse
import os
import time
import traceback
from pathlib import Path

import numpy as np
from scipy import sparse

from experiments.reid_adaptation.resources import cpu_allocation
from .extract import COUNTS, block_hash, read_json, save_json, writer
from .models import SPECS, digest
from .neighbors import DistanceRows, ecn_rank_blocks, jaccard_encoding, jaccard_rank_blocks, leading_neighbors
from .propagation import cheb_propagate, mr_rank_blocks
from .rankings import accept_ranking, frozen_input
from .resources import GIB, cloud_state, require_disk_space

REFERENCE = {
    "k-reciprocal": "04bf1b5886ef06275d88be4d68d8f8cfd3ebc892e026f41e848ce294ab192b8e",
    "caj": "75ea25170204baf67f61d48d5f43e6282bed548f4e0076f14d769dfa07f15d36",
    "mr-fixed-kernel": "94e0180ccc72b6491ca2c5b52f704669f309b72223432787435cd8b9091fea4d",
    "ecn-rankdist": "9103d9467eb843f0860af89b79a02f46072ceab46259f366bad1c01b9c03a4aa",
    "cheb-gr": "b927d20c894f06e580f8c99eadc399b24534eaecadf5496b46486d0bd175f29f",
}
PARAMETERS = {
    "k-reciprocal": dict(k1=20, k2=6, lambda_value=.3, input="unsquared Euclidean"),
    "caj": dict(k1=30, k2=6, k1_intra=5, k1_inter=20, k2_intra=3, k2_inter=3,
                lambda_value=.3, ckrnns=True, clqe=True, input="squared Euclidean"),
    "mr-fixed-kernel": dict(alpha=.99, mutual_k=20, gamma=.4, zero_diagonal=True,
                            rtol=1e-11, residual_limit=1e-9, maxiter=5000),
    "ecn-rankdist": dict(k=25, t=3, q=8, method="rankdist", zero_merge="author cosine"),
    "cheb-gr": dict(beta=.2, lambda_threshold=6., rounds=3, mode="nonsym",
                    internal_distance="squared Euclidean", output_distance="1 - normalized dot"),
}


def cheb_rank_blocks(z, qn, width, start):
    for lo in range(start, qn, width):
        hi = min(qn, lo+width)
        distances = 1.-z[lo:hi] @ z[qn:].T
        if not np.isfinite(distances).all():
            raise ArithmeticError("Invalid Cheb-GR distance")
        yield lo, np.argsort(distances, axis=1, kind="stable").astype(np.int32)+qn


def run(frozen, backbone, cache, method, output):
    with writer(output):
        started = time.monotonic()
        feature, rows, input_binding = frozen_input(frozen, backbone)
        nq, ng = COUNTS["query"], COUNTS["gallery"]
        x = np.load(feature, mmap_mode="r")
        workers = cpu_allocation()["allocated"]
        sources = {f: digest(Path(__file__).with_name(f)) for f in
                   ("comparators.py", "rankings.py", "neighbors.py", "propagation.py")}
        binding = dict(input=input_binding, method=method, parameters=PARAMETERS[method],
                       reference_sha256=REFERENCE[method], execution_sources=sources,
                       device="cpu", allocated_cpus=workers)
        if method == "mr-fixed-kernel":
            meta = read_json(cache/"manifest.json")
            if (meta["input"] != input_binding or (meta["kd"], meta["ks"], meta["gamma"]) != (20, 20, .4)
                    or digest(cache/"graph.npz") != meta["files"]["graph.npz"]):
                raise ValueError("MR requires the registered graph of this backbone")
            binding["graph_sha256"] = meta["files"]["graph.npz"]
        path = output/"rankings.npy"
        progress_file = output/"progress.json"
        if (output/"FULL_ACCEPTANCE.json").is_file():
            done = read_json(output/"FULL_ACCEPTANCE.json")
            if (done["binding"] != binding or digest(path) != done["ranking_sha256"]
                    or digest(output/"metrics.json") != done["metrics_sha256"]
                    or digest(output/"per-query.json") != done["per_query_sha256"]):
                raise ValueError("Completed comparator differs")
            return
        state = cloud_state([output])
        if progress_file.exists():
            progress = read_json(progress_file)
            if progress["binding"] != binding:
                raise ValueError("Comparator checkpoint signature differs")
            table = np.load(path, mmap_mode="r+")
        else:
            if path.exists():
                raise FileExistsError(path)
            require_disk_space(output, nq*ng*4 + 4096 + 64*2**20)
            table = np.lib.format.open_memmap(path, mode="w+", dtype=np.int32, shape=(nq, ng))
            progress = dict(binding=binding, blocks=[], compute_s=0., save_s=0.)
            save_json(progress_file, progress)
        if table.shape != (nq, ng) or table.dtype != np.int32:
            raise ValueError("Wrong comparator ranking shape")
        start = 0
        for hi, expected in progress["blocks"]:
            if not start < hi <= nq or block_hash(table[start:hi]) != expected:
                raise ValueError("Comparator checkpoint row hash differs")
            start = hi
        monitor = 0.

        def notify(phase, count, total):
            nonlocal monitor
            if count == total or time.monotonic() >= monitor:
                mem = cloud_state([output])
                save_json(output/"phase.json", dict(phase=phase, completed=count, total=total,
                                                    resources=mem, wall_s=time.monotonic()-started))
                monitor = time.monotonic()+30.

        if start < nq:
            tick = time.monotonic()
            width = 64
            if method == "mr-fixed-kernel":
                graph = sparse.load_npz(cache/"graph.npz")
                stats_path = output/"mr-solve-statistics.json"
                stats = read_json(stats_path)[:start] if stats_path.exists() else []
                if len(stats) != start:
                    raise ValueError("Incomplete accepted MR solve-statistics checkpoint")
                blocks = mr_rank_blocks(graph, nq, width, workers, stats, start=start)
            elif method == "cheb-gr":
                # One full FP32 buffer, up to three live feature matrices, a
                # row mask/outer-product workspace and the same 2 GiB margin.
                required = len(x)**2*4 + 3*x.nbytes + 128*len(x)*5 + 2*GIB
                if state["memory_current"]+required > .7*state["memory_limit"]:
                    raise MemoryError(f"Complete CPU Cheb-GR requires {required} additional bytes")
                import torch
                torch.set_num_threads(workers)
                updated = cheb_propagate(x, "cpu", notify)
                blocks = cheb_rank_blocks(updated, nq, width, start)
            else:
                distance = DistanceRows(x, nq, method)
                cameras = np.array([row["camid"] for row in rows]) if method == "caj" else None
                neighbors = leading_neighbors(distance, len(x), method, cameras, notify)
                if method == "ecn-rankdist":
                    blocks = ecn_rank_blocks(distance, neighbors, nq, width, start=start)
                else:
                    encoding = jaccard_encoding(distance, neighbors, len(x), method, notify)
                    blocks = jaccard_rank_blocks(distance, encoding, nq, width, start=start)
            for lo, ordered in blocks:
                now = time.monotonic()
                progress["compute_s"] += now-tick
                end = lo+len(ordered)
                if lo != start or end > nq or ordered.shape != (end-lo, ng) or ordered.dtype != np.int32:
                    raise ValueError("Comparator omitted a query or produced a partial gallery")
                table[lo:end] = ordered
                table.flush()
                if method == "mr-fixed-kernel":
                    save_json(stats_path, stats)
                progress["blocks"].append([end, block_hash(ordered)])
                progress["save_s"] += time.monotonic()-now
                save_json(progress_file, progress)
                start = end
                notify("complete_gallery_sort", start, nq)
                tick = time.monotonic()
        del table
        if start != nq:
            raise RuntimeError("Comparator stopped before the full population")
        tick = time.monotonic()
        receipt = accept_ranking(path, rows, output, binding, progress["blocks"])
        receipt.update(compute_s=progress["compute_s"], save_s=progress["save_s"],
                       readback_s=time.monotonic()-tick, current_process_wall_s=time.monotonic()-started,
                       ranking_path=str(path.resolve()), blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"))
        if method == "mr-fixed-kernel":
            receipt["solve_statistics_sha256"] = digest(output/"mr-solve-statistics.json")
        save_json(output/"FULL_ACCEPTANCE.json", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--backbone", choices=SPECS, required=True)
    parser.add_argument("--method", choices=PARAMETERS, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.method == "mr-fixed-kernel" and not args.cache:
        parser.error("MR requires --cache")
    try:
        run(args.frozen, args.backbone, args.cache, args.method, args.output)
    except BlockingIOError:
        raise
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        save_json(args.output/f"FAILED-{time.time_ns()}.json", dict(error=traceback.format_exc()))
        raise
