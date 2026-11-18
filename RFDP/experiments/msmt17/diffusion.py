"""Run a fixed-schedule or early-feedback MSMT17 cell on the complete k20 graph."""
import os

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import datetime
import hashlib
import io
import json
import multiprocessing as mp
import resource
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
from scipy import sparse

from experiments.continuous_fitting import full, original_solver
from experiments.reid_adaptation.resources import cpu_allocation
from experiments.reid_adaptation.schedule import ScheduledIHyRDP
from .extract import COUNTS, block_hash, read_json, save_json, writer
from .graph import PackedSymmetric
from .models import SPECS, digest
from .rankings import accept_ranking, frozen_input
from .resources import GIB, cloud_state

PRIVATE_BUDGET = GIB
MEMORY_MARGIN = 2*GIB
EVIDENCE_BUDGET = 2*GIB


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def installed_sources():
    root = Path(__file__).resolve().parents[2]
    folders = ("MODEL/Paper_RFDP", "experiments/continuous_fitting", "experiments/msmt17",
               "experiments/reid_adaptation", "experiments/full_reid", "EVAL")
    return {str(f.relative_to(root)): digest(f) for folder in folders
            for f in sorted((root/folder).glob("*.py"))}


def load_graph(cache, binding):
    meta = read_json(cache/"manifest.json")
    n = sum(COUNTS.values())
    if (meta["input"] != binding or (meta["n"], meta["kd"], meta["ks"], meta["gamma"]) != (n, 20, 20, .4)
            or meta["full_matrix_shape"] != [n, n] or meta["dtype"] != "float32"):
        raise ValueError("The k20 graph is not bound to this official feature population")
    for name in ("features.npy", "row-sums.npy", "graph.npz"):
        if digest(cache/name) != meta["files"][name]:
            raise ValueError(f"Graph input changed: {name}")
    packed_path = Path(meta["packed_path"])
    if digest(packed_path) != meta["packed_sha256"]:
        raise ValueError("Complete packed affinity hash differs")
    z = np.load(cache/"features.npy", mmap_mode="r")
    sums = np.load(cache/"row-sums.npy", mmap_mode="r")
    packed = np.load(packed_path, mmap_mode="r")
    graph = sparse.load_npz(cache/"graph.npz").astype(np.float64).tocsr()
    graph.sort_indices()
    if (not graph.has_canonical_format or graph.shape != (n, n) or graph.nnz != meta["graph"]["nnz"]
            or z.shape != (n, SPECS[binding["backbone"]][0]) or z.dtype != np.float32
            or sums.shape != (n,) or sums.dtype != np.float64):
        raise ValueError("Graph dimensions, sparse structure or feature type differ")
    return graph, PackedSymmetric(packed, n, sums), z


def start_worker():
    # The parent shares immutable NumPy mappings and CSR buffers via fork.
    try:
        full.ENGINE = ScheduledIHyRDP(full.GRAPH, full.FULL, alpha=5., beta=25., step=5,
                                     promotion_schedule=full.CFG["promotion_schedule"],
                                     stage_solver=original_solver.WarmOriginalSolver())
        current_vm = next(int(line.split()[1])*1024 for line in Path("/proc/self/status").read_text().splitlines()
                          if line.startswith("VmSize:"))
        limit = current_vm + PRIVATE_BUDGET
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception:
        # Pool replaces a process when its initializer raises. Propagate its
        # cause through the ordinary query result instead of respawning it.
        full.INIT_ERROR = traceback.format_exc()


def saved_row_hash(row):
    buf = io.BytesIO()
    np.save(buf, row, allow_pickle=False)
    return hashlib.sha256(buf.getbuffer()).hexdigest()


def recover_row(output, table, q, signature):
    """Restore a committed row or let the pool finish an interrupted query."""
    dest = output/"queries"
    row_file = dest/f"{q:04d}.json"
    checked = output/"accepted"/f"{q:04d}.json"
    if not row_file.exists() or not checked.exists():
        return False
    meta = read_json(row_file)
    if meta["signature"] != signature:
        raise ValueError(f"Another input owns query {q}")
    if not meta.get("readback_sha256"):
        return False
    if digest(checked) != meta["readback_sha256"]:
        raise ValueError(f"Saved readback receipt differs for query {q}")
    small = dest/f"{q:04d}.rankings.npy"
    if small.exists():
        if digest(small) != meta["rankings_sha256"]:
            raise ValueError(f"Individual ranking differs for query {q}")
        table[q] = np.load(small, allow_pickle=False)
    if saved_row_hash(table[q]) != meta["rankings_sha256"]:
        raise ValueError(f"Combined ranking differs for query {q}")
    return True


def commit_rows(table, output, pending):
    if not pending:
        return
    table.flush()
    for q in pending:
        (output/"queries"/f"{q:04d}.rankings.npy").unlink(missing_ok=True)
    pending.clear()


def run(frozen, backbone, cache, schedule, output, store, revision):
    with writer(output):
        started = time.monotonic()
        _, rows, binding = frozen_input(frozen, backbone)
        graph, affinity, z = load_graph(cache, binding)
        nq, ng = COUNTS["query"], COUNTS["gallery"]
        store.mkdir(parents=True, exist_ok=True)
        disk_paths = [output, store, Path("/"), Path(affinity.values.filename).parent]
        state = cloud_state(disk_paths)
        available = min(cpu_allocation()["allocated"],
                        int((.7*state["memory_limit"] - state["memory_current"] - MEMORY_MARGIN)//PRIVATE_BUDGET))
        profile = cache/"diffusion-worker-profile.json"
        if profile.exists():
            workers = read_json(profile)["workers"]
        else:
            workers = available
        if not 1 <= workers <= available:
            raise MemoryError(f"Paired worker budget requires {workers}, currently available {available}")
        cfg = dict(source="original", promotion_schedule=schedule, step=5, depth=500, bridge_budget=5,
                   alpha=5., beta=25., numerical_tail=False, retain_initial_state=True,
                   trace_retention="sampled_after_readback", queries=nq, gallery=ng, workers=workers,
                   retained_queries=np.linspace(0, nq-1, 32, dtype=int).tolist())
        sig_data = dict(input=binding, graph_manifest_sha256=digest(cache/"manifest.json"),
                        config=cfg, revision=revision, source_sha256=installed_sources(),
                        solver=original_solver.SOLVER_ID)
        signature = hashlib.sha256(json.dumps(sig_data, sort_keys=True).encode()).hexdigest()
        manifest = output/"run-manifest.json"
        rank_file = store/"rankings.npy"
        if manifest.exists():
            if read_json(manifest)["signature"] != signature:
                raise ValueError("Run signature changed")
        else:
            if rank_file.exists():
                raise FileExistsError(rank_file)
            reserve = {}
            for p, needed in ((output, EVIDENCE_BUDGET), (store, nq*ng*4+4096)):
                device = p.stat().st_dev
                reserve[device] = reserve.get(device, 0)+needed
                disk = shutil.disk_usage(p)
                if disk.free-reserve[device] < max(disk.total//4, 10*GIB):
                    raise OSError(f"Insufficient disk for the full output and retained traces at {p}")
            save_json(manifest, dict(signature=signature, **sig_data, start=utc_now(),
                                     ranking_path=str(rank_file.resolve()), initial_resources=state))
        if not profile.exists():
            save_json(profile, dict(workers=workers, private_allowance_bytes=PRIVATE_BUDGET,
                                    memory_margin_bytes=MEMORY_MARGIN, blas_threads=1))
        if (output/"FULL_ACCEPTANCE.json").is_file():
            if read_json(output/"FULL_ACCEPTANCE.json")["ranking_sha256"] != digest(rank_file):
                raise ValueError("Completed ranking differs")
            return
        for sub in ("queries", "accepted"):
            (output/sub).mkdir(exist_ok=True)
        table = (np.load(rank_file, mmap_mode="r+") if rank_file.exists() else
                 np.lib.format.open_memmap(rank_file, mode="w+", dtype=np.int32, shape=(nq, ng)))
        if table.shape != (nq, ng) or table.dtype != np.int32:
            raise ValueError("Wrong full ranking shape")
        if store.resolve() != output.resolve():
            link = output/"rankings.npy"
            if not link.exists():
                link.symlink_to(rank_file.resolve())
            if link.resolve() != rank_file.resolve():
                raise ValueError("Ranking path differs from manifest")
        full.CFG, full.OUT, full.QN, full.GN, full.SIG = cfg, output, nq, ng, signature
        full.GRAPH, full.FULL, full.X = graph, affinity, z
        full.READBACK_KERNEL, full.READBACK_SUMS = None, affinity.row_sums
        full.INIT_ERROR, full.STAGE_CERTIFIER, full.ADMISSION_CAMERAS = None, None, None
        full.EXECUTION_COMMIT = revision
        pending, todo = [], []
        for q in range(nq):
            if recover_row(output, table, q, signature):
                pending.append(q)
                if len(pending) >= 32:
                    commit_rows(table, output, pending)
            else:
                todo.append(q)
        commit_rows(table, output, pending)
        finished = nq-len(todo)
        if todo:
            with mp.get_context("fork").Pool(workers, initializer=start_worker) as pool:
                results = pool.imap_unordered(full.compute_query, todo, chunksize=1)
                monitor = 0.
                while finished < nq:
                    if time.monotonic() >= monitor:
                        state = cloud_state(disk_paths)
                        save_json(output/"progress.json", dict(phase="compute_and_readback", completed=finished,
                                  queries=nq, workers=workers, current_time=utc_now(), resources=state,
                                  wall_s=time.monotonic()-started))
                        monitor = time.monotonic()+20.
                    try:
                        result = results.next(timeout=10)
                    except mp.TimeoutError:
                        continue
                    q = result["query"]
                    if "error" in result:
                        raise RuntimeError(f"Query {q}: {result['error']}")
                    if not recover_row(output, table, q, signature):
                        raise RuntimeError(f"Missing readback receipt for query {q}")
                    finished += 1
                    pending.append(q)
                    if len(pending) >= 32:
                        commit_rows(table, output, pending)
            commit_rows(table, output, pending)
        table.flush()
        block_digests = [[min(i+64, nq), block_hash(table[i:i+64])] for i in range(0, nq, 64)]
        del table
        save_json(output/"COMPUTE_COMPLETE.json", dict(signature=signature, rankings_sha256=digest(rank_file)))
        readback_s = full.aggregate_inline_readback(configuration=cfg, output=output, remove_duplicates=False)
        tick = time.monotonic()
        receipt = accept_ranking(rank_file, rows, output, sig_data, block_digests)
        sums = dict(compute_seconds=0., saving_seconds=0., readback_seconds=0.)
        max_residual = max_frontier = max_rss = 0
        for q in range(nq):
            item = read_json(output/"queries"/f"{q:04d}.json")
            for key in sums:
                sums[key] += item[key]
            diag = item["diagnostics"]
            if diag["numerical_tail_count"] or diag["status"] != "complete_diffusion":
                raise ValueError(f"Unfinished dynamic diffusion at query {q}")
            max_residual = max(max_residual, diag["max_relative_residual"])
            max_frontier = max(max_frontier, diag["max_unlabeled"])
            max_rss = max(max_rss, item["peak_rss_bytes"])
        if max_residual > 1e-8:
            raise ValueError("Solver residual exceeds the frozen tolerance")
        receipt.update(finished=utc_now(), signature=signature, ranking_path=str(rank_file.resolve()),
                       saved_scores_readback_queries=nq, retained_full_traces=32,
                       complete_initial_states=nq, numerical_tail_count=0, max_relative_residual=max_residual,
                       max_unlabeled=max_frontier, max_worker_rss_bytes=max_rss,
                       summed_worker_times=sums, aggregate_trace_check_s=readback_s,
                       final_evaluation_and_audit_s=time.monotonic()-tick, current_process_wall_s=time.monotonic()-started)
        save_json(output/"progress.json", dict(phase="accepted", completed=nq, queries=nq, blocks=block_digests))
        save_json(output/"FULL_ACCEPTANCE.json", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--backbone", choices=SPECS, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--schedule", choices=("fixed5", "early5"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank-store", type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    try:
        run(args.frozen, args.backbone, args.cache, args.schedule, args.output,
            args.rank_store or args.output, args.revision)
    except BlockingIOError:
        raise
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output/f"FAILED-{time.time_ns()}.json"
        save_json(path, dict(time=utc_now(), error=traceback.format_exc()))
        raise
