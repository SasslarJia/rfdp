"""Freeze the two MSMT17 inputs in official row order, then verify their CPU handoff."""
import argparse
import fcntl
import hashlib
import io
import json
import os
import platform
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import numpy as np

from .models import SPECS, check_assets, digest, load_model, pinned_assets


COUNTS = {"query": 11659, "gallery": 82161}
TOTAL = sum(COUNTS.values())
TEST_DIR = {"V1": "test", "V2": "mask_test_v2"}
CHUNK = 1024  # A multiple of both fixed forward batch sizes.
NORM_ATOL = 1e-5


def save_json(path, value):
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as tmp:
        json.dump(value, tmp, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)


def read_json(path):
    return json.loads(path.read_text())


@contextmanager
def writer(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".extract.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def parse_list(content, split, count):
    rows = []
    for line in content.decode("utf-8-sig").splitlines():
        relative, pid_text = line.rsplit(maxsplit=1)
        rel = PurePosixPath(relative)
        if rel.is_absolute() or ".." in rel.parts or rel.suffix.lower() != ".jpg":
            raise ValueError(f"Unexpected relative image path: {relative}")
        cam = int(rel.name.split("_")[2])
        pid = int(pid_text)
        if not 1 <= cam <= 15 or pid < 0:
            raise ValueError(f"Invalid official metadata: {line}")
        rows.append(dict(split=split, path=str(rel), pid=pid, camera_original=cam, camid=cam-1))
    if len(rows) != count:
        raise ValueError(f"{split}: {len(rows)} rows; expected {count}")
    return rows


def validate_rows(rows):
    if len(rows) != TOTAL or len({r["path"] for r in rows}) != TOTAL:
        raise ValueError("Expected 93,820 distinct official image paths")
    if len({r["pid"] for r in rows}) != 3060 or {r["camid"] for r in rows} != set(range(15)):
        raise ValueError("Expected 3,060 test identities and 15 cameras")
    for i, r in enumerate(rows):
        if r["split"] != ("query" if i < COUNTS["query"] else "gallery"):
            raise ValueError(f"Split order differs at row {i}")


def data_root(base, version):
    root = base if (base / "list_query.txt").is_file() else base / ("MSMT17_" + version)
    if not (root / TEST_DIR[version]).is_dir():
        raise ValueError(f"Expected {TEST_DIR[version]} and both official lists in {root}")
    return root


def inventory(base, version, output):
    root = data_root(base, version)
    rows, lists = [], {}
    for split, count in COUNTS.items():
        raw = (root / f"list_{split}.txt").read_bytes()
        lists[split] = hashlib.sha256(raw).hexdigest()
        rows.extend(parse_list(raw, split, count))
    validate_rows(rows)
    path = output / "image-manifest.json"
    receipt = output / "dataset.json"
    if receipt.exists():
        meta = read_json(receipt)
        if meta["version"] != version or meta["list_sha256"] != lists or digest(path) != meta["manifest_sha256"]:
            raise ValueError("Output belongs to a different image manifest or release")
        return root / TEST_DIR[version], read_json(path), meta
    images = root / TEST_DIR[version]
    for row in rows:
        row["sha256"] = digest(images / row["path"])
    save_json(path, rows)
    meta = dict(version=version, counts=COUNTS, test_dir=TEST_DIR[version],
                list_sha256=lists, manifest_sha256=digest(path))
    save_json(receipt, meta)
    return images, rows, meta


class Images:
    def __init__(self, root, rows, backbone, start):
        from torchvision import transforms
        _, _, mean, std = SPECS[backbone]
        self.root, self.rows, self.start = root, rows, start
        self.transform = transforms.Compose([
            transforms.Resize((256, 128), interpolation=transforms.InterpolationMode.BILINEAR,
                              antialias=True),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])

    def __len__(self):
        return len(self.rows) - self.start

    def __getitem__(self, offset):
        from PIL import Image
        i = self.start + offset
        row = self.rows[i]
        raw = (self.root / row["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"Image bytes changed at official row {i}")
        with Image.open(io.BytesIO(raw)) as image:
            return i, self.transform(image.convert("RGB")), row["camid"]


def check_vectors(values):
    if values.dtype != np.float32 or values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite FP32 feature rows")
    deviation = float(np.abs(np.linalg.norm(values.astype(np.float64), axis=1) - 1).max())
    if deviation > NORM_ATOL:
        raise ValueError(f"Feature L2 norm differs by {deviation}")
    return deviation


def block_hash(block):
    return hashlib.sha256(memoryview(np.ascontiguousarray(block)).cast("B")).hexdigest()


def open_progress(folder, shape, signature):
    progress = folder / "progress.json"
    target = folder / "features.npy"
    if progress.exists():
        state = read_json(progress)
        if state["signature"] != signature:
            raise ValueError(f"Resume signature differs: {folder}")
    else:
        if target.exists():
            raise ValueError(f"Feature file without a progress record: {target}")
        state = dict(signature=signature, chunks=[], compute_seconds=0., write_seconds=0.,
                     peak_allocated_bytes=0, peak_reserved_bytes=0)
        save_json(progress, state)
    if target.exists():
        matrix = np.load(target, mmap_mode="r+", allow_pickle=False)
    elif not state["chunks"]:
        matrix = np.lib.format.open_memmap(target, mode="w+", dtype=np.float32, shape=shape)
    else:
        raise ValueError("Committed feature file is missing")
    if matrix.shape != shape or matrix.dtype != np.float32:
        raise ValueError("Feature file shape or dtype differs")
    start = 0
    for end, expected in state["chunks"]:
        if end != min(start + CHUNK, shape[0]) or start >= end or block_hash(matrix[start:end]) != expected:
            raise ValueError(f"Corrupted committed feature rows at {start}")
        start = end
    return matrix, state, start


def flush_chunk(folder, matrix, state, start, end, compute_seconds):
    clock = time.monotonic()
    matrix.flush()
    with (folder / "features.npy").open("rb") as handle:
        os.fsync(handle.fileno())
    state["chunks"].append([end, block_hash(matrix[start:end])])
    state["compute_seconds"] += compute_seconds
    state["write_seconds"] += time.monotonic() - clock
    save_json(folder / "progress.json", state)


def audit_array(path, shape):
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if matrix.shape != shape or matrix.dtype != np.float32:
        raise ValueError(f"Invalid feature matrix {path}: {matrix.shape}, {matrix.dtype}")
    err = max(check_vectors(matrix[i:i + CHUNK]) for i in range(0, shape[0], CHUNK))
    return digest(path), err


def runtime():
    import torch
    import torchvision
    import PIL
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                torchvision=str(torchvision.__version__), pillow=PIL.__version__, numpy=np.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()))


def software_hashes():
    return {name: digest(Path(__file__).with_name(name)) for name in ("extract.py", "models.py", "assets.json")}


def extract_one(backbone, assets, images, rows, meta, output, workers):
    import torch
    from torch.utils.data import DataLoader
    dim, batch, _, _ = SPECS[backbone]
    folder = output / backbone
    folder.mkdir(exist_ok=True)
    signature = dict(backbone=backbone, batch=batch, dimensions=dim,
                     count=len(rows), dataset=meta, runtime=runtime(),
                     assets={k: v["sha256"] for k, v in pinned_assets().items()},
                     source=software_hashes())
    if (folder / "COMPLETE.json").exists():
        complete = verify_one(folder, backbone, meta)
        if complete["signature"] != signature:
            raise ValueError("Existing features belong to a different extraction configuration")
        return complete["feature_sha256"]
    matrix, progress, start = open_progress(folder, (len(rows), dim), signature)
    if start < len(rows):
        model = load_model(backbone, assets).cuda()
        loader = DataLoader(Images(images, rows, backbone, start), batch_size=batch, shuffle=False,
                            drop_last=False, num_workers=workers, pin_memory=True)
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        offset = chunk_start = start
        with torch.inference_mode():
            for indices, pixels, cameras in loader:
                end = offset + len(indices)
                if not np.array_equal(indices.numpy(), np.arange(offset, end)):
                    raise ValueError(f"Image ordering at {offset}")
                pixels = pixels.cuda(non_blocking=True)
                if backbone == "transreid":
                    cam = cameras.cuda(non_blocking=True)
                    features = model(pixels, cam_label=cam, view_label=torch.zeros_like(cam))
                else:
                    features = model(pixels)
                block = torch.nn.functional.normalize(features, dim=1).cpu().numpy()
                if block.shape != (end-offset, dim):
                    raise ValueError("Wrong backbone output shape")
                check_vectors(block)
                matrix[offset:end] = block
                offset = end
                if end - chunk_start == CHUNK or end == len(rows):
                    progress["peak_allocated_bytes"] = max(progress["peak_allocated_bytes"], torch.cuda.max_memory_allocated())
                    progress["peak_reserved_bytes"] = max(progress["peak_reserved_bytes"], torch.cuda.max_memory_reserved())
                    flush_chunk(folder, matrix, progress, chunk_start, end, time.monotonic() - tick)
                    print(f"{backbone}: {end}/{len(rows)} rows committed", flush=True)
                    chunk_start, tick = end, time.monotonic()
        if offset != len(rows):
            raise ValueError("Inference ended before the last gallery image")
        del model
        torch.cuda.empty_cache()
    tick = time.monotonic()
    sha, deviation = audit_array(folder / "features.npy", (len(rows), dim))
    save_json(folder / "COMPLETE.json", dict(signature=signature, feature_sha256=sha,
              max_norm_error=deviation, acceptance_seconds=time.monotonic()-tick,
              extraction_and_copy_seconds=progress["compute_seconds"], flush_seconds=progress["write_seconds"],
              peak_allocated_bytes=progress["peak_allocated_bytes"], peak_reserved_bytes=progress["peak_reserved_bytes"]))
    return sha


def verify_one(folder, backbone, meta):
    complete = read_json(folder / "COMPLETE.json")
    sig = complete["signature"]
    dim, batch, _, _ = SPECS[backbone]
    if (sig["dataset"] != meta or sig["backbone"] != backbone or sig["dimensions"] != dim
            or sig["count"] != TOTAL or sig["batch"] != batch
            or sig["assets"] != {k: v["sha256"] for k, v in pinned_assets().items()}
            or sig["source"] != software_hashes()):
        raise ValueError(f"Feature provenance differs: {backbone}")
    sha, _ = audit_array(folder / "features.npy", (TOTAL, dim))
    if sha != complete["feature_sha256"]:
        raise ValueError(f"Feature SHA256 differs: {backbone}")
    return complete


def verify(output):
    meta = read_json(output / "dataset.json")
    manifest = output / "image-manifest.json"
    if digest(manifest) != meta["manifest_sha256"] or meta["counts"] != COUNTS:
        raise ValueError("Shared image manifest differs")
    validate_rows(read_json(manifest))
    return {name: verify_one(output / name, name, meta)["feature_sha256"] for name in SPECS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("models", "inventory", "run", "verify"):
        p = commands.add_parser(action)
        if action in ("models", "run"):
            p.add_argument("--assets", type=Path, required=True)
        if action != "models":
            p.add_argument("--out", type=Path, required=True)
        if action in ("inventory", "run"):
            p.add_argument("--data", type=Path, required=True)
            p.add_argument("--version", required=True, choices=TEST_DIR)
        if action == "run":
            p.add_argument("--io-workers", type=int, default=4)
    args = parser.parse_args()
    if args.action == "verify":
        actual = verify(args.out)
        handoff = read_json(args.out / "HANDOFF.json")
        expected = dict(features=actual, dataset_sha256=digest(args.out / "dataset.json"),
                        receipts={n: digest(args.out/n/"COMPLETE.json") for n in SPECS})
        if handoff != expected:
            raise ValueError("Handoff receipts differ")
        print(json.dumps(handoff))
        return
    if args.action in ("models", "run"):
        check_assets(args.assets)
        import torch
        torch.set_num_threads(2)
        if args.action == "models":
            for name in SPECS:
                model = load_model(name, args.assets)
                print(f"{name}: strict load, {sum(p.numel() for p in model.parameters())} parameters", flush=True)
                del model
            return
        if not torch.cuda.is_available() or args.io_workers < 0:
            raise RuntimeError("CUDA and a nonnegative I/O worker count are required")
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    with writer(args.out):
        images, rows, meta = inventory(args.data, args.version, args.out)
        if args.action == "inventory":
            print(json.dumps(meta))
            return
        sums = {name: extract_one(name, args.assets, images, rows, meta, args.out, args.io_workers)
                for name in SPECS}
        save_json(args.out / "HANDOFF.json", dict(features=sums, dataset_sha256=digest(args.out/"dataset.json"),
                  receipts={n: digest(args.out/n/"COMPLETE.json") for n in SPECS}))
        print(json.dumps(sums), flush=True)


if __name__ == "__main__":
    main()
