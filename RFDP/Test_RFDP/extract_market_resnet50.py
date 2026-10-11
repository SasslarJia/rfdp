"""Export deterministic Market1501 features for the RFDP .mat interface.

Uses the unmodified resnet.py from KaiyangZhou/deep-person-reid and an
explicit same-domain ReID checkpoint, rather than ImageNet-only weights.
"""

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy.io
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def image_records(folder):
    records = []
    for path in sorted(Path(folder).glob("*.jpg")):
        match = re.match(r"(-?\d+)_c([1-6])", path.name)
        if match is None:
            raise ValueError(f"Unexpected Market1501 filename: {path.name}")
        records.append((path, int(match.group(1)), int(match.group(2))))
    if not records:
        raise ValueError(f"No images in {folder}")
    return records


class Images(Dataset):
    def __init__(self, records):
        self.records = records
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with Image.open(self.records[index][0]) as im:
            return self.transform(im.convert("RGB"))


def load_model(upstream, weights):
    path = Path(upstream) / "torchreid/models/resnet.py"
    spec = importlib.util.spec_from_file_location("torchreid_resnet_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # This official 2021 checkpoint includes a NumPy scalar in its metadata.
    # Permit legacy loading only for the exact file downloaded from Model Zoo.
    expected_hash = "15d3fab6912688516ed2c281961eaf7ee8590ff51dfd6f61ba64daba90325c31"
    if sha256(weights) != expected_hash:
        raise ValueError("Checkpoint SHA256 differs from the verified official ResNet50 file")
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False, encoding="latin1")
    state = checkpoint.get("state_dict", checkpoint)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    if "classifier.weight" not in state or state["classifier.weight"].shape != (751, 2048):
        raise ValueError("Expected plain ResNet50 Market1501 checkpoint (751 classes, 2048 features)")
    model = module.resnet50(num_classes=751, pretrained=False)
    model.load_state_dict(state, strict=True)
    return model.eval().cuda(), path


def extract(model, records, batch_size, workers, name):
    loader = DataLoader(Images(records), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    blocks = []
    completed = 0
    with torch.inference_mode():
        for images in loader:
            raw = model(images.cuda(non_blocking=True))
            if raw.ndim != 2 or raw.shape[1] != 2048 or not torch.isfinite(raw).all():
                raise ValueError("Invalid ResNet50 embeddings")
            if torch.any(torch.linalg.vector_norm(raw, dim=1) <= 1e-12):
                raise ValueError("Zero embedding")
            features = torch.nn.functional.normalize(raw, p=2, dim=1)
            blocks.append(features.cpu().numpy().astype(np.float32, copy=False))
            completed += len(images)
            print(f"{name}: {completed}/{len(records)}", flush=True)
    result = np.concatenate(blocks)
    if not np.allclose(np.linalg.norm(result, axis=1), 1, atol=1e-5):
        raise ValueError("Features are not L2 normalized")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only for checking extraction, never a full benchmark")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        raise FileExistsError("Choose a new output path to preserve previous features")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this extraction job")
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    start = time.monotonic()
    query = image_records(args.dataset / "query")
    gallery = image_records(args.dataset / "bounding_box_test")
    counts = {"query": len(query), "gallery": len(gallery)}
    if args.limit is None and counts != {"query": 3368, "gallery": 19732}:
        raise ValueError(f"Unexpected standard Market1501 counts: {counts}")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        query, gallery = query[:args.limit], gallery[:args.limit]
    model, model_source = load_model(args.upstream, args.weights)
    torch.cuda.reset_peak_memory_stats()
    qf = extract(model, query, args.batch_size, args.workers, "query")
    gf = extract(model, gallery, args.batch_size, args.workers, "gallery")
    fields = {"query_f": qf, "gallery_f": gf,
              "query_label": np.array([r[1] for r in query], dtype=np.int32),
              "gallery_label": np.array([r[1] for r in gallery], dtype=np.int32),
              "query_cam": np.array([r[2] for r in query], dtype=np.int32),
              "gallery_cam": np.array([r[2] for r in gallery], dtype=np.int32)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(args.output, fields, do_compression=False)
    commit = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    manifest = {"benchmark": "Market1501 single-query", "full_dataset_counts": counts,
                "development_limit": args.limit, "upstream_commit": commit,
                "model_source_sha256": sha256(model_source),
                "checkpoint": str(args.weights), "checkpoint_sha256": sha256(args.weights),
                "feature_sha256": sha256(args.output), "script_sha256": sha256(__file__),
                "model": "torchreid resnet50, last_stride=2, no fc, 751 training classes",
                "preprocessing": "RGB, resize 256x128 bilinear antialias, ImageNet mean/std, no flip, float32, L2 normalization",
                "gallery_policy": "Retain pid=0 distractors and pid=-1 junk; evaluator removes junk",
                "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(), "batch_size": args.batch_size,
                "workers": args.workers, "elapsed_seconds": time.monotonic() - start,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "shapes": {k: list(v.shape) for k, v in fields.items()},
                "query_paths": [r[0].name for r in query],
                "gallery_paths": [r[0].name for r in gallery]}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if not k.endswith("paths")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
