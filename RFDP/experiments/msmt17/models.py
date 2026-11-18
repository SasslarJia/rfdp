"""Frozen same-domain author models. The feature handoff does not need PyTorch."""
import collections.abc
import hashlib
import importlib.util
import json
import sys
import types
import zipfile
from pathlib import Path


TRANS_REV = "dec55046fcdfadee14e2c28e2df89305d8f7557a"
SPECS = {
    "osnet": (512, 128, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "transreid": (3840, 16, [0.5] * 3, [0.5] * 3),
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def pinned_assets():
    return json.loads(Path(__file__).with_name("assets.json").read_text())


def check_assets(folder):
    pins = pinned_assets()
    for name, spec in pins.items():
        path = folder / name
        if path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
            raise ValueError(f"Author asset differs: {name}")
    return {name: spec["sha256"] for name, spec in pins.items()}


def load_model(backbone, folder):
    import torch

    if backbone == "osnet":
        import numpy as np
        spec = importlib.util.spec_from_file_location("frozen_msmt_osnet", folder / "osnet.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.osnet_x1_0(num_classes=1041, pretrained=False)
        # The original training checkpoint stores a NumPy metric alongside its tensors.
        def scalar(*args):
            return np.core.multiarray.scalar(*args)
        scalar.__module__ = "numpy.core.multiarray"
        with torch.serialization.safe_globals(
            [scalar, np.dtype, type(np.dtype("float32")), type(np.dtype("float64"))]
        ):
            checkpoint = torch.load(folder / "osnet.pth", map_location="cpu",
                                    weights_only=True, encoding="latin1")
    else:
        from yacs.config import CfgNode
        archive = (folder / "transreid.zip").resolve()
        prefix = f"TransReID-{TRANS_REV}"
        package = str(archive) + "/" + prefix
        if package not in sys.path:
            sys.path.insert(0, package)
        compat = types.ModuleType("torch._six")
        compat.container_abcs = collections.abc
        sys.modules.setdefault("torch._six", compat)
        from config import cfg as defaults
        from model import make_model
        with zipfile.ZipFile(archive) as upstream:
            yaml = upstream.read(prefix + "/configs/MSMT17/vit_transreid_stride.yml").decode()
        cfg = defaults.clone()
        cfg.merge_from_other_cfg(CfgNode.load_cfg(yaml))
        cfg.MODEL.PRETRAIN_CHOICE = "none"
        cfg.MODEL.DEVICE_ID = "0"
        cfg.freeze()
        model = make_model(cfg, num_class=1041, camera_num=15, view_num=1)
        checkpoint = torch.load(folder / "transreid.pth", map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    return model.eval()
