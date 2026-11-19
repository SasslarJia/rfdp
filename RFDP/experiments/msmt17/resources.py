"""Resource accounting for the Linux cgroup holding the MSMT17 experiment."""
import shutil
from pathlib import Path

GIB = 2**30
DISK_MARGIN_BYTES = 512 * 2**20


def require_disk_space(path, output_bytes):
    """Reserve the known output bytes plus a small allowance for filesystem writes."""
    free = shutil.disk_usage(path).free
    if free < output_bytes + DISK_MARGIN_BYTES:
        raise OSError(f"Insufficient space at {path}: {free} bytes free, "
                      f"{output_bytes} output bytes and {DISK_MARGIN_BYTES} bytes of headroom needed")


def cloud_state(paths):
    cgroup = Path("/sys/fs/cgroup")
    cap = int((cgroup/"memory.max").read_text())
    used = int((cgroup/"memory.current").read_text())
    if used > cap*.7:
        raise MemoryError(f"Cloud RAM reserve reached ({used}/{cap} bytes)")
    disks = {}
    for path in set(paths):
        d = shutil.disk_usage(path)
        floor = DISK_MARGIN_BYTES
        if d.free < floor:
            raise OSError(f"Disk reserve reached at {path}: {d.free} < {floor}")
        disks[str(path)] = dict(free=d.free, floor=floor, total=d.total)
    return dict(memory_current=used, memory_limit=cap, disks=disks)
