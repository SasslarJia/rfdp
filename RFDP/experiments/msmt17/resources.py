"""Resource accounting for the Linux cgroup holding the MSMT17 experiment."""
import shutil
from pathlib import Path

GIB = 2**30


def cloud_state(paths):
    cgroup = Path("/sys/fs/cgroup")
    cap = int((cgroup/"memory.max").read_text())
    used = int((cgroup/"memory.current").read_text())
    if used > cap*.7:
        raise MemoryError(f"Cloud RAM reserve reached ({used}/{cap} bytes)")
    disks = {}
    for path in set(paths):
        d = shutil.disk_usage(path)
        floor = max(d.total//4, 10*GIB)
        if d.free < floor:
            raise OSError(f"Disk reserve reached at {path}: {d.free} < {floor}")
        disks[str(path)] = dict(free=d.free, floor=floor, total=d.total)
    return dict(memory_current=used, memory_limit=cap, disks=disks)
