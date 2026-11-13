"""CPU-only WSL limits; Linux virtual disk and backing Windows disks are separate."""
import math
import os
import shutil
from pathlib import Path

GIB=2**30


def cpu_allocation(cgroup=Path('/sys/fs/cgroup')):
    visible=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count()
    quota=cgroup/'cpu.max'
    if quota.exists():
        maximum,period=quota.read_text().split()
        allocated=visible if maximum=='max' else min(visible,max(1,math.ceil(int(maximum)/int(period))))
    else:
        allocated=visible
        for name in ('cpu','cpu,cpuacct'):
            parent=cgroup/name
            if (parent/'cpu.cfs_quota_us').exists():
                q=int((parent/'cpu.cfs_quota_us').read_text());p=int((parent/'cpu.cfs_period_us').read_text())
                if q>0:allocated=min(allocated,max(1,math.ceil(q/p)))
                break
    return dict(visible=int(visible),allocated=int(allocated),workers=min(12,int(allocated)),blas_threads=1)


def memory_allocation(cgroup=Path('/sys/fs/cgroup'), meminfo=Path('/proc/meminfo')):
    values={line.split(':')[0]:int(line.split()[1])*1024 for line in meminfo.read_text().splitlines() if ':' in line}
    total=values['MemTotal'];available=values['MemAvailable']
    cap=min(total,24*GIB);current=total-available;source='WSL MemTotal/MemAvailable'
    candidates=[(cgroup/'memory.max',cgroup/'memory.current'),
                (cgroup/'memory/memory.limit_in_bytes',cgroup/'memory/memory.usage_in_bytes')]
    for limit,used in candidates:
        if limit.exists():
            text=limit.read_text().strip()
            if text!='max' and int(text)<2**60:
                cap=min(cap,int(text));current=max(current,int(used.read_text()));source+=' and cgroup'
                break
    return dict(cap=int(cap),current=int(current),available=int(available),source=source)


def native_path(path):
    p=Path(path).resolve()
    if str(p).startswith('/mnt/'):
        raise ValueError('Active run/cache must use native Linux filesystem: '+str(p))
    return p


def snapshot(cfg,reserve=False):
    cpu=cpu_allocation();mem=memory_allocation()
    if cfg['workers']!=cpu['workers']:
        raise RuntimeError('CPU allocation changed; rebind the WSL profile before running')
    if mem['current']>.7*mem['cap']:
        raise RuntimeError('Memory exceeds the 70 percent reserve line')
    out=native_path(cfg['output']);native_path(cfg['cache']);native_path(cfg['workspace'])
    allocated=sum(p.stat().st_blocks*512 for p in out.rglob('*') if p.is_file()) if out.exists() else 0
    if allocated>cfg['output_budget_bytes']:
        raise RuntimeError('Output budget reached; do not truncate rankings/traces')
    disks={}
    paths=[str(Path(cfg['workspace']).resolve())]+cfg['host_storage_roots']
    for path in dict.fromkeys(paths):
        if not Path(path).exists():raise RuntimeError('Storage guard path is not mounted: '+path)
        d=shutil.disk_usage(path);floor=max(d.total//4,10*GIB)
        # The backing C drive may fill before ext4's advertised VHD maximum.
        remaining=max(0,cfg['output_budget_bytes']-allocated) if reserve else 0
        if d.free-remaining<floor:
            raise RuntimeError('Disk reserve floor on '+path)
        disks[path]=dict(total=d.total,free=d.free,floor=floor)
    return dict(memory_current=mem['current'],memory_max=mem['cap'],memory_source=mem['source'],
                cpu=cpu,disks=disks,allocated_bytes=allocated,
                output_budget_bytes=cfg['output_budget_bytes'],execution='CPU only; no CUDA or nvidia-smi')


def reservation(cfg, state):
    shared=sum((Path(cfg['cache'])/n).stat().st_size for n in ('affinity.npy','features.npy'))
    # Address-space budget subtracts explicitly shared immutable mappings;
    # this is a conservative reservation, in addition to the live memory guard.
    private=cfg['worker_address_space_limit_bytes']-shared
    if private<=0:raise RuntimeError('Worker limit cannot map complete inputs')
    total=state['memory_current']+shared+cfg['workers']*private+2*GIB
    if total>.7*state['memory_max']:
        raise RuntimeError('Complete worker/input memory reservation exceeds 70 percent; no lower-precision fallback')
    return dict(shared_mapping_bytes=shared,static_total_bytes=total)
