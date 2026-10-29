"""Shared IO/resources; no retrieval or numerical certificate implementation."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,hashlib,datetime,resource,shutil,subprocess,sys,threading,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE/'attempt-1'
ROOT=Path('/root/autodl-tmp/rfdp-graph-routes-20261026')
QUERIES=[0,38,481,962,1443,1924,2405,2886,3367]
ARMS=[('theta0',0.),('theta0p5',.5)]
BUDGET=2*2**30
def now():return datetime.datetime.now().astimezone().isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()
def dump(p,obj):
    p=Path(p);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(p)
def canonical(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def check_sources():
    lock=json.loads((HERE/'source-lock.json').read_text())
    for n,h in lock['files'].items():assert sha(HERE/n)==h,n
    for n,h in lock['dependencies'].items():assert sha(HERE/n)==h,n
    refs=json.loads((HERE/'references.json').read_text())
    for n,h in refs['locked_files'].items():assert sha(n)==h,n
    return refs
def resources(admission=False):
    cg=Path('/sys/fs/cgroup')
    m=int((cg/'memory.current').read_text());cap=int((cg/'memory.max').read_text())
    assert m+(10*2**30 if admission else 0)<=.7*cap,'RAM limit/reserve'
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    assert peak<=8*2**30,'Single-worker RSS stop'
    allocated=sum(p.stat().st_blocks*512 for p in OUT.rglob('*') if p.is_file()) if OUT.exists() else 0
    assert allocated<=BUDGET,'Qualification output cap; stop without truncation'
    remaining=BUDGET-allocated;disks={}
    for n in ['/','/root/autodl-tmp']:
        total,used,free=shutil.disk_usage(n);extra=remaining if n=='/root/autodl-tmp' else 0
        assert free-extra>=max(total*.25,10*2**30),'Disk floor with remaining output reserved'
        disks[n]=dict(total=total,free=free)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip()
    for line in gpu.splitlines():
        used,total=map(int,line.split(','));assert used<=.75*total,'GPU limit'
    return dict(time=now(),memory_current=m,memory_max=cap,cpu_max=(cg/'cpu.max').read_text().strip(),
                peak_rss_bytes=peak,allocated_bytes=allocated,remaining_output_budget_bytes=remaining,gpu=gpu,disks=disks)
def prerequisite():
    src=ROOT/'cas-strength-full-lambda2-v1-20261028'
    assert all((src/(k+'.exit')).read_text().strip()=='0' for k in ['compute','analysis','analysis-lambda1','acceptance','total'])
    accepted=ROOT/'market-relation-mass-lambda2-v1/FULL_ACCEPTANCE.json'
    a=json.loads(accepted.read_text());assert a
    # Created only after the completed population result has actually been reflected on.
    reflection=json.loads((HERE/'prerequisite-reflection.json').read_text())
    assert reflection['lambda2_acceptance_sha256']==sha(accepted)
    assert reflection['reflection_complete'] is True and reflection['new_model_unchanged'] is True
    assert sha(reflection['reflection_path'])==reflection['reflection_sha256']
    active=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:argv=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
        except (FileNotFoundError,PermissionError):continue
        if not argv:continue
        exe=Path(argv[0]).name
        if (('python' in exe or exe in ('tar','scp','rsync','gzip','pigz')) and
            any('rfdp-' in s for s in argv)):active.append(dict(pid=int(p.name),argv=argv))
    assert not active,active
    return dict(acceptance_sha256=sha(accepted),reflection_sha256=sha(HERE/'prerequisite-reflection.json'),all_argv_active=active)
def start_guard(stage):
    stop=threading.Event()
    def loop():
        while not stop.wait(10):
            try:dump(OUT/'latest-resources.json',resources())
            except BaseException:
                dump(OUT/'FAILED.json',dict(stage=stage,kind='resource_stop',time=now(),error=traceback.format_exc()))
                (OUT/(stage+'.exit')).write_text('2\n');os._exit(2)
    thread=threading.Thread(target=loop,daemon=True);thread.start()
    return stop,thread
