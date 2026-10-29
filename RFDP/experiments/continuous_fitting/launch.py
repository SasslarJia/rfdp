"""Sequential compute, paired analysis and full read-back; no automatic retry."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse,json,resource,subprocess,sys,traceback
from pathlib import Path
from . import full


def analysis_limits():
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))


def run_analysis(command,stream):
    initial=full.resources()
    assert initial['memory_current']+10*2**30<=.7*initial['memory_max'],'Analysis memory reservation'
    process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,preexec_fn=analysis_limits)
    try:
        while True:
            try:return process.wait(timeout=20)
            except subprocess.TimeoutExpired:full.resources()
    except BaseException:
        process.terminate();process.wait();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.config.read_text());out=Path(cfg['output']);logs=out.with_name(out.name+'-control')
    full.configure(args.config)
    logs.mkdir(exist_ok=False)
    stages=[('compute',[sys.executable,'-B','-m','experiments.continuous_fitting.full','compute','--config',str(args.config.resolve())]),
            ('analysis',[sys.executable,'-B','-m','experiments.internal_nonlinear.runtime.analyze_route','--run',str(out),'--baseline-queries',cfg['baseline_queries'],'--baseline-summary',cfg['baseline_summary'],'--reference-name','original-IHy','--baseline-feature-sha',cfg['feature_sha256']]),
            ('acceptance',[sys.executable,'-B','-m','experiments.continuous_fitting.full','acceptance','--config',str(args.config.resolve())])]
    rc=1
    for stage,command in stages:
        with (logs/(stage+'.log')).open('w') as stream:
            try:rc=run_analysis(command,stream) if stage=='analysis' else subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT).returncode
            except BaseException:
                traceback.print_exc(file=stream);rc=1
        (logs/(stage+'.exit')).write_text(str(rc)+'\n')
        if out.exists():(out/(stage+'.exit')).write_text(str(rc)+'\n')
        if rc:break
    (logs/'total.exit').write_text(str(rc)+'\n')
    if out.exists():(out/'total.exit').write_text(str(rc)+'\n')
    return rc

if __name__=='__main__':sys.exit(main())
