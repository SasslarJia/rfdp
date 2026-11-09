"""Run one prepared candidate, then return for result-dependent research review.

May be copied outside a frozen execution checkout as an operational launcher;
PYTHONPATH must name that checkout's RFDP directory. No equation is redefined.
"""
import argparse
import json
import traceback
from pathlib import Path
from experiments.relation_routes import protocol
from experiments.relation_routes.full_run import lock,checkout,run_one
from experiments.relation_routes.report import collect
from experiments.continuous_fitting import full


def launch(root,route,resume=False):
    root=Path(root).resolve()
    if route not in protocol.ROUTES:raise ValueError('Unknown active route')
    with lock(root/'.execution.lock'):
        prepared=json.loads((root/'PREPARED.json').read_text())
        if prepared['commit']!=checkout():raise ValueError('Prepared checkout differs')
        for name,value in prepared['files_sha256'].items():
            if full.sha(root/name)!=value:raise ValueError('Prepared input/config changed: '+name)
        status=root/f'{route}.status.json'
        try:
            full.dump(status,dict(status='running',route=route,time=full.now()))
            result=run_one(root/'configs'/f'{route}.json',resume)
            collect(root,root/'references.json')
            full.dump(status,dict(status=result,route=route,time=full.now(),next='review_full_result_before_transfer'))
        except BaseException:
            failure=dict(status='technical_failure',route=route,time=full.now(),error=traceback.format_exc())
            full.dump(status,failure)
            out=root/'runs'/route
            if out.exists():full.dump(out/'FAILED.json',failure)
            collect(root,root/'references.json')
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--route',choices=protocol.ROUTES,required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();launch(a.root,a.route,a.resume)
