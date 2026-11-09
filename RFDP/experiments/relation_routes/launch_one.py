"""Run one prepared candidate, then return for result-dependent research review.

May be copied outside a frozen execution checkout as an operational launcher;
PYTHONPATH must name that checkout's RFDP directory. No equation is redefined.
"""
import argparse
import json
import traceback
from pathlib import Path
from experiments.relation_routes import protocol
from experiments.relation_routes.full_run import lock,checkout,run_one,accepted_result
from experiments.relation_routes.report import collect
from experiments.continuous_fitting import full


def transfer_gate(root,route,decision_path):
    """An agent's documented full-Market assessment authorizes a transfer.

    This is an audit record, not an additional request for user permission.
    There is deliberately no automatic parameter sweep or accuracy threshold
    that pretends to replace the requested parameter/literature assessment.
    """
    cfg=json.loads((Path(root)/'configs'/f'{route}.json').read_text())
    if cfg['input_id']=='market-osnet':return
    if decision_path is None:raise ValueError('Transfer requires a recorded complete-Market decision')
    decision=json.loads(Path(decision_path).read_text())
    if (decision.get('verdict')!='continue_transfer' or decision.get('route')!=route or
        cfg['input_id'] not in decision.get('allowed_inputs',[])):
        raise ValueError('This transfer is not supported by the recorded decision')
    parent=Path(decision['market_run'])
    acceptance=accepted_result(parent)
    if acceptance is None:raise ValueError('Market must have complete population acceptance first')
    source=json.loads((parent/'run-manifest.json').read_text())['config']
    if source['input_id']!='market-osnet' or source['route']!=route:
        raise ValueError('Market assessment belongs to a different route or input')
    if full.sha(parent/'FULL_ACCEPTANCE.json')!=decision['market_acceptance_sha256']:
        raise ValueError('Market acceptance changed after the assessment')
    if not decision.get('rationale'):raise ValueError('Document the research assessment')


def launch(root,route,resume=False,decision_path=None):
    root=Path(root).resolve()
    if route not in protocol.ROUTES:raise ValueError('Unknown active route')
    with lock(root/'.execution.lock'):
        prepared=json.loads((root/'PREPARED.json').read_text())
        if prepared['commit']!=checkout():raise ValueError('Prepared checkout differs')
        for name,value in prepared['files_sha256'].items():
            if full.sha(root/name)!=value:raise ValueError('Prepared input/config changed: '+name)
        transfer_gate(root,route,decision_path)
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
    p.add_argument('--decision',type=Path)
    a=p.parse_args();launch(a.root,a.route,a.resume,a.decision)
