"""Fixed early-context switch: exact Rank-1, then one complete official input."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import json
from pathlib import Path
import traceback

import numpy as np

from experiments.continuous_fitting import full
from experiments.context_graph import full_run as context
from experiments.reid_adaptation import full_run as shared
from experiments.reid_adaptation.io import lock, load_labels
from . import protocol, rank1


def prepare(recipe_path,node_path):
    return context.prepare(recipe_path,node_path,plan=protocol)


def decision(cfg,path):
    if not cfg['requires_decision']:return
    if path is None:raise ValueError('Transfer requires the completed Duke result and a decision')
    d=json.loads(Path(path).read_text())
    if (d.get('schema')!=protocol.SCHEMA or d.get('verdict')!='continue_fixed_transfer' or
        shared.unit_id(cfg) not in d.get('allowed_units',[]) or not d.get('rationale')):
        raise ValueError('Invalid early-context transfer decision')
    root=Path(d['duke_output']);accepted=shared.accepted_result(root)
    manifest=json.loads((root/'run-manifest.json').read_text())['config']
    if (not accepted or full.sha(root/'FULL_ACCEPTANCE.json')!=d['acceptance_sha256'] or
        manifest['adaptation_schema']!=protocol.SCHEMA or manifest['input_id']!='duke-osnet'):
        raise ValueError('Decision lacks the fixed candidate full Duke evidence')


def run(config_path,rank1_path,decision_path=None,resume=False):
    cfg=json.loads(Path(config_path).read_text());protocol.validate(cfg,require_bound=True)
    with lock(Path(cfg['workspace'])/'.execution.lock'):
        decision(cfg,decision_path);head,rows=rank1.check(cfg,rank1_path)
        try:
            result=shared._run(config_path,resume)
            labels=load_labels(cfg['labels_file'],cfg['queries'],cfg['gallery'])
            ranks=np.load(Path(cfg['output'])/'rankings.npy',mmap_mode='r')
            for q,row in enumerate(rows):
                assert rank1.first_valid(ranks[q],q,cfg['queries'],labels)==row['first_gallery']
            metrics=json.loads((Path(cfg['output'])/'metrics.json').read_text())
            assert abs(metrics['rank-1']-head['rank1'])<1e-14
            full.dump(Path(cfg['output'])/'RANK1_PARITY.json',dict(status='all_queries_identical',
                   queries=cfg['queries'],first_gallery_equal=cfg['queries'],rank1=head['rank1'],
                   rank1_acceptance_sha256=full.sha(Path(rank1_path)/'RANK1_ACCEPTANCE.json'),
                   rank1_output=str(Path(rank1_path).resolve()),time=full.now()))
            return result
        except BaseException:
            out=Path(cfg['output'])
            if out.exists():full.dump(out/'FAILED.json',dict(time=full.now(),error=traceback.format_exc()))
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['recipes','prepare','rank1','run','collect','archive'])
    for key in ('recipe','node','config','output','decision','rank1'):parser.add_argument('--'+key,type=Path)
    parser.add_argument('--resume',action='store_true');a=parser.parse_args()
    if a.action=='recipes':result=protocol.write_recipes(a.output)
    elif a.action=='prepare':result=prepare(a.recipe,a.node)
    elif a.action=='rank1':result=rank1.run(a.config,a.output)
    elif a.action=='run':result=run(a.config,a.rank1,a.decision,a.resume)
    elif a.action=='collect':result=context.collect(a.config,a.output)
    else:result=context.archive(a.config)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
