"""Read-only paired analysis of collected complete official-input evidence."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from . import protocol


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def query_table(metric,queries):
    rows=metric['per_query']
    if len(rows)!=queries or [r['query_id'] for r in rows]!=list(range(queries)):
        raise ValueError('Full official query order required')
    if metric['valid_queries']!=queries or not all(r['valid'] for r in rows):
        raise ValueError('Unexpected invalid/missing official queries')
    if any(r['positive_count']!=r['retrieved_positive_count'] for r in rows):
        raise ValueError('Incomplete gallery recall')
    for field,key in [('AP','mAP'),('rank1','rank-1'),('INP','mINP')]:
        x=np.asarray([r[field] for r in rows],dtype=float)
        if not np.isfinite(x).all() or np.any((x<0)|(x>1)) or abs(x.mean()-metric[key])>1e-12:
            raise ValueError('Per-query/aggregate metric mismatch: '+key)
    return rows


def paired(reference,candidate,queries):
    old=query_table(reference,queries);new=query_table(candidate,queries)
    for a,b in zip(old,new):
        if any(a[k]!=b[k] for k in ('query_id','pid','camid','positive_count')):
            raise ValueError('Paired input metadata differs')
    deltas={k:np.array([b[k]-a[k] for a,b in zip(old,new)]) for k in ('AP','rank1','INP')}
    counts={k:dict(improved=int(np.sum(d>1e-12)),worsened=int(np.sum(d < -1e-12)),unchanged=int(np.sum(abs(d)<=1e-12))) for k,d in deltas.items()}
    metrics={key:dict(reference_pct=100*reference[key],candidate_pct=100*candidate[key],
        delta_pp=100*(candidate[key]-reference[key]),
        relative_change_pct=100*(candidate[key]/reference[key]-1) if reference[key] else None)
        for key in ('mAP','rank-1','mINP')}
    def case(i):
        return dict(query_id=i,pid=old[i]['pid'],camera=old[i]['camid'],positive_count=old[i]['positive_count'],
                    reference={k:old[i][k] for k in ('AP','rank1','INP')},
                    candidate={k:new[i][k] for k in ('AP','rank1','INP')},delta_AP_pp=float(100*deltas['AP'][i]))
    ids=np.arange(queries);order=np.lexsort((ids,deltas['AP']))
    return dict(metrics=metrics,paired_query_changes=counts,
        rank1_improved_query_ids=np.flatnonzero(deltas['rank1']>0).tolist(),
        rank1_worsened_query_ids=np.flatnonzero(deltas['rank1']<0).tolist(),
        AP_largest_losses=[case(int(i)) for i in order if deltas['AP'][i]<-1e-12][:10],
        AP_largest_gains=[case(int(i)) for i in order[::-1] if deltas['AP'][i]>1e-12][:10],
        query_change_tolerance=1e-12)


def analyze(evidence,reference,index):
    evidence=Path(evidence);bindings=json.loads((evidence/'SHA256.json').read_text())
    for name,h in bindings.items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or sha(evidence/p)!=h:raise ValueError('Collected evidence SHA differs')
    required={'metrics.json','run-manifest.json','COMPUTE_COMPLETE.json','FULL_ACCEPTANCE.json','config.json','query-runtime.json'}
    if not required.issubset(bindings):raise ValueError('Missing complete evidence files')
    cfg=json.loads((evidence/'config.json').read_text());protocol.validate(cfg,require_bound=True)
    accepted=json.loads((evidence/'FULL_ACCEPTANCE.json').read_text())
    manifest=json.loads((evidence/'run-manifest.json').read_text());computed=json.loads((evidence/'COMPUTE_COMPLETE.json').read_text())
    if (accepted['status']!='accepted_full_population' or manifest['config']!=cfg or
        accepted['signature']!=protocol.digest(cfg) or accepted['signature']!=manifest['signature'] or
        computed['signature']!=accepted['signature'] or accepted['queries']!=cfg['queries'] or accepted['gallery']!=cfg['gallery']):
        raise ValueError('Acceptance/configuration population differs')
    if accepted['metrics_sha256']!=sha(evidence/'metrics.json') or computed['metrics_sha256']!=accepted['metrics_sha256']:
        raise ValueError('Computed metrics differ')
    if (accepted['rankings_sha256']!=computed['rankings_sha256'] or
        accepted['max_recomputed_residual']>1e-8 or
        not accepted['actual_saved_scores_recomputed'] or not accepted['all_frontiers_bridges_rankings_rebuilt']):
        raise ValueError('Complete ranking/certificate binding differs')
    refs=json.loads(Path(index).read_text());ref=next(x for x in refs if x['input_id']==cfg['input_id'])
    if sha(reference)!=ref['sha256'] or (ref['queries'],ref['gallery'])!=(cfg['queries'],cfg['gallery']):
        raise ValueError('Reference input/metrics differ')
    report=paired(json.loads(Path(reference).read_text()),json.loads((evidence/'metrics.json').read_text()),cfg['queries'])
    runtime=json.loads((evidence/'query-runtime.json').read_text())
    if len(runtime)!=cfg['queries'] or [r['query'] for r in runtime]!=list(range(cfg['queries'])):
        raise ValueError('Runtime population differs')
    frontiers=[r['diagnostics']['max_unlabeled'] for r in runtime]
    elapsed=(datetime.datetime.fromisoformat(accepted['time'])-datetime.datetime.fromisoformat(manifest['time'])).total_seconds()
    report.update(input_id=cfg['input_id'],route=cfg['adaptation_route'],queries=cfg['queries'],gallery=cfg['gallery'],
        parameters={k:cfg[k] for k in ('kd','ks','gamma','alpha','beta','step','depth')},
        config_sha256=sha(evidence/'config.json'),reference_sha256=sha(reference),
        complete_acceptance_sha256=sha(evidence/'FULL_ACCEPTANCE.json'),
        timing=dict(accepted['timing'],run_elapsed_seconds=elapsed,workers=cfg['workers'],
                    compute_pool_wall_seconds=computed['latest_attempt_seconds'],
                    per_query_mean_seconds={k:float(np.mean([r[k] for r in runtime])) for k in ('compute_seconds','saving_seconds','readback_seconds')},
                    baseline_speed_ratio=None,baseline_speed_ratio_reason='No same-machine baseline timing in this run; old accuracy is reused'),
        solver_calls=sum(r['diagnostics']['solver_calls'] for r in runtime),
        frontier_quantiles=dict(zip(('min','median','p90','max'),np.quantile(frontiers,[0,.5,.9,1]).tolist())),
        analysis_source_sha256=sha(__file__),
        scope='Complete accepted official input; descriptive paired differences, not a causal decomposition or significance test')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('evidence','reference','reference-index','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();report=analyze(a.evidence,a.reference,a.reference_index)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(input_id=report['input_id'],metrics=report['metrics'],paired_query_changes=report['paired_query_changes']),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
