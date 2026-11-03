"""Analyze accepted full rankings and bounded saved SC frontiers.

No retrieval trajectory is rerun. Optional original-objective solves on saved
SC frontiers are explicitly counterfactual diagnostics, not new test metrics.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import gzip
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.io import loadmat

from ..continuous_fitting import certified
from ..full_reid.common import dump, sha
from .csw_graph import load_inputs, ReadbackKernel
from .score_components import components


def records(path):
    d=json.loads(Path(path).read_text())
    return {int(r['query_id']):r for r in d.get('per_query',d.get('queries'))}


def positive_positions(rank,q,qn,qp,qc,gp,gc):
    local=np.asarray(rank,dtype=int)-qn
    assert len(local)==len(gp) and np.array_equal(np.sort(local),np.arange(len(gp)))
    keep=(gp[local]!=-1)&~((gp[local]==qp[q])&(gc[local]==qc[q]))
    filtered=local[keep];hit=gp[filtered]==qp[q]
    positions=np.flatnonzero(hit)+1
    expected=np.flatnonzero((gp==qp[q])&(gc!=qc[q])&(gp!=-1))
    assert len(positions)==len(expected)>0
    index=np.argsort(filtered[hit])
    return positions,positions[index],float(np.mean(np.arange(1,len(positions)+1)/positions)),int(hit[0])


def analyze_cell(cell,out):
    cfg=json.loads(Path(cell['config']).read_text());qn,gn=cfg['queries'],cfg['gallery']
    assert (qn,gn) in ((3368,19732),(2228,17661))
    data=loadmat(cfg['mat'],variable_names=['query_label','query_cam','gallery_label','gallery_cam'])
    qp,qc,gp,gc=[np.asarray(data[k]).ravel() for k in ['query_label','query_cam','gallery_label','gallery_cam']]
    assert (len(qp),len(gp))==(qn,gn)
    arrays={};metrics={};position={};audits={}
    for method,entry in cell['methods'].items():
        compact=None
        if 'positive_positions' in entry:
            path=Path(entry['positive_positions'])
            with gzip.open(path,'rt') as f:compact=json.load(f)
            assert compact['shape']==[qn,gn] and compact['all_full_permutations_checked']
            assert compact['feature_sha256']==cfg['feature_sha256']
            assert len(compact['queries'])==qn
            a=None
        else:
            path=Path(entry['rankings']);a=np.load(path,mmap_mode='r');assert a.shape==(qn,gn)
        rows=records(entry['metrics']);assert set(rows)==set(range(qn))
        current=[]
        for q in range(qn):
            if compact is None:
                pos,by_id,ap,r1=positive_positions(a[q],q,qn,qp,qc,gp,gc)
            else:
                item=compact['queries'][q];assert item['query']==q
                pos=np.asarray(item['positions']);by_id=np.asarray(item['positions_by_gallery_id'])
                assert len(pos)==np.sum((gp==qp[q])&(gc!=qc[q])&(gp!=-1))
                assert np.array_equal(np.sort(by_id),pos)
                ap=float(np.mean(np.arange(1,len(pos)+1)/pos));r1=int(pos[0]==1)
            row=rows[q];assert row['pid']==int(qp[q]) and row['camid']==int(qc[q])
            assert abs(ap-row['AP'])<1e-12 and r1==row['rank1']
            current.append((pos,by_id))
        arrays[method]=a;metrics[method]=rows;position[method]=current
        audits[method]=dict(shape=[qn,gn],rankings_sha256=sha(path) if compact is None else compact['rankings_sha256'],
                            compact_positions_sha256=None if compact is None else sha(path),metrics_sha256=sha(entry['metrics']),
                            all_queries_recomputed=True,mAP=float(np.mean([r['AP'] for r in rows.values()])),
                            rank1=float(np.mean([r['rank1'] for r in rows.values()])))
    contrasts={}
    for left,right in [('original','sc'),('csw','combined'),('original','csw'),('sc','combined')]:
        details=[];groups={k:dict(queries=0,contribution_mAP_pp=0.) for k in ['both_correct','both_wrong','gained','lost']}
        for q in range(qn):
            a,b=metrics[left][q],metrics[right][q];pa,ia=position[left][q];pb,ib=position[right][q]
            group='gained' if b['rank1']>a['rank1'] else 'lost' if b['rank1']<a['rank1'] else 'both_correct' if a['rank1'] else 'both_wrong'
            delta=b['AP']-a['AP'];groups[group]['queries']+=1;groups[group]['contribution_mAP_pp']+=100*delta/qn
            details.append(dict(query=q,pid=int(qp[q]),group=group,AP_delta_pp=100*delta,
                positive_count=len(pa),positive_images_forward=int(np.sum(ib<ia)),
                positive_images_backward=int(np.sum(ib>ia)),positive_images_same=int(np.sum(ib==ia)),
                first_positive_before=int(pa[0]),first_positive_after=int(pb[0]),
                last_positive_before=int(pa[-1]),last_positive_after=int(pb[-1]),
                AP_contribution_first_positive_pp=float((1/pb[0]-1/pa[0])*100/len(pa)),
                AP_contribution_other_positives_pp=float((delta-(1/pb[0]-1/pa[0])/len(pa))*100)))
        name=right+'-minus-'+left;dump(out/(name+'-queries.json'),details)
        ordered=sorted(details,key=lambda r:(r['AP_delta_pp'],r['query']))
        contrasts[name]=dict(mAP_delta_pp=float(np.mean([r['AP_delta_pp'] for r in details])),groups=groups,
            first_positive_contribution_pp=float(np.mean([r['AP_contribution_first_positive_pp'] for r in details])),
            other_positive_contribution_pp=float(np.mean([r['AP_contribution_other_positives_pp'] for r in details])),
            changed_queries=sum(abs(r['AP_delta_pp'])>1e-10 for r in details),
            diagnostic_examples=ordered[:5]+ordered[-5:])
    return cfg,(qp,qc,gp,gc),dict(cell=cell['name'],official_population=True,audits=audits,contrasts=contrasts)


def frontier_diagnostics(cell,cfg,labels,out,max_stages):
    qp,qc,gp,gc=labels;qn=cfg['queries'];pids=np.r_[qp,gp];cameras=np.r_[qc,gc]
    cache=Path(cfg['cache']);base=sparse.load_npz(cache/'graph.npz').astype(float).tocsr()
    full=np.load(cache/'affinity.npy',mmap_mode='r');result={}
    selected=np.linspace(0,qn-1,32,dtype=int).tolist()
    for relation,method in [('original','sc'),('csw','combined')]:
        if relation=='csw':
            cams,support,_=load_inputs(cfg,full)
            graph=ReadbackKernel(base,full,cams,support,cfg['rho'],cfg['support_lambda']).graph
        else:graph=base
        rows=[];run=Path(cell['methods'][method]['run'])
        for q in selected:
            path=run/'queries'/f'{q:04d}.trace.jsonl.gz'
            if not path.exists():
                rows.append(dict(query=q,status='saved_trace_unavailable'));continue
            boundary=np.asarray(graph[[q]].sum(axis=0)).ravel();found=False;visited=0;last=None
            with gzip.open(path,'rt') as stream:
                for line in stream:
                    t=json.loads(line)
                    if t['iteration']>=max_stages:break
                    visited+=1;ids=np.asarray(t['unlabeled'],dtype=int);e=boundary[ids].copy()
                    for i,j,v in t['bridges']:
                        where=np.flatnonzero(ids==i)
                        if len(where):e[where[0]]+=v
                    if t['returned_scores'] is not None:
                        w=graph[ids][:,ids]
                        old,_,event=certified.solve(w,e,alpha=5.,beta=25.,source='original')
                        old_ids=ids[event['returned_prefix_local']][:len(t['selected'])].tolist()
                        if old_ids!=t['selected']:
                            c=components(w,e,5.,25.)
                            actual=np.array([float(v) for v in certified.unpack(t['returned_scores'])])
                            error=float(np.max(np.abs(c['score']-actual))/max(1.,np.max(np.abs(actual))))
                            assert error<1e-8
                            def kind(i):
                                return 'query_node' if i<qn else 'filtered' if pids[i]==-1 or (pids[i]==pids[q] and cameras[i]==cameras[q]) else 'positive' if pids[i]==pids[q] else 'negative'
                            details=[]
                            for i in sorted(set(old_ids+t['selected'])):
                                j=int(np.flatnonzero(ids==i)[0]);details.append(dict(node=i,kind=kind(i),degree=float(c['h'][j]),
                                    boundary=float(c['u'][j]),indirect=float(25*c['v'][j]),score=float(c['score'][j]),
                                    original_score=float(old[j]),selected_original=i in old_ids,selected_sc=i in t['selected']))
                            last=dict(query=q,status='first_counterfactual_difference',iteration=t['iteration'],
                                frontier_size=len(ids),original_selected=old_ids,sc_selected=t['selected'],nodes=details,
                                decomposition_error=error,boundary_mass_error=float(np.max(abs(c['mass']-1))),
                                shared_before_difference=True)
                            found=True;break
                    for node in t['selected']:
                        a,b=graph.indptr[node:node+2];boundary[graph.indices[a:b]]+=graph.data[a:b]
            rows.append(last if found else dict(query=q,status='no_difference_in_checked_window',checked_stages=visited))
        dump(out/(relation+'-frontiers.json'),rows)
        result[relation]=dict(queries=len(rows),max_checked_stages=max_stages,
            differences=sum(r['status']=='first_counterfactual_difference' for r in rows),
            missing_traces=sum(r['status']=='saved_trace_unavailable' for r in rows),
            scope='Original objective solved on saved SC frontiers until the first decision difference, within the declared window. No new full-ranking metric; later divergence is not measured.')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--frontier-stages',type=int,default=20)
    args=p.parse_args();config=json.loads(args.config.read_text());args.output.mkdir(exist_ok=False)
    start=time.monotonic();summaries=[]
    for cell in config['cells']:
        out=args.output/cell['name'];out.mkdir()
        cfg,labels,summary=analyze_cell(cell,out);dump(out/'rank-summary.json',summary)
        print(json.dumps(dict(cell=cell['name'],stage='rankings_complete',contrasts={k:v['mAP_delta_pp'] for k,v in summary['contrasts'].items()})),flush=True)
        summary['frontiers']=frontier_diagnostics(cell,cfg,labels,out,args.frontier_stages)
        dump(out/'summary.json',summary);summaries.append(summary)
        dump(args.output/'progress.json',dict(completed=len(summaries),total=len(config['cells']),seconds=time.monotonic()-start))
    dump(args.output/'summary.json',dict(status='complete',cells=summaries,config_sha256=sha(args.config),
        code_sha256=sha(__file__),seconds=time.monotonic()-start,reranked_queries=0))


if __name__=='__main__':main()
