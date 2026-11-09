"""Publish only accepted complete populations; reuse the two frozen references."""
import json
import math
from pathlib import Path
from . import protocol
from experiments.continuous_fitting import full

NAMES={'camera_channel':'内部相机通道＋I-HyRDP',
       'csw_grouped':'分组CSW＋I-HyRDP','th_target':'TH目标条件三元＋I-HyRDP'}


def load_references(path):
    data=json.loads(Path(path).read_text())
    rows=data['records']
    if len(rows)!=2 or {r['method'] for r in rows}!=set(protocol.REFERENCE_METHODS):
        raise ValueError('Exactly the existing I-HyRDP and CSW reference records are required')
    for r in rows:
        if (r['input_key'],r['queries'],r['gallery'],r['feature_sha256']) != (
                'market-osnet',3368,19732,protocol.FROZEN['feature_sha256']):
            raise ValueError('Reference population/input differs')
        for key in ('mAP','rank1','mINP'):
            if not math.isfinite(r[key]) or not 0<=r[key]<=1:raise ValueError('Reference metric must be a fraction')
        if not r['provenance']['verification'].get('official_population_verified'):
            raise ValueError('Reference official population is not verified')
    return data


def collect(root,references_path):
    from .full_run import accepted_result
    root=Path(root);references=load_references(references_path)
    records=[dict(r,status='reused_existing_full_result') for r in references['records']]
    by_name={r['method']:r for r in records}
    pending=[]
    for route in protocol.ROUTES:
        out=root/'runs'/route
        accepted=accepted_result(out)
        if accepted is None:
            failure=out/'FAILED.json'
            pending.append(dict(route=route,status='failed' if failure.exists() else 'not_fully_accepted'))
            continue
        metrics=json.loads((out/'metrics.json').read_text())
        queries=metrics['per_query']
        if len(queries)!=3368 or sorted(r['query_id'] for r in queries)!=list(range(3368)):
            raise ValueError('Metrics do not cover the official complete query set')
        for q in queries:
            if q['retrieved_positive_count']!=q['positive_count']:
                raise ValueError('Incomplete gallery positive coverage')
        cfg=json.loads((out/'run-manifest.json').read_text())['config']
        parent=by_name['CSW+I-HyRDP' if route=='csw_grouped' else 'I-HyRDP']
        runtimes=[json.loads((out/'queries'/f'{q:04d}.json').read_text()) for q in range(3368)]
        row=dict(method=NAMES[route],route=route,status='accepted_full_population',input_key='market-osnet',
            queries=3368,gallery=19732,mAP=metrics['mAP'],rank1=metrics['rank-1'],mINP=metrics['mINP'],
            reference_method=parent['method'],parameters=protocol.validate(cfg,require_bound=True),
            mean_query_compute_seconds=sum(r['compute_seconds'] for r in runtimes)/3368,
            mean_query_readback_seconds=sum(r['readback_seconds'] for r in runtimes)/3368,
            acceptance_sha256=full.sha(out/'FULL_ACCEPTANCE.json'),metrics_sha256=full.sha(out/'metrics.json'),
            commit=cfg['commit'],feature_sha256=cfg['feature_sha256'],
            delta_percentage_points={k:100*(v-parent[k]) for k,v in
                [('mAP',metrics['mAP']),('rank1',metrics['rank-1']),('mINP',metrics['mINP'])]})
        records.append(row)
    result=dict(scope='official full Market-OSNet only; no partial-query accuracy',metric_scale='fraction_0_to_1',
                references_source=references.get('source_registry'),records=records,pending=pending)
    full.dump(root/'results.json',result)
    text=['# Market–OSNet：三条路线完整结果','',
          '原 I-HyRDP 与原 CSW 直接复用既有官方完整结果。新增路线只在 FULL_ACCEPTANCE 通过后进入指标表。',
          '', '| 方法 | 状态 | mAP (%) | Rank-1 (%) | mINP (%) |', '|---|---|---:|---:|---:|']
    for row in records:
        status='复用旧全量结果' if row['status'].startswith('reused') else '新全量已验收'
        text.append(f"| {row['method']} | {status} | {100*row['mAP']:.4f} | {100*row['rank1']:.4f} | {100*row['mINP']:.4f} |")
    for row in pending:
        status='技术失败，尚无全量结果' if row['status']=='failed' else '尚无全量验收结果'
        text.append(f"| {NAMES[row['route']]} | {status} | — | — | — |")
    text += ['', 'α=5、β=25、step=5、depth=500；全部3368查询与19732图库。三条路线独立、配置固定。',
             '分组CSW直接对照原CSW；两条内部平滑直接对照原I-HyRDP。mAP/Rank-1/mINP的差值以百分点计，详见results.json。',
             '完整排名、大轨迹和逐查询证据留在各run目录；本表不把局部查询分数或原论文分数作为实测结果。','']
    (root/'结果汇总.md').write_text('\n'.join(text))
    return result
