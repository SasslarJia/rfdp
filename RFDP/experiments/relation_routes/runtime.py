"""Bind the already tested equations to full-population producer/readback."""
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from . import protocol, smoothing
from experiments.continuous_fitting import certified


def initialize(cfg, graph, affinity, compute=True):
    from MODEL.Paper_RFDP.PaperFullIterative import FullIHyRDP
    protocol.validate(cfg,require_bound=True)
    cameras=np.load(cfg['camera_ids_file'],allow_pickle=False)
    if cameras.shape!=(graph.shape[0],) or cameras.dtype.kind not in 'iu':
        raise ValueError('Camera order/shape differs from the frozen graph')
    kernel=certifier=engine=None
    if cfg['route']=='csw_grouped':
        from .csw_grouped import validate_grouped_inputs
        from experiments.reliability.reliability_model import ReliabilityIHyRDP
        from experiments.stationary_measure.csw_graph import ReadbackKernel
        raw=sparse.load_npz(cfg['grouped_graph_file']).astype(np.float64).tocsr()
        support=sparse.load_npz(cfg['support_file']).astype(np.float64).tocsr()
        metadata=json.loads(Path(cfg['grouped_metadata_file']).read_text())
        validate_grouped_inputs(raw,support,metadata,cameras=cameras,k_intra=5,k_inter=20)
        reader=ReadbackKernel(raw,affinity,cameras,support,cfg['rho'],cfg['support_lambda'])
        graph,sums,kernel=reader.graph,reader.sums,reader.block
        if compute:
            certified.install('original')
            engine=ReliabilityIHyRDP(raw,affinity,cameras,eta=cfg['rho'],support=support,
                lam=cfg['support_lambda'],alpha=cfg['alpha'],beta=cfg['beta'],step=5)
            assert (engine.w-graph).nnz==0 and np.array_equal(engine.full_sums,sums)
    else:
        sums=np.asarray(affinity.sum(axis=1,dtype=np.float64))
        def context(indices,targets):
            return dict(source=cfg['source'],cameras=cameras[indices],
                        target_count=len(targets),theta=cfg.get('theta',.5))
        def solve(w,e,alpha,beta,*,diagnostics,indices,targets,bridges):
            values,residual,event=smoothing.solve(w,e,alpha,beta,**context(indices,targets))
            event['cg_calls']=int(event['candidate_method']=='cg')
            event['direct_calls']=int(event['candidate_method']=='sparse-rank-one-lu')
            certified.EVENTS.append(event)
            diagnostics.update(cg_calls=int(event['candidate_method']=='cg'),
                cg_iterations=event['cg_iterations'],
                direct_calls=int(event['candidate_method']=='sparse-rank-one-lu'),uncertified_cg=0)
            return values,residual
        def verify(w,e,values,event,*,alpha,beta,source,indices,targets,bridges):
            assert source==cfg['source']
            result=smoothing.certify_saved(w,e,values,alpha,beta,**context(indices,targets))
            assert result['input_sha256']==event['input_sha256']
            return result
        certifier=verify
        if compute:engine=FullIHyRDP(graph,affinity,alpha=cfg['alpha'],beta=cfg['beta'],
                                   step=5,stage_solver=solve)
    return engine,graph,kernel,sums,certifier
