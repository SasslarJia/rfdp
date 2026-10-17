"""Bind a new control to exactly the input of its frozen v2 paired test report."""
import hashlib
import json
from pathlib import Path


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_input_binding(path,digest,*,model,mode,source_lambda,rho,selection_sha256,
                           mat,cache,support,query_count,gallery_count,depth):
    path=Path(path).resolve();cache=Path(cache).resolve()
    require(sha(path)==digest,'Frozen input binding SHA mismatch')
    binding=json.loads(path.read_text())
    require(binding['schema']=='uniform-frozen-input-binding-v1','Unknown input binding schema')
    require((binding['model'],binding['source_support_mode'],binding['source_lambda'],binding['rho'],binding['selection_sha256'])==
            (model,mode,source_lambda,rho,selection_sha256),'Input binding model/mode/parameter/selection differs')
    require(binding['feature_sha256']==sha(mat),'Paired model feature SHA mismatch')
    require(binding['support_sha256']==sha(support),'Paired source support archive SHA mismatch')
    names=['manifest.json','features.npy','affinity.npy','graph.npz']
    require(set(binding['cache_files'])==set(names),'Incomplete input binding cache inventory')
    for name in names:require(binding['cache_files'][name]==sha(cache/name),'Paired cache SHA mismatch: '+name)
    extra={str(path):digest}
    if binding['scope']=='synthetic-only':
        draft=Path(__file__).resolve().parent.parent
        require((query_count,gallery_count)==(12,84),'Synthetic binding cannot authorize formal population')
        require(all(Path(p).resolve().is_relative_to(draft)for p in [path,mat,cache,support]),'Synthetic binding must stay within draft')
        require(binding.get('description')=='artificial software fixture; no real ReID results','Synthetic input must be explicitly identified')
    else:
        require(binding['scope']=='frozen-v2-test','Unknown binding scope')
        require((query_count,gallery_count,depth)==(3368,19732,500),'Formal control requires full test population/depth')
        report_path=Path(binding['paired_standard_report_path']).resolve()
        manifest_path=Path(binding['paired_run_manifest_path']).resolve()
        for p,key in [(report_path,'paired_standard_report_sha256'),(manifest_path,'paired_run_manifest_sha256')]:
            require(sha(p)==binding[key],'Paired frozen evidence SHA mismatch');extra[str(p)]=binding[key]
        report=json.loads(report_path.read_text());manifest=json.loads(manifest_path.read_text());cfg=manifest['config']
        require(report_path.parents[1].name==model and report_path.parents[2].name=='test','Paired test report has wrong model path')
        require(report['status']=='verified','Paired report is not verified')
        require(Path(report['manifest']['path']).resolve()==manifest_path,'Paired manifest path differs')
        require(report['files'][str(manifest_path)]['sha256']==binding['paired_run_manifest_sha256'],'Paired report does not bind manifest')
        require(manifest['signature']==canonical(cfg)==report['manifest']['signature'],'Paired signature differs')
        require((cfg['rho'],cfg['lambda'],cfg['support_mode'])==(rho,source_lambda,mode),'Paired source configuration differs')
        require(all(cfg.get(key)==value for key,value in {'alpha':5,'beta':25,'step':5,'k':20,'gamma':.4,'depth':500,'numerical_solver':'same-float64-exact-rational-mp-v1'}.items()),'Paired production parameters/solver differ')
        require(cfg.get('numerical_tail',False) is False,'Paired numerical tail enabled')
        token=str(float(source_lambda)).replace('.','p')
        require(cfg['method']==report['method']==mode+'-r0p5-l'+token,'Paired source method differs')
        require((cfg['query_count'],cfg['gallery_count'],cfg['depth'])==(query_count,gallery_count,depth),'Paired coverage differs')
        require(cfg['feature_sha256']==report['input_mat_sha256']==binding['feature_sha256'],'Paired report feature binding differs')
        require(cfg['support_sha256']==binding['support_sha256'],'Paired report support binding differs')
        original_cache=Path(cfg['cache_manifest_path']).parent
        for name in names:
            require(cfg['inputs'][str(original_cache/name)]==binding['cache_files'][name],'Paired report cache binding differs: '+name)
        checks=report['checks']
        require(checks['no_diffusion'] is False,'Paired result did not use diffusion')
        require(checks['full_permutation_queries']==checks['diffusion_checkpoints']==query_count and checks['diffusion_depth']==depth,'Paired standard coverage incomplete')
        require(checks['required_max_relative_residual']==1e-8 and checks['max_relative_residual']<=1e-8,'Paired numerical acceptance differs')
    return binding,extra
