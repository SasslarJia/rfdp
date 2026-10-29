from pathlib import Path
import sys,importlib.util,types
HERE=Path(__file__).resolve().parent
ROOT=Path("/root/autodl-tmp/rfdp-graph-routes-20261026")
OUT=ROOT/"market-internal-nonlinear-theta0p5-v1"
QUERIES=[0,38,481,962,1443,1924,2405,2886,3367]
sys.path.insert(0,str(HERE.parent/"adaptive-precision-v1"))
import adaptive_adapter as adapter
def load_model(refs,theta):
    repo=Path(refs['repo']);source=repo/'RFDP/MODEL/Paper_RFDP'
    sys.path[:0]=[str(ROOT),str(repo/'RFDP/experiments/reliability')]
    import high_precision_recovery as recovery
    import accurate_hybrid_ties as original
    assert original.SOLVER_ID=='original-formula-certified-exact-ties-v3'
    name='internal_qualification_'+str(theta).replace('.','p')
    pkg=types.ModuleType(name);pkg.__path__=[str(source)];sys.modules[name]=pkg;mods=[]
    for leaf in ['PaperFullIterative','PaperPrecision']:
        spec=importlib.util.spec_from_file_location(name+'.'+leaf,source/(leaf+'.py'))
        mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
        setattr(pkg,leaf,mod);mods.append(mod)
    mod,precision=mods;precision.install()
    recovery.install(mod,events=precision.EVENTS,evidence_dir=OUT/'numerical-evidence')
    original.install(mod,precision.EVENTS);events=[];adapter.install(mod,events,theta)
    return mod,precision,events
