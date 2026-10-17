"""TEST ONLY: force one fallback on the first synthetic system, same equation.

This fixture is not a production CLI option. It refuses inputs outside this
local draft, non-96-node populations and a protocol without the test marker.
The source and the marked synthetic protocol are signed by the normal runner.
"""
import sys
from pathlib import Path
from scipy.io import loadmat
import numpy as np
import high_precision_recovery as recovery
import run_uniform_candidate


def main():
    def argument(name):return Path(sys.argv[sys.argv.index(name)+1]).resolve()
    draft=Path(__file__).resolve().parent.parent
    for name in ['--mat-path','--output-dir','--protocol','--repo']:
        if not argument(name).is_relative_to(draft):raise ValueError('Forced fallback test must stay inside local draft')
    if 'SYNTHETIC_FORCE_ONE_MP_FALLBACK' not in argument('--protocol').read_text():raise ValueError('Missing synthetic-only protocol marker')
    data=loadmat(argument('--mat-path'),variable_names=['query_f','gallery_f'])
    if len(data['query_f'])!=12 or len(data['gallery_f'])!=84:raise ValueError('Forced fallback test requires 12+84 artificial nodes')
    original_install=recovery.install
    def install_test(module,**kwargs):
        previous=module.solve_hybrid_sparse
        triggered=[False]
        def first_failure(*args,**options):
            if not triggered[0]:
                triggered[0]=True
                raise np.linalg.LinAlgError('SYNTHETIC TEST ONLY: force the same-equation MP fallback once')
            return previous(*args,**options)
        module.solve_hybrid_sparse=first_failure
        return original_install(module,**kwargs)
    recovery.install=install_test
    run_uniform_candidate.main()

if __name__=='__main__':main()
