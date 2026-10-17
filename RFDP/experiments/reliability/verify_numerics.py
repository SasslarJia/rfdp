"""Audit every persisted arbitrary-precision fallback after complete ranking."""
import argparse
import hashlib
import json
from pathlib import Path
from fractions import Fraction
import numpy as np
from high_precision_recovery import verify_evidence


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--method',required=True)
    args=parser.parse_args();root=args.output_dir.resolve()
    standard=json.loads((root/f'verified-{args.method}.json').read_text())
    assert standard['status']=='verified'
    count=standard['checks']['diffusion_checkpoints'];events=0;files={};maximum=0.
    for q in range(count):
        record_path=root/args.method/f'{q:04d}.json'
        assert sha(record_path)==standard['files'][str(record_path.resolve())]['sha256'],'Checkpoint changed since primary verification'
        record=json.loads(record_path.read_text())
        for event in record['refinement_events']:
            if event.get('kind')!='same-float64-equation-high-precision-recovery':continue
            events+=1;evidence=event['evidence']
            assert Path(evidence['metadata']).resolve()==Path(evidence['npz']).with_suffix('.json').resolve(),'Mismatched evidence pair'
            for field in ['npz','metadata']:
                path=Path(evidence[field]).resolve()
                assert path.is_relative_to(root/'numerical-evidence'),'Evidence escaped run directory'
                assert sha(path)==evidence[field+'_sha256'],'Saved numerical evidence changed'
                files[str(path)]=evidence[field+'_sha256']
            certificate=verify_evidence(evidence['npz'])
            assert event['query']==q,'Numerical event query mismatch'
            step=next(t for t in record['trace']if t['iteration']==event['iteration'])
            ids=event['unlabeled']
            assert len(ids)==step['size'] and sorted(set(ids))==ids
            assert all(0<=i<standard['metrics']['query_count']+standard['metrics']['gallery_count']for i in ids)
            assert event['selected']==step['selected'] and 0<len(event['selected'])<=5
            with np.load(evidence['npz'],allow_pickle=False)as saved:
                scores=[Fraction(int(n),int(d))for n,d in zip(saved['score_numerator'],saved['score_denominator'])]
            assert len(scores)==len(ids)
            order=sorted(range(len(ids)),key=lambda j:(scores[j],ids[j]))
            assert [ids[j]for j in order[:len(event['selected'])]]==event['selected'],'Saved scores do not reproduce selected nodes'
            assert certificate['relative_residual']==step['residual'],'Numerical event residual differs from actual iteration'
            assert certificate==event['certificate']
            assert certificate['relative_residual']<=1e-8
            assert certificate['relative_residual']<=record['diagnostics']['max_relative_residual']+1e-16
            maximum=max(maximum,certificate['relative_residual'])
    existing={str(p.resolve())for p in (root/'numerical-evidence').glob('*')if p.is_file()}
    assert set(files)<=existing,'Missing numerical evidence'
    unreferenced=sorted(existing-set(files))
    for name in unreferenced:
        if name.endswith('.npz'):
            verify_evidence(name)
            assert str(Path(name).with_suffix('.json')) in existing
        elif name.endswith('.json'):
            assert str(Path(name).with_suffix('.npz')) in existing
        else:raise ValueError('Unknown numerical evidence file')
    report=dict(status='verified',method=args.method,query_checkpoints=count,arbitrary_precision_events=events,max_relative_residual=maximum,files=files,unreferenced_prior_attempt_files={name:sha(name)for name in unreferenced},standard_report_sha256=sha(root/f'verified-{args.method}.json'),verifier_sha256=sha(__file__),recovery_sha256=sha(Path(__file__).with_name('high_precision_recovery.py')))
    out=root/'verified-numerical-evidence.json'
    if out.exists():assert json.loads(out.read_text())==report,'Numerical audit changed'
    else:out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='verified',events=events,max_relative_residual=maximum)),flush=True)


if __name__=='__main__':main()
