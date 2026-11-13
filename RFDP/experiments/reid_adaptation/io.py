"""Metadata conversion and fixed-checkout helpers; no identity enters scoring."""
import contextlib
import fcntl
import json
import subprocess
from pathlib import Path
import numpy as np
from experiments.full_reid.common import sha, dump


def checkout(repo):
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True)
    if dirty.strip():raise ValueError('Use a clean execution checkout')
    return head


@contextlib.contextmanager
def lock(path):
    with Path(path).open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def labels_from_manifest(path, queries, gallery):
    rows=json.loads(Path(path).read_text())
    if len(rows)!=queries+gallery:raise ValueError('Official metadata population differs')
    if any(r['split']!=('query' if i<queries else 'gallery') for i,r in enumerate(rows)):
        raise ValueError('Metadata order must be queries then gallery')
    keys=[(r['split'],Path(r['path']).name) for r in rows]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate image metadata')
    for row in rows:
        if type(row['pid']) is not int or type(row['camid']) is not int:
            raise ValueError('Integer identity/camera metadata required')
    pid=np.asarray([r['pid'] for r in rows],dtype=np.int64)
    cam=np.asarray([r['camid'] for r in rows],dtype=np.int64)
    return dict(query_label=pid[:queries],query_cam=cam[:queries],
                gallery_label=pid[queries:],gallery_cam=cam[queries:])


def load_labels(path, queries, gallery):
    with np.load(path,allow_pickle=False) as z:
        data={k:z[k] for k in ('query_label','query_cam','gallery_label','gallery_cam')}
    for key,a in data.items():
        if a.shape!=(queries if key.startswith('query') else gallery,) or a.dtype.kind not in 'iu':
            raise ValueError('Saved evaluation metadata differs: '+key)
    return data
