"""Describe frozen graph coverage after a completed run; never solve or rank."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import loadmat


def analyze(root):
    root=Path(root)
    cfg=json.loads((root/'configs/csw_grouped.json').read_text())
    acceptance=json.loads((root/'runs/csw_grouped/FULL_ACCEPTANCE.json').read_text())
    assert acceptance['status']=='accepted_full_population'
    assert (acceptance['queries'],acceptance['gallery'])==(cfg['queries'],cfg['gallery'])
    original=sparse.load_npz(Path(cfg['cache'])/'graph.npz')
    grouped=sparse.load_npz(cfg['grouped_graph_file'])
    cameras=np.load(cfg['camera_ids_file'],allow_pickle=False)
    labels=loadmat(cfg['mat'],variable_names=['query_label','gallery_label'])
    pids=np.r_[labels['query_label'].ravel(),labels['gallery_label'].ravel()]
    known=np.isin(pids,np.unique(labels['query_label']))&(pids>0)
    n=len(pids)
    assert original.shape==grouped.shape==(n,n) and len(cameras)==n

    def codes(graph):
        i,j=graph.nonzero();keep=i<j
        return i[keep].astype(np.int64)*n+j[keep]

    def stats(edges):
        i,j=edges//n,edges%n;cross=cameras[i]!=cameras[j]
        identified=known[i]&known[j];same=pids[i]==pids[j]
        return dict(edges=int(len(edges)),cross_camera_edges=int(cross.sum()),
            cross_camera_fraction=float(cross.mean()) if len(edges) else 0.,
            known_identity_edges=int(identified.sum()),
            same_identity_fraction_known=float(same[identified].mean()) if identified.any() else None,
            known_cross_edges=int((identified&cross).sum()),
            same_identity_fraction_known_cross=float(same[identified&cross].mean()) if (identified&cross).any() else None)

    a,b=codes(original),codes(grouped)
    result=dict(scope='Existing frozen input graphs only; labels used solely after ranking; topology description, not causal attribution',
        population=n,original=stats(a),grouped=stats(b),added=stats(np.setdiff1d(b,a)),
        removed=stats(np.setdiff1d(a,b)),retained=stats(np.intersect1d(a,b)),
        mean_nonself_degree_original=2*len(a)/n,mean_nonself_degree_grouped=2*len(b)/n)
    path=root/'runs/csw_grouped/graph-mechanism.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    analyze(parser.parse_args().root)
