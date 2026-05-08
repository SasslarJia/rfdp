#
import os
import sys
import numpy as np
#
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MODEL.Asst_RFDP.OperAsstNumpy import OperAsstNumpy
from MODEL.Iput_RFDP.IputWhl import IputWhl
from MODEL.Paper_RFDP.PaperRanker import HyRdpPaper
from MODEL.Paper_RFDP.PaperRanker import GmfptPaper
from MODEL.Paper_RFDP.PaperIterative import IHyRdpPaper
from MODEL.Paper_RFDP.PaperIterative import IGmfptPaper
from MODEL.Paper_RFDP.PaperSolvers import HyRdpClosedForm


def build_demo_graph():
    """Two clusters with one relevant object hidden in another component."""
    base_w = np.array([
        [1.0, 0.9, 0.0, 0.0, 0.0],
        [0.9, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.85, 0.2],
        [0.0, 0.0, 0.85, 1.0, 0.3],
        [0.0, 0.0, 0.2, 0.3, 1.0]
    ])
    full_w = np.array([
        [1.0, 0.9, 0.10, 0.75, 0.20],
        [0.9, 1.0, 0.80, 0.20, 0.10],
        [0.10, 0.80, 1.0, 0.85, 0.2],
        [0.75, 0.20, 0.85, 1.0, 0.3],
        [0.20, 0.10, 0.2, 0.3, 1.0]
    ])
    return base_w, full_w


def main():
    asst = OperAsstNumpy()
    base_w, full_w = build_demo_graph()
    mdl_iput = IputWhl(smth_w=base_w, fit_w=base_w, full_w=full_w, asst_rela=asst)
    hyrdp = HyRdpPaper(mdl_iput, solver=HyRdpClosedForm(), alpha=5.0, beta=25.0, incl_qry=False)
    gmfpt = GmfptPaper(mdl_iput, incl_qry=False)
    print("Single-shot HyRDP:", hyrdp([[0]], 4)[0])
    print("Single-shot GMFPT:", gmfpt([[0]], 4)[0])
    print("Iterative HyRDP:", IHyRdpPaper(hyrdp, nq=4, lambda_step=1)([[0]])[0])
    print("Iterative GMFPT:", IGmfptPaper(gmfpt, nq=4, lambda_step=1)([[0]])[0])


if __name__ == '__main__':
    main()
