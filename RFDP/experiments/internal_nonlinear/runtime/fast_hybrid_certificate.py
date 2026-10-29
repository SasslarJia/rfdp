"""Vectorized outward float64 enclosure of the original HyRDP residual.

Every elementary operation is widened with nextafter in both directions.
Positive sparse row sums use explicit per-column-offset additions (no opaque
BLAS reduction). This is a cheap sufficient certificate, never a new solver.
An inconclusive/overflowing enclosure returns None for arbitrary precision.
"""
import numpy as np


def down(x): return np.nextafter(x, -np.inf)
def up(x): return np.nextafter(x, np.inf)


def certify(w, e, alpha, beta, scores, tolerance=1e-8):
    n=len(e)
    with np.errstate(all='ignore'):
        values=np.asarray([float(v) for v in scores])
        if not np.isfinite(values).all() or not np.all(values>0):return None
        # Enclose the actual mpf/longdouble output before any float conversion.
        lo_x,hi_x=down(values),up(values)
        row=np.arange(n);start=w.indptr[:-1];end=w.indptr[1:]
        width=int(np.max(end-start));lo_h=np.zeros(n);hi_h=np.zeros(n)
        # Accumulate one known positive binary input at a time.
        for offset in range(width):
            active=start+offset<end;ids=row[active];idx=start[active]+offset;weight=w.data[idx]
            lo_h[ids]=down(lo_h[ids]+weight);hi_h[ids]=up(hi_h[ids]+weight)
        if np.any(lo_h<=0) or not np.isfinite(hi_h).all():return None
        lo_root,hi_root=down(np.sqrt(lo_h)),up(np.sqrt(hi_h))
        lo_z,hi_z=np.maximum(0,down(lo_x/hi_root)),up(hi_x/lo_root)
        lo_lap=np.zeros(n);hi_lap=np.zeros(n)
        for offset in range(width):
            active=start+offset<end;ids=row[active];idx=start[active]+offset;j=w.indices[idx];weight=w.data[idx]
            mask=ids!=j;ids=ids[mask];j=j[mask];weight=weight[mask]
            lo_term=down(weight*down(lo_z[ids]-hi_z[j]))
            hi_term=up(weight*up(hi_z[ids]-lo_z[j]))
            lo_lap[ids]=down(lo_lap[ids]+lo_term);hi_lap[ids]=up(hi_lap[ids]+hi_term)
        # All operands of d=alpha*h*e/(h+e) are nonnegative.
        lo_d=down(down(down(alpha*lo_h)*e)/up(hi_h+e))
        hi_d=up(up(up(alpha*hi_h)*e)/down(lo_h+e))
        # Intersect an interval bound with analytically known nonnegativity;
        # no graph, score, or coefficient is clipped or changed.
        lo_d=np.maximum(0,lo_d)
        # Exact e=0 gives exact d=0 (avoid introducing artificial subnormal d).
        lo_d[e==0]=0;hi_d[e==0]=0
        lo_numerator=down(lo_lap+down(lo_d*lo_z))
        hi_numerator=up(hi_lap+up(hi_d*hi_z))
        # Numerator can have either sign: use all endpoint quotients.
        lo_ax=np.minimum(down(lo_numerator/lo_root),down(lo_numerator/hi_root))
        hi_ax=np.maximum(up(hi_numerator/lo_root),up(hi_numerator/hi_root))
        b=np.where(e>0,alpha,beta)
        lo_res=down(b-hi_ax);hi_res=up(b-lo_ax)
        absolute=np.maximum(np.abs(lo_res),np.abs(hi_res))
        eta=float(np.max(up(absolute/b)))
        if not np.isfinite(eta) or eta>=1 or eta>tolerance:return None
        lower=down(lo_x/up(1+eta));upper=up(hi_x/down(1-eta))
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():return None
        return dict(accepted=True,relative=eta,eta=eta,positive=True,
                    intervals=list(zip(lower,upper)),certificate_backend='outward-float64-edge-sums')
