"""The registered fixed-kernel MR and nonsymmetric Cheb-GR configurations."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg


def mr_system(graph):
    graph = graph.astype(np.float64).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    if (graph-graph.T).nnz or np.any(graph.data < 0):
        raise ValueError("Invalid fixed-kernel MR graph")
    degree = np.asarray(graph.sum(axis=1)).ravel()
    inv = np.zeros_like(degree)
    np.divide(1., np.sqrt(degree), out=inv, where=degree>0.)
    normalized = sparse.diags(inv) @ graph @ sparse.diags(inv)
    return (sparse.eye(len(degree)) - .99*normalized).tocsr()


def mr_rank_blocks(graph, nq, width, workers, stats, start=0):
    matrix = mr_system(graph)

    def rank(q):
        rhs = np.zeros(matrix.shape[0], dtype=np.float64)
        rhs[q] = 1.
        count = 0

        def step(_):
            nonlocal count
            count += 1

        scores, status = cg(matrix, rhs, rtol=1e-11, atol=0., maxiter=5000, callback=step)
        residual = float(np.linalg.norm(matrix@scores-rhs))
        if status != 0 or not np.isfinite(scores).all() or residual > 1e-9:
            raise ArithmeticError(f"MR failed at query {q}: {status}, {residual}")
        return np.argsort(-scores[nq:], kind="stable").astype(np.int32)+nq, residual, count

    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as pool:
        for lo in range(start, nq, width):
            result = list(pool.map(rank, range(lo, min(lo+width, nq))))
            stats.extend([dict(query_id=lo+i, residual=r, iterations=n) for i, (_, r, n) in enumerate(result)])
            yield lo, np.stack([row for row, _, _ in result])


def cheb_propagate(features, device, notify, width=128):
    """Execute the full-shaped native reductions and GEMMs with one N² buffer.

    Overwriting the distance after thresholding and S after degree calculation
    releases their original storage. Only the elementwise mask and outer product
    are tiled. The complete row and column reduction shapes are preserved.
    """
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x = torch.tensor(np.asarray(features), dtype=torch.float32, device=device)
    n = len(x)
    with torch.inference_mode():
        buffer = torch.empty((n, n), dtype=x.dtype, device=x.device)
        for round_idx in range(3):
            squares = (x*x).sum(dim=1, keepdim=True)
            torch.mm(-2.*x, x.T, out=buffer)
            buffer.add_(squares).add_(squares.T).clamp_min_(0.)
            threshold = torch.clamp(buffer.mean(dim=1, keepdim=True) -
                                    6.*buffer.std(dim=1, keepdim=True), min=buffer.diagonal().max())
            notify(f"cheb_{round_idx+1}_distances", n, n)
            for lo in range(0, n, width):
                hi = min(lo+width, n)
                part = buffer[lo:hi]
                part.masked_fill_(part > threshold[lo:hi], float("inf"))
            buffer.neg_().div_(.2).exp_()
            row_degree, col_degree = buffer.sum(dim=1), buffer.sum(dim=0)
            if not torch.all((row_degree>0) & (col_degree>0)):
                raise ArithmeticError("Zero Cheb-GR propagation degree")
            dr, dc = torch.sqrt(1./row_degree), torch.sqrt(1./col_degree)
            for lo in range(0, n, width):
                hi = min(lo+width, n)
                buffer[lo:hi].mul_(torch.outer(dr[lo:hi], dc))
            new = buffer @ x
            x = new / torch.linalg.norm(new, ord=2, dim=1, keepdim=True)
            del new
            notify(f"cheb_{round_idx+1}_propagation", n, n)
    return x.cpu().numpy()
