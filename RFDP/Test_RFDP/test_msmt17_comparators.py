"""Numerical equivalence to frozen comparison implementations, on synthetic arrays."""
import ast
from collections import defaultdict
import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace as NS
import unittest

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg

from experiments.msmt17.comparators import REFERENCE
from experiments.msmt17.models import digest
from experiments.msmt17.neighbors import (
    DistanceRows, ecn_rank_blocks, jaccard_encoding, jaccard_rank_blocks, leading_neighbors, unit_float32,
)
from experiments.msmt17.propagation import cheb_propagate, mr_rank_blocks


def features():
    x = np.random.default_rng(20261118).normal(size=(77, 32))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    x[22], x[46] = x[1], x[30]
    return x.astype(np.float32)


def notice(*_):
    pass


def definitions(path, names, namespace):
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    if set(names) != {n.name for n in nodes}:
        raise ValueError("Frozen function definitions are missing")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class SparseECNTests(unittest.TestCase):
    def test_repeated_expansion_neighbors_and_zero_cosine_fallback(self):
        x, nq = features(), 18
        distance = DistanceRows(x, nq, "ecn-rankdist")
        n = len(x)
        direct = distance.rows(0, n)
        rank = np.argsort(direct, axis=1)
        weights = np.maximum(0, 25-np.argsort(rank, axis=1))
        pair_score = -(weights @ weights.T)
        first = rank[:, 1:4]
        expanded = np.c_[first, rank[first, 1:9].reshape(n, -1)]
        mean = (pair_score[expanded[:nq], nq:].sum(axis=1) +
                pair_score[expanded[nq:], :nq].sum(axis=1).T) / 54.
        expected = np.argsort(np.where(mean != 0, mean, direct[nq:, :nq].T),
                              kind="stable", axis=1) + nq
        calculated = np.vstack([v for _, v in ecn_rank_blocks(distance, rank[:, :25], nq, 7)])
        np.testing.assert_array_equal(calculated, expected)


@unittest.skipUnless(os.environ.get("MSMT17_COMPARATOR_SOURCES"),
                     "Set MSMT17_COMPARATOR_SOURCES to the directory of pinned source files")
class FrozenComparatorTests(unittest.TestCase):
    def ref(self, method, filename):
        path = Path(os.environ["MSMT17_COMPARATOR_SOURCES"])/filename
        self.assertEqual(digest(path), REFERENCE[method])
        return path

    def test_author_neighbor_ranks(self):
        x, nq = features(), 18
        q, g = x[:nq], x[nq:]
        cams = np.arange(len(x))%4
        for method, filename in (("k-reciprocal", "kr.py"), ("caj", "caj.py"), ("ecn-rankdist", "ecn.py")):
            with self.subTest(method=method):
                path = self.ref(method, filename)
                spec = importlib.util.spec_from_file_location("frozen_neighbor", path)
                upstream = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(upstream)
                if method == "k-reciprocal":
                    z = unit_float32(x)
                    d = np.sqrt(np.maximum(2.-2.*(z @ z.T), 0.))
                    np.fill_diagonal(d, 0.)
                    result = upstream.re_ranking(d[:nq, nq:], d[:nq, :nq], d[nq:, nq:], 20, 6, .3)
                elif method == "caj":
                    def squared(a, b):
                        return np.maximum(np.sum(a*a, axis=1)[:, None] + np.sum(b*b, axis=1)[None, :]
                                          - 2*a @ b.T, 0.)
                    config = NS(k1=30, k2=6, ckrnns=True, clqe=True, k1_intra=5, k1_inter=20,
                                k2_intra=3, k2_inter=3)
                    result = upstream.re_ranking(squared(q, g), squared(q, q), squared(g, g), cams, config, .3)
                else:
                    result = upstream.ECN(q, g, 25, 3, 8, "rankdist").T
                self.assertTrue(np.isfinite(result).all())
                distance = DistanceRows(x, nq, method)
                k = leading_neighbors(distance, len(x), method, cams, notice, width=13)
                if method == "ecn-rankdist":
                    blocks = ecn_rank_blocks(distance, k, nq, 7)
                else:
                    v = jaccard_encoding(distance, k, len(x), method, notice, width=13)
                    blocks = jaccard_rank_blocks(distance, v, nq, 7)
                got = np.vstack([r for _, r in blocks])
                np.testing.assert_array_equal(got, np.argsort(result, axis=1, kind="stable")+nq)

    def test_mr_matches_archived_cg_with_isolated_nodes(self):
        x, nq = features(), 18
        w = np.maximum(x @ x.T, 0.)
        w[w<.35] = 0.
        w[-1] = 0.
        w[:, -1] = 0.
        graph = sparse.csr_matrix(w)
        path = self.ref("mr-fixed-kernel", "run_reference.py")
        with TemporaryDirectory() as tmp:
            dest = Path(tmp)
            (dest/"queries").mkdir()
            functions = definitions(path, ("make_system", "solve_query"), dict(np=np, sparse=sparse, cg=cg,
                                                                                QN=nq, CELL=dest))
            functions["A"], _, _ = functions["make_system"](graph)
            for q in range(nq):
                functions["solve_query"](q)
            expected = np.stack([np.load(dest/"queries"/f"{q:04d}.npy") for q in range(nq)])
            statistics = []
            actual = np.vstack([rows for _, rows in mr_rank_blocks(graph, nq, 5, 2, statistics)])
            np.testing.assert_array_equal(actual, expected)
            self.assertLessEqual(max(x["residual"] for x in statistics), 1e-9)

    def test_cheb_complete_native_reductions_preserve_duplicate_feature_ties(self):
        import torch
        x = features()
        x = np.vstack([x, x[:10], x[:10]])
        nq = 18
        path = self.ref("cheb-gr", "cheb.py")
        functions = definitions(path, ("fast_mergesetfeat4", "fast_run_gcr"),
                                dict(np=np, torch=torch, time=time, defaultdict=defaultdict))
        config = NS(GCR=NS(WITH_GPU=False, BETA1=.2, BETA2=.2, LAMBDA1=6., LAMBDA2=6., SCALE=0.,
                          GAL_ROUND=3, MODE="nonsym", ENABLE_GCR=True), COMMON=NS(VERBOSE=False))
        with torch.inference_mode():
            a, b = functions["fast_run_gcr"](config, [torch.from_numpy(x[:nq]), np.zeros(nq, int), None,
                                             torch.from_numpy(x[nq:]), np.zeros(len(x)-nq, int), None])
            expected = torch.cat([a, b]).numpy()
        np.testing.assert_array_equal(cheb_propagate(x, "cpu", notice, width=13), expected)
