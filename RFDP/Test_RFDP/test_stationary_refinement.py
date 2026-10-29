"""Exact, dependency-free checks of fixed-frontier refinement, not retrieval tests."""
from fractions import Fraction as F
from math import isqrt
import unittest


def solve(a, b):
    a = [[F(v) for v in row] + [F(rhs)] for row, rhs in zip(a, b)]
    n = len(b)
    for j in range(n):
        k = next(k for k in range(j, n) if a[k][j])
        a[j], a[k] = a[k], a[j]
        pivot = a[j][j]
        a[j] = [v / pivot for v in a[j]]
        for i in range(n):
            if i != j:
                factor = a[i][j]
                a[i] = [u - factor * v for u, v in zip(a[i], a[j])]
    return [row[-1] for row in a]


def root(q):
    q = F(q)
    a, b = isqrt(q.numerator), isqrt(q.denominator)
    assert a * a == q.numerator and b * b == q.denominator
    return F(a, b)


def equation(w, e, kind, alpha=5, beta=25):
    n = len(e)
    h = list(map(sum, w))
    p = [F(ei, hi + ei) for hi, ei in zip(h, e)]
    b = [F(alpha if ei else beta) for ei in e]
    if kind == 'parent':
        a = [[F(i == j) - w[i][j] / root(h[i] * h[j])
              + (alpha * p[i] if i == j else 0) for j in range(n)] for i in range(n)]
    else:
        a = [[F(h[i] if i == j else 0) - w[i][j]
              + ((e[i] if kind == 'gmfpt' else alpha * h[i] * p[i]) if i == j else 0)
              for j in range(n)] for i in range(n)]
        b = [h[i] + e[i] if kind == 'gmfpt' else h[i] * b[i] for i in range(n)]
    return a, b


def refine(w, e, parts):
    assert all(sum(ts) == 1 and all(t > 0 for t in ts) for ts in parts)
    groups = [(i, F(t)) for i, ts in enumerate(parts) for t in ts]
    wr = [[a * b * w[i][j] for j, b in groups] for i, a in groups]
    er = [a * e[i] for i, a in groups]
    return wr, er, [i for i, _ in groups]


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.w = [[F(v) for v in row] for row in [[2, 1, 1], [1, 3, 0], [1, 0, 3]]]
        self.e = [F(12), F(1, 2), F(0)]

    def check_lift(self, parts, kind):
        wr, er, ids = refine(self.w, self.e, parts)
        x = solve(*equation(self.w, self.e, kind))
        xr = solve(*equation(wr, er, kind))
        self.assertEqual(xr, [x[i] for i in ids])
        return wr, er, ids

    def test_uniform_split_invariance(self):
        self.check_lift([[1], [1], [F(1, 4)] * 4], 'candidate')

    def test_unequal_split_invariance(self):
        self.check_lift([[F(9, 25), F(16, 25)], [1], [F(1, 3), F(2, 3)]], 'candidate')

    def test_lifted_operator_and_mass(self):
        wr, er, ids = refine(self.w, self.e, [[F(1, 3), F(2, 3)], [1], [1]])
        h, hr = list(map(sum, self.w)), list(map(sum, wr))
        a, b = equation(self.w, self.e, 'candidate')
        ar, br = equation(wr, er, 'candidate')
        for u, i in enumerate(ids):
            self.assertEqual(br[u] / hr[u], b[i] / h[i])
            for j in range(len(self.e)):
                self.assertEqual(sum(ar[u][v] / hr[u] for v in range(len(ids)) if ids[v] == j), a[i][j] / h[i])
        self.assertEqual(sum(hr), sum(h))
        self.assertEqual(sum(er), sum(self.e))

    def test_gmfpt_also_has_property(self):
        self.check_lift([[1], [F(2, 7), F(5, 7)], [F(1, 4)] * 4], 'gmfpt')

    def test_parent_reverses_untouched_pair(self):
        wr, er, ids = refine(self.w, self.e, [[1], [1], [F(1, 4)] * 4])
        x = solve(*equation(self.w, self.e, 'parent'))
        xr = solve(*equation(wr, er, 'parent'))
        self.assertEqual(x, [F(732, 91), F(792, 91), F(9832, 91)])
        self.assertEqual(xr, [F(1312, 91), F(972, 91)] + [F(9756, 91)] * 4)
        self.assertLess(x[0], x[1])
        self.assertGreater(xr[0], xr[1])

    def test_unit_coefficients_do_not_make_gmfpt(self):
        self.assertNotEqual(solve(*equation(self.w, self.e, 'candidate', 1, 1)),
                            solve(*equation(self.w, self.e, 'gmfpt')))

    def test_plain_photo_duplication_is_not_mass_refinement(self):
        # Copying similarities without dividing them is outside the theorem.
        ids = [0, 1, 2, 2]
        wr = [[self.w[i][j] for j in ids] for i in ids]
        er = [self.e[i] for i in ids]
        x = solve(*equation(self.w, self.e, 'candidate'))
        xr = solve(*equation(wr, er, 'candidate'))
        self.assertNotEqual(xr, [x[i] for i in ids])


if __name__ == '__main__':
    unittest.main()
