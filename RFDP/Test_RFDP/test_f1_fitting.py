"""Independent small-matrix checks for the single-update F1 adaptation.

The oracle assembles both ideal systems in 100-digit arithmetic, without the
production matrix, weight, or certificate builders. No retrieval inputs are used.
"""
import copy
import hashlib
import json
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import mpmath as mp
import numpy as np
from scipy import sparse

from experiments.continuous_fitting import certified


def exact_number(value):
    """Lift stored binary inputs without a decimal-string approximation."""
    if isinstance(value, mp.mpf):
        return value
    numerator, denominator = float(value).as_integer_ratio()
    return mp.mpf(numerator) / denominator


def fixture():
    w = np.eye(8)
    for i, j, value in ((0, 1, .75), (1, 2, .5), (2, 3, .25),
                        (3, 4, 1.), (4, 5, .6), (5, 6, .4),
                        (6, 7, .8), (7, 0, .125), (1, 5, .2),
                        (2, 6, .15)):
        w[i, j] = w[j, i] = value
    return sparse.csr_matrix(w), np.array([2., .001, 0., .2, 0., .4, .005, 0.])


def ideal_oracle(w, e, alpha=5., beta=25., source='f1_robust', nu=1.5):
    """Solve the mathematical two-stage definition, independently, at 100 dps."""
    if source not in ('f1_robust', 'f1_uniform'):
        raise ValueError('Oracle source')
    with mp.workdps(100):
        dense = sparse.csr_matrix(w).toarray()
        n = len(e)
        ww = [[exact_number(dense[i, j]) for j in range(n)] for i in range(n)]
        ee = [exact_number(v) for v in e]
        aa, bb, power = exact_number(alpha), exact_number(beta), exact_number(nu)
        h = [mp.fsum(row) for row in ww]
        p = [v / (degree + v) for degree, v in zip(h, ee)]
        direct = [i for i, value in enumerate(ee) if value > 0]
        k = mp.matrix([[mp.mpf(i == j) - ww[i][j] / mp.sqrt(h[i] * h[j])
                        for j in range(n)] for i in range(n)])
        m0 = k.copy()
        b0 = mp.matrix([aa if value > 0 else bb for value in ee])
        for i in direct:
            m0[i, i] += aa * p[i]
        x0 = mp.lu_solve(m0, b0)
        weights = [mp.mpf(1) for _ in range(n)]
        for i in direct:
            r = p[i] * x0[i] - 1
            weights[i] = (1 + r * r) ** (power / 2 - 1)
        if source == 'f1_uniform':
            average = mp.fsum(p[i] * weights[i] for i in direct) / mp.fsum(p)
            for i in direct:
                weights[i] = average
        m1 = k.copy()
        b1 = mp.matrix([aa * weights[i] if ee[i] > 0 else bb for i in range(n)])
        for i in direct:
            m1[i, i] += aa * p[i] * weights[i]
        x1 = mp.lu_solve(m1, b1)
        return dict(k=k, p=p, h=h, e=ee, direct=direct, weights=weights,
                    m0=m0, b0=b0, x0=x0, m1=m1, b1=b1, x1=x1,
                    alpha=aa, beta=bb, nu=power)


def prefix(values):
    return sorted(range(len(values)), key=lambda i: (values[i], i))[:5]


def packed_digest(values):
    return hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()


def robust_energy(values, reference):
    with mp.workdps(100):
        x = mp.matrix([exact_number(value) for value in values])
        result = (x.T * reference['k'] * x)[0] / 2
        for i, p in enumerate(reference['p']):
            if reference['e'][i] > 0:
                r = p * x[i] - 1
                rho = ((1 + r * r) ** (reference['nu'] / 2) - 1) / reference['nu']
                result += reference['alpha'] * rho / p
            else:
                result -= reference['beta'] * x[i]
        return result


class F1FittingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from experiments.continuous_fitting import f1
        cls.f1 = f1

    def setUp(self):
        self.w, self.e = fixture()

    def assert_ideal_scores(self, values, reference, key='x1'):
        with mp.workdps(100):
            actual = mp.matrix([exact_number(value) for value in values])
            expected = reference[key]
            self.assertEqual(len(actual), len(expected))
            self.assertTrue(all(mp.isfinite(v) and v > 0 for v in actual))
            relative_error = max(abs(v - target) / target
                                 for v, target in zip(actual, expected))
            matrix = reference['m0' if key == 'x0' else 'm1']
            rhs = reference['b0' if key == 'x0' else 'b1']
            residual = matrix * actual - rhs
            eta = max(abs(value) / b for value, b in zip(residual, rhs))
            # For this M-matrix, |M^-1 r| <= eta * x_exact. These checks use
            # the ideal high-precision coefficients, not reported residuals.
            self.assertLessEqual(eta, mp.mpf('1e-8'), 'ideal-system residual')
            self.assertLessEqual(relative_error, mp.mpf('1e-8'), 'ideal score error')
            self.assertLessEqual(relative_error, eta + mp.mpf('1e-85'))

    def check_case(self, w, e, *, alpha=5., beta=25., source='f1_robust', nu=1.5):
        reference = ideal_oracle(w, e, alpha, beta, source, nu)
        values, residual, event = self.f1.solve(w, e, alpha=alpha, beta=beta,
                                               source=source, nu=nu)
        self.assert_ideal_scores(values, reference)
        initial = certified.unpack(event['initial_scores'])
        self.assert_ideal_scores(initial, reference, 'x0')
        self.assertEqual(event['updates'], 1)
        self.assertEqual(event['source'], source)
        self.assertEqual(event['nu'], nu)
        self.assertEqual(event['returned_prefix_local'], prefix(reference['x1']))
        self.assertEqual(prefix(values), prefix(reference['x1']))
        self.assertTrue(np.isfinite(float(residual)))
        self.assertLessEqual(float(residual), 1e-8)
        checked = self.f1.certify_saved(w, e, values, event, alpha=alpha,
                                        beta=beta, source=source, nu=nu)
        self.assertEqual(checked['prefix'], prefix(reference['x1']))
        self.assertLessEqual(float(checked['relative']), 1e-8)
        return values, event, reference

    def test_robust_and_uniform_match_independent_ideal_systems(self):
        for source in ('f1_robust', 'f1_uniform'):
            with self.subTest(source=source):
                self.check_case(self.w, self.e, source=source)

    def test_nu_two_recovers_original_equation_and_rank(self):
        original, _, original_event = certified.solve(self.w, self.e, source='original')
        reference = ideal_oracle(self.w, self.e, nu=2.)
        self.assert_ideal_scores(original, reference, 'x0')
        self.assertEqual(original_event['returned_prefix_local'], prefix(reference['x0']))
        for source in ('f1_robust', 'f1_uniform'):
            with self.subTest(source=source):
                values, event, ref = self.check_case(self.w, self.e, source=source, nu=2.)
                self.assert_ideal_scores(values, ref, 'x0')
                self.assertTrue(all(value == 1. for value in event['weights']))

    def test_uniform_is_a_real_global_fitting_control(self):
        robust, robust_event, ref_r = self.check_case(self.w, self.e)
        uniform, uniform_event, ref_u = self.check_case(self.w, self.e, source='f1_uniform')
        direct = ref_r['direct']
        self.assertGreater(max(robust_event['weights'][i] for i in direct)
                           - min(robust_event['weights'][i] for i in direct), .01)
        self.assertTrue(all(uniform_event['weights'][i] == uniform_event['weights'][direct[0]]
                            for i in direct))
        self.assertNotEqual(prefix(robust), prefix(uniform))
        with mp.workdps(100):
            self.assertAlmostEqual(float(mp.fsum(ref_r['p'][i] * ref_r['weights'][i]
                                                for i in direct)),
                                   float(mp.fsum(ref_u['p'][i] * ref_u['weights'][i]
                                                for i in direct)), places=14)
            difference = max(abs(exact_number(a) - exact_number(b)) / exact_number(b)
                             for a, b in zip(robust, uniform))
            self.assertGreater(difference, mp.mpf('1e-4'))

    def test_indirect_linear_term_is_not_multiplied_by_weights(self):
        values, event, reference = self.check_case(self.w, self.e, alpha=2.5, beta=9.5)
        with mp.workdps(100):
            x = mp.matrix([exact_number(value) for value in values])
            laplacian = reference['k'] * x
            for i, boundary in enumerate(self.e):
                if boundary == 0:
                    self.assertEqual(event['weights'][i], 1.)
                    self.assertLess(abs(laplacian[i] - mp.mpf('9.5')) / mp.mpf('9.5'),
                                    mp.mpf('1e-8'))

    def test_nondefault_parameters_are_bound_into_readback(self):
        values, event, _ = self.check_case(self.w, self.e, alpha=2.5, beta=9.5,
                                         source='f1_uniform')
        for change in (dict(alpha=5.), dict(beta=25.), dict(nu=2.),
                       dict(source='f1_robust')):
            arguments = dict(alpha=2.5, beta=9.5, source='f1_uniform', nu=1.5)
            arguments.update(change)
            with self.subTest(change=change), self.assertRaises(
                    (ValueError, AssertionError, np.linalg.LinAlgError)):
                self.f1.certify_saved(self.w, self.e, values, event, **arguments)

    def test_weak_boundary_ill_conditioned_component(self):
        w = np.eye(7)
        for i in range(6):
            w[i, i + 1] = w[i + 1, i] = .25
        e = np.zeros(7)
        e[0] = 2. ** -44
        values, _, reference = self.check_case(sparse.csr_matrix(w), e)
        self.assertGreater(min(reference['x0']), mp.mpf('1e12'))
        self.assertTrue(all(mp.isfinite(value) for value in values))

    def test_disconnected_components_remain_independently_anchored(self):
        w = sparse.block_diag((self.w, sparse.csr_matrix([[1., .25], [.25, 1.]])),
                              format='csr')
        e = np.r_[self.e, [.125, 0.]]
        self.check_case(w, e)
        e[-2:] = 0.
        with self.assertRaises((ValueError, AssertionError, np.linalg.LinAlgError)):
            self.f1.solve(w, e)

    def test_exact_ties_use_existing_local_id_order(self):
        w, e = sparse.eye(7, format='csr'), np.ones(7)
        for source in ('f1_robust', 'f1_uniform'):
            with self.subTest(source=source):
                values, event, _ = self.check_case(w, e, source=source)
                self.assertTrue(all(value == values[0] for value in values))
                self.assertEqual(event['returned_prefix_local'], [0, 1, 2, 3, 4])

    def test_one_ulp_boundary_difference_is_not_a_tie(self):
        w, e = sparse.eye(7, format='csr'), np.ones(7)
        e[-1] = np.nextafter(1., 2.)
        values, event, _ = self.check_case(w, e)
        self.assertLess(values[-1], values[0])
        self.assertEqual(event['returned_prefix_local'], [6, 0, 1, 2, 3])

    def test_one_mm_step_decreases_its_robust_energy(self):
        values, event, reference = self.check_case(self.w, self.e)
        energy0 = robust_energy(reference['x0'], reference)
        energy1 = robust_energy(values, reference)
        self.assertLess(energy1, energy0)
        self.assertEqual(event['updates'], 1)
        # Matching the independent one-update oracle excludes silently running
        # an unconstrained sequence of further IRLS/MM updates.
        self.assert_ideal_scores(values, reference)

    def test_weight_intervals_cover_positive_negative_and_zero_crossing_residuals(self):
        w = sparse.csr_matrix([[1., 0., 0., 0.], [0., 1., 0., 0.],
                               [0., 0., 1., 1.], [0., 0., 1., 1.]])
        e = np.array([1., 1., 2., 0.])
        # The direct p values are all exactly 1/2. These synthetic positive
        # score boxes exercise the interval transform, separately from the
        # question of whether a solver has certified such boxes.
        boxes = [(2.5, 6.), (.5, 1.5), (1., 3.5), (.25, 9.)]
        residual_boxes = [(.25, 2.), (-.75, -.25), (-.5, .75)]
        old_interval_dps = mp.iv.dps
        try:
            mp.iv.dps = 80
            with mp.workdps(100):
                block = certified.original.Block(w, e, 5., 25.)
                for source in ('f1_robust', 'f1_uniform'):
                    for nu in (1.5, 2.):
                        with self.subTest(source=source, nu=nu):
                            extrema = []
                            for a, b in residual_boxes:
                                a, b = exact_number(a), exact_number(b)
                                nearest = mp.mpf(0) if a <= 0 <= b else min(abs(a), abs(b))
                                farthest = max(abs(a), abs(b))
                                exponent = exact_number(nu) / 2 - 1
                                extrema.append(((1 + farthest ** 2) ** exponent,
                                                (1 + nearest ** 2) ** exponent))
                            if source == 'f1_uniform':
                                # Equal exact p values make this independent
                                # weighted-average oracle the ordinary mean.
                                average = (mp.fsum(v[0] for v in extrema) / 3,
                                           mp.fsum(v[1] for v in extrema) / 3)
                                extrema = [average] * 3
                            extrema.append((mp.mpf(1), mp.mpf(1)))
                            float_bounds = self.f1._float_weights(w, e, boxes, source, nu)
                            self.assertIsNotNone(float_bounds)
                            mp_bounds = self.f1._mp_weights(block, boxes, source, nu)
                            for i, (true_lower, true_upper) in enumerate(extrema):
                                self.assertLessEqual(exact_number(float_bounds[0][i]), true_lower)
                                self.assertGreaterEqual(exact_number(float_bounds[1][i]), true_upper)
                                lower = mp.make_mpf(mp_bounds[i]._mpi_[0])
                                upper = mp.make_mpf(mp_bounds[i]._mpi_[1])
                                self.assertLessEqual(lower, true_lower)
                                self.assertGreaterEqual(upper, true_upper)
        finally:
            mp.iv.dps = old_interval_dps

    def test_coupled_certificate_rejects_wide_initial_boxes_with_exact_second_scores(self):
        for source in ('f1_robust', 'f1_uniform'):
            reference = ideal_oracle(self.w, self.e, source=source)
            with mp.workdps(100):
                x0, x1 = reference['x0'], reference['x1']
                exact_residual = reference['m1'] * x1 - reference['b1']
                self.assertLess(max(abs(v) / b for v, b in zip(exact_residual, reference['b1'])),
                                mp.mpf('1e-80'))
                narrow = dict(accepted=True, intervals=[(v, v) for v in x0])
                wide = dict(accepted=True, intervals=[(v / 2, 3 * v / 2) for v in x0])
                for dps in (0, 80):
                    with self.subTest(source=source, dps=dps):
                        accepted = self.f1._coupled_certificate(
                            self.w, self.e, 5., 25., list(x1), narrow, source, 1.5, dps)
                        self.assertIsNotNone(accepted)
                        self.assertTrue(accepted['accepted'])
                        rejected = self.f1._coupled_certificate(
                            self.w, self.e, 5., 25., list(x1), wide, source, 1.5, dps)
                        self.assertFalse(rejected is not None and rejected.get('accepted', False))

    def test_large_sparse_candidate_branch_matches_block_analytic_solution(self):
        blocks = 65
        w = sparse.kron(sparse.eye(blocks, format='csr'),
                        sparse.csr_matrix([[1., .5], [.5, 1.]]), format='csr')
        e = np.zeros(2 * blocks)
        e[::2] = .25 + np.arange(blocks) / 64.
        # Each two-node block has equal h=3/2 and off-diagonal conductance g=1/3.
        # Its original A residual is beta/alpha=5, hence the ideal robust weight
        # is the same in every block. No dense matrix or LU oracle is needed.
        with mp.workdps(100):
            omega = mp.mpf(26) ** (-mp.mpf(1) / 4)
            expected, initial_expected = [], []
            probabilities = []
            for boundary in e[::2]:
                boundary = exact_number(boundary)
                p = boundary / (mp.mpf('1.5') + boundary)
                probabilities.append(p)
                xa = (1 + 5 / omega) / p
                expected.extend((xa, xa + 75))
                x0a = 6 / p
                initial_expected.extend((x0a, x0a + 75))
            expected_prefix = prefix(expected)
        for source in ('f1_robust', 'f1_uniform'):
            with self.subTest(source=source):
                with patch.object(self.f1, 'cg', wraps=self.f1.cg) as cg_call:
                    values, residual, event = self.f1.solve(w, e, source=source)
                self.assertTrue(cg_call.called)
                candidate_matrix = cg_call.call_args.args[0]
                self.assertTrue(sparse.issparse(candidate_matrix))
                self.assertEqual(candidate_matrix.shape, (130, 130))
                self.assertLessEqual(candidate_matrix.nnz, 4 * blocks)
                self.assertEqual(event['candidate_method'], 'cg')
                self.assertEqual(event['linear_systems'], 2)
                self.assertEqual(event['returned_prefix_local'], expected_prefix)
                self.assertEqual(prefix(values), expected_prefix)
                self.assertLessEqual(float(residual), 1e-8)
                with mp.workdps(100):
                    actual = [exact_number(value) for value in values]
                    self.assertLessEqual(max(abs(a - b) / b for a, b in zip(actual, expected)),
                                         mp.mpf('1e-8'))
                    initial = certified.unpack(event['initial_scores'])
                    self.assertLessEqual(max(abs(a - b) / b for a, b in zip(initial, initial_expected)),
                                         mp.mpf('1e-8'))
                    for k, p in enumerate(probabilities):
                        a, b = actual[2 * k:2 * k + 2]
                        ra = (a - b) / 3 + 5 * p * omega * a - 5 * omega
                        rb = (b - a) / 3 - 25
                        self.assertLessEqual(abs(ra) / (5 * omega), mp.mpf('1e-8'))
                        self.assertLessEqual(abs(rb) / 25, mp.mpf('1e-8'))
                with patch.object(self.f1, 'cg', side_effect=AssertionError('readback CG')), \
                        patch.object(self.f1, 'splu', side_effect=AssertionError('readback LU')):
                    checked = self.f1.certify_saved(w, e, values, event, source=source)
                self.assertEqual(checked['prefix'], expected_prefix)

    def test_changed_initial_scores_rejected_even_with_updated_digest(self):
        values, event, _ = self.check_case(self.w, self.e)
        damaged = copy.deepcopy(event)
        initial = certified.unpack(damaged['initial_scores'])
        with mp.workdps(100):
            initial[0] *= mp.mpf('1.01')
            damaged['initial_scores'] = [certified.packed(value) for value in initial]
        damaged['initial_scores_sha256'] = packed_digest(damaged['initial_scores'])
        with self.assertRaises((ValueError, AssertionError, np.linalg.LinAlgError)):
            self.f1.certify_saved(self.w, self.e, values, damaged)

    def test_changed_weight_diagnostics_rejected_with_valid_point_digest(self):
        values, event, _ = self.check_case(self.w, self.e)
        # The digest covers weight_points. Changing only the float diagnostics
        # must still be detected, including a B entry whose actual p*omega is 0.
        for i in (0, 2):
            damaged = copy.deepcopy(event)
            damaged['weights'][i] *= .9
            with self.subTest(index=i), self.assertRaises(
                    (ValueError, AssertionError, np.linalg.LinAlgError)):
                self.f1.certify_saved(self.w, self.e, values, damaged)

    def test_changed_weight_points_rejected_even_with_updated_digest(self):
        values, event, _ = self.check_case(self.w, self.e)
        damaged = copy.deepcopy(event)
        points = certified.unpack(damaged['weight_points'])
        with mp.workdps(100):
            points[0] *= mp.mpf('.9')
            damaged['weight_points'] = [certified.packed(value) for value in points]
            damaged['weights'] = [float(value) for value in points]
        damaged['weights_sha256'] = packed_digest(damaged['weight_points'])
        with self.assertRaises((ValueError, AssertionError, np.linalg.LinAlgError)):
            self.f1.certify_saved(self.w, self.e, values, damaged)

    def test_changed_returned_scores_rejected_even_with_updated_digest(self):
        values, event, _ = self.check_case(self.w, self.e)
        damaged_event = copy.deepcopy(event)
        damaged_values = list(values)
        with mp.workdps(100):
            damaged_values[0] *= mp.mpf('1.01')
            packed = [certified.packed(value) for value in damaged_values]
        damaged_event['returned_scores_sha256'] = packed_digest(packed)
        with self.assertRaises((ValueError, AssertionError, np.linalg.LinAlgError)):
            self.f1.certify_saved(self.w, self.e, damaged_values, damaged_event)

    def test_readback_does_not_solve_or_refine(self):
        values, event, reference = self.check_case(self.w, self.e)
        forbidden = AssertionError('Saved-score verification must not solve')
        with ExitStack() as stack:
            for owner, name in ((self.f1, 'solve'), (certified, 'solve'),
                                (certified.original.Block, 'solve'),
                                (mp, 'lu_solve'), (np.linalg, 'solve')):
                stack.enter_context(patch.object(owner, name, side_effect=forbidden))
            # Block direct aliases too, if the new adapter imports a SciPy solver.
            for name in ('cg', 'splu', 'spsolve'):
                if hasattr(self.f1, name):
                    stack.enter_context(patch.object(self.f1, name, side_effect=forbidden))
            checked = self.f1.certify_saved(self.w, self.e, values, event)
        self.assertEqual(checked['prefix'], prefix(reference['x1']))

    def test_install_emits_one_composite_event_per_scoring_call(self):
        from MODEL.Paper_RFDP import PaperFullIterative as model
        previous = model.solve_hybrid_sparse
        saved_events = list(certified.EVENTS)
        certified.EVENTS.clear()
        try:
            self.f1.install('f1_robust', nu=1.5)
            values, residual = model.solve_hybrid_sparse(self.w, self.e)
            self.assertEqual(len(certified.EVENTS), 1)
            event = certified.EVENTS[0]
            self.assertEqual(event['updates'], 1)
            self.assertEqual(event['source'], 'f1_robust')
            self.assertEqual(len(event['initial_scores']), len(self.e))
            self.assertEqual(len(values), len(self.e))
            self.assertLessEqual(float(residual), 1e-8)
        finally:
            model.solve_hybrid_sparse = previous
            certified.EVENTS[:] = saved_events


if __name__ == '__main__':
    unittest.main()
