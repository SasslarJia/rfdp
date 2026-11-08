"""Converged-F1 engineering checks against closed-form mathematical targets.

No retrieval inputs, identity labels, parameter search, or timing estimates are
used. The fixed graph has two regular anchored components and two singleton
anchors. Their robust stationary equations reduce to scalar invertible forms.
"""
import copy
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import mpmath as mp
import numpy as np
from scipy import sparse

from experiments.continuous_fitting import certified, f1

ROBUST = 'f1_robust_converged'
UNIFORM = 'f1_uniform_converged'


def fixed_graph():
    w = np.eye(7)
    w[0, 1] = w[1, 0] = .25
    for i in range(2, 5):
        for j in range(i + 1, 5):
            w[i, j] = w[j, i] = .25
    return sparse.csr_matrix(w), np.array([.5, 0., 1., 0., 0., 2., 4.])


def exact_number(value):
    if isinstance(value, mp.mpf):
        return value
    numerator, denominator = float(value).as_integer_ratio()
    return mp.mpf(numerator) / denominator


def closed_form_targets():
    """Independent solutions; no production builders or numerical optimizer."""
    with mp.workdps(100):
        p = [mp.mpf(2) / 7, 0, mp.mpf(2) / 5, 0, 0,
             mp.mpf(2) / 3, mp.mpf(4) / 5]
        direct = [0, 2, 5, 6]

        def inverse_phi(s):
            return s * mp.sqrt((s * s + mp.sqrt(s**4 + 4)) / 2)

        r0, r2 = inverse_phi(mp.mpf(5)), inverse_phi(mp.mpf(10))
        robust = [(1 + r0) / p[0], 0, (1 + r2) / p[2], 0, 0,
                  1 / p[5], 1 / p[6]]
        # Equal degree within each component makes the B equations exact:
        # (x_B-x_A)/5=25 for the pair, (x_B-x_A)/6=25 for the triple.
        robust[1] = robust[0] + 125
        robust[3] = robust[4] = robust[2] + 150
        weights = [(1 + (p[i] * robust[i] - 1)**2)**(-mp.mpf(1) / 4)
                   if i in direct else mp.mpf(1) for i in range(7)]
        average = mp.fsum(p[i] * weights[i] for i in direct) / mp.fsum(p)
        uniform = [(1 + 5 / average) / p[0], 0,
                   (1 + 10 / average) / p[2], 0, 0, 1 / p[5], 1 / p[6]]
        uniform[1] = uniform[0] + 125
        uniform[3] = uniform[4] = uniform[2] + 150
        # The old one-MM definition uses r=5 and r=10 from the quadratic
        # initial solution, not the converged r above.
        first = [(1 + 5 * mp.root(26, 4)) / p[0], 0,
                 (1 + 10 * mp.root(101, 4)) / p[2], 0, 0, 1 / p[5], 1 / p[6]]
        first[1] = first[0] + 125
        first[3] = first[4] = first[2] + 150
        initial_weights = [mp.mpf(1) for _ in range(7)]
        initial_weights[0] = mp.power(26, -mp.mpf(1) / 4)
        initial_weights[2] = mp.power(101, -mp.mpf(1) / 4)
        return dict(robust=robust, uniform=uniform, first=first, p=p,
                    direct=direct, weights=weights, average=average,
                    initial_weights=initial_weights)


def independent_residual(values, *, source, target):
    """Evaluate the declared target using independently assembled coefficients."""
    with mp.workdps(100):
        dense = fixed_graph()[0].toarray()
        w = [[exact_number(v) for v in row] for row in dense]
        h = [mp.fsum(row) for row in w]
        k = mp.matrix([[mp.mpf(i == j) - w[i][j] / mp.sqrt(h[i] * h[j])
                        for j in range(7)] for i in range(7)])
        x = mp.matrix([exact_number(v) for v in values])
        gradient = k * x
        rhs = []
        for i in range(7):
            if i in target['direct']:
                r = target['p'][i] * x[i] - 1
                weight = ((1 + r * r)**(-mp.mpf(1) / 4)
                          if source == ROBUST else target['average'])
                gradient[i] += 5 * weight * r
                rhs.append(5 * weight)
            else:
                gradient[i] -= 25
                rhs.append(mp.mpf(25))
        return max(abs(v) / b for v, b in zip(gradient, rhs))


class ConvergedFittingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from experiments.continuous_fitting import converged
        cls.method = converged
        cls.w, cls.e = fixed_graph()
        cls.target = closed_form_targets()
        cls.solutions = {source: converged.solve(cls.w, cls.e, source=source)
                         for source in (ROBUST, UNIFORM)}

    def test_both_sources_match_independent_closed_form_targets(self):
        for source, key in ((ROBUST, 'robust'), (UNIFORM, 'uniform')):
            values, residual, event = self.solutions[source]
            expected = self.target[key]
            with self.subTest(source=source), mp.workdps(100):
                self.assertTrue(all(mp.isfinite(exact_number(v)) and v > 0 for v in values))
                relative = max(abs(exact_number(v) - t) / t for v, t in zip(values, expected))
                self.assertLessEqual(relative, mp.mpf('1e-8'))
                self.assertLessEqual(independent_residual(values, source=source, target=self.target),
                                     mp.mpf('1e-8'))
                self.assertLessEqual(float(residual), 1e-8)
                expected_prefix = sorted(range(7), key=lambda i: (expected[i], i))[:5]
                self.assertEqual(event['returned_prefix_local'], expected_prefix)
                self.assertEqual(event['source'], source)
                self.assertEqual(event['nu'], 1.5)

    def test_converged_target_is_not_the_old_one_mm_output(self):
        values = self.solutions[ROBUST][0]
        actual = np.asarray(values, dtype=float)
        first = np.asarray(self.target['first'], dtype=float)
        self.assertGreater(np.max(np.abs(actual - first)), 1.)
        # This checks the target equation, not just an implementation's
        # claimed iteration count or convergence flag.
        self.assertGreater(float(independent_residual(first, source=ROBUST,
                                                     target=self.target)), 1e-3)

    def test_intermediate_updates_do_not_trigger_per_step_mp_certification(self):
        # Regression for the cost bug: a regular well-resolved graph should
        # not enter a persistent MP loop merely because MM is converging.
        for source in (ROBUST, UNIFORM):
            event = self.solutions[source][2]
            self.assertGreater(event['mm_iterations'], 10)
            self.assertLess(event['reference_certifications'], event['mm_iterations']/4)
            self.assertEqual(event['mp_mm_iterations'], 0)
            self.assertLess(event['refinement_solve_calls'], event['mm_iterations']/4)

    def test_previous_certificate_format_remains_readable(self):
        for source in (ROBUST, UNIFORM):
            values, _, event = self.solutions[source]
            for solver_id in (self.method.LEGACY_SOLVER_ID,self.method.PREVIOUS_SOLVER_ID):
                old = copy.deepcopy(event)
                old['solver_id'] = solver_id
                for key in ('additional_factorizations', 'reference_certifications', 'mp_mm_iterations'):
                    old.pop(key)
                checked = self.method.certify_saved(self.w, self.e, values, old, source=source)
                self.assertEqual(checked['prefix'], event['returned_prefix_local'])

    def test_successful_correction_clears_preceding_stagnation_flag(self):
        from experiments.continuous_fitting.fitting_workspace import LinearSystem
        # A corrective solve makes progress after the candidate stayed put.
        # The old flag must not latch all subsequent MM updates into MP.
        with patch.object(LinearSystem, 'candidate',
                          side_effect=lambda previous, rtol: np.asarray(previous, dtype=float)):
            values, residual, event = self.method.solve(self.w, self.e, source=ROBUST)
        self.assertEqual(event['mp_mm_iterations'],0)
        self.assertLessEqual(residual,1e-8)
        self.assertLessEqual(float(independent_residual(values,source=ROBUST,target=self.target)),1e-8)
        self.method.certify_saved(self.w,self.e,values,event,source=ROBUST)

    def test_unrepaired_stagnation_retains_mp_recovery(self):
        from experiments.continuous_fitting.fitting_workspace import LinearSystem
        actual=self.method._attempt_reference
        calls=[]
        def first_check_cannot_correct(work,system,values,counters,minimum_dps):
            calls.append(1)
            if len(calls)==1:return values,None,None,minimum_dps
            return actual(work,system,values,counters,minimum_dps)
        with patch.object(LinearSystem,'candidate',side_effect=lambda previous,rtol: np.asarray(previous,dtype=float)), \
             patch.object(self.method,'_attempt_reference',side_effect=first_check_cannot_correct):
            values,residual,event=self.method.solve(self.w,self.e,source=ROBUST)
        self.assertGreater(event['mp_mm_iterations'],0)
        self.assertLessEqual(residual,1e-8)
        self.assertLessEqual(float(independent_residual(values,source=ROBUST,target=self.target)),1e-8)
        self.method.certify_saved(self.w,self.e,values,event,source=ROBUST)

    def test_weak_anchor_with_large_self_loops_matches_analytic_target(self):
        # A fixed numerical-conditioning case, not a retrieval/parameter trial.
        w = sparse.csr_matrix([[1e8, .25], [.25, 1e8]])
        e = np.array([.5, 0.])
        values, residual, event = self.method.solve(w, e, source=ROBUST)
        with mp.workdps(100):
            h = mp.mpf(10)**8 + mp.mpf(1)/4
            p = mp.mpf('.5')/(h+mp.mpf('.5'))
            r = 5*mp.sqrt((25+mp.sqrt(629))/2)
            target = [(1+r)/p, (1+r)/p+100*h]
            self.assertLessEqual(max(abs(exact_number(x)-y)/y for x,y in zip(values,target)),
                                 mp.mpf('1e-8'))
        self.assertLessEqual(residual, 1e-8)
        self.method.certify_saved(w, e, values, event, source=ROBUST)

    def test_uniform_uses_the_converged_reference_including_zero_residual_anchors(self):
        robust, uniform = [np.asarray(self.solutions[s][0], dtype=float)
                           for s in (ROBUST, UNIFORM)]
        self.assertGreater(abs(float(self.target['weights'][0] - self.target['weights'][2])), .01)
        self.assertEqual(self.target['weights'][5], 1)
        self.assertEqual(self.target['weights'][6], 1)
        self.assertGreater(np.max(np.abs(robust - uniform)), 1.)
        # The independent uniform residual uses the p-weighted average of
        # all A anchors at the exact robust optimum, including r=0 anchors.
        self.assertLessEqual(float(independent_residual(uniform, source=UNIFORM,
                                                       target=self.target)), 1e-8)

    def test_nu_is_fixed_for_solving_installation_and_saved_readback(self):
        values, _, event = self.solutions[ROBUST]
        for nu in (1., 2.):
            with self.subTest(nu=nu):
                with self.assertRaises((TypeError, ValueError)):
                    self.method.solve(self.w, self.e, source=ROBUST, nu=nu)
                with self.assertRaises((TypeError, ValueError)):
                    self.method.install(ROBUST, nu=nu)
                with self.assertRaises((TypeError, ValueError)):
                    self.method.certify_saved(self.w, self.e, values, event,
                                              source=ROBUST, nu=nu)

    def test_iteration_limit_cannot_return_an_unconverged_approximation(self):
        # On this fixture the exact first MM output is independently known
        # above and fails the nonlinear target. A failure cap is not a new
        # approximation method or permission to return that point.
        for source in (ROBUST, UNIFORM):
            with self.subTest(source=source), self.assertRaises(certified.original.ArithmeticFailure):
                self.method.solve(self.w, self.e, source=source, max_mm_iterations=1)

    def test_saved_readback_does_not_resolve_or_replace_saved_points(self):
        with ExitStack() as stack:
            for obj, name in ((self.method, 'solve'), (certified, 'solve'),
                              (certified.original.Block, 'solve'), (f1.WeightedBlock, 'solve')):
                stack.enter_context(patch.object(obj, name,
                                                 side_effect=AssertionError('readback ran a solver')))
            for source in (ROBUST, UNIFORM):
                values, _, event = self.solutions[source]
                checked = self.method.certify_saved(self.w, self.e, values, event, source=source)
                self.assertEqual(checked['prefix'], event['returned_prefix_local'])
                self.assertLessEqual(float(checked['relative']), 1e-8)

    def test_changed_saved_point_is_rejected_even_with_an_updated_score_digest(self):
        for source in (ROBUST, UNIFORM):
            values, _, original_event = self.solutions[source]
            with mp.workdps(100):
                damaged = [exact_number(v) * mp.mpf('1.01') for v in values]
            event = copy.deepcopy(original_event)
            event['returned_scores_sha256'] = certified.digest_scores(damaged)
            if source == ROBUST:
                event['certificate']['scores_sha256'] = certified.digest_scores(damaged)
            with self.subTest(source=source), self.assertRaises(
                    (ValueError, AssertionError, np.linalg.LinAlgError)):
                self.method.certify_saved(self.w, self.e, damaged, event, source=source)

    def test_uniform_cannot_relabel_a_certified_one_step_reference_as_converged(self):
        values, _, original_event = self.solutions[UNIFORM]
        event = copy.deepcopy(original_event)
        reference = self.target['first']
        weights = self.target['initial_weights']
        event['reference_scores'] = [certified.packed(v) for v in reference]
        event['reference_scores_sha256'] = certified.digest_scores(reference)
        record = event['reference_certificate']
        record['scores_sha256'] = certified.digest_scores(reference)
        record['frozen_weight_points'] = [certified.packed(v) for v in weights]
        record['frozen_weights_sha256'] = certified.digest_scores(weights)
        # The substituted point really solves its frozen weighted system:
        # it must fail the final nonlinear certificate, not just a checksum.
        with self.assertRaisesRegex(certified.original.ArithmeticFailure, 'nonlinear'):
            self.method.certify_saved(self.w, self.e, values, event, source=UNIFORM)

    def test_saved_mechanism_weights_and_certificate_diagnostics_cannot_be_changed(self):
        cases = [
            (UNIFORM, ('reference_weights', 0), 0.),
            (UNIFORM, ('reference_weight_points', 0), certified.packed(mp.mpf(1))),
            (UNIFORM, ('reference_certificate', 'nonlinear_residual_upper'), 1.),
            (UNIFORM, ('reference_certificate', 'relative_score_error_upper'), 1.),
            (UNIFORM, ('reference_certificate', 'inner_linear_residual_upper'), 1.),
            (UNIFORM, ('certificate', 'uniform_weight_interval', 0), certified.packed(mp.mpf(0))),
            (ROBUST, ('certificate', 'nonlinear_residual_upper'), 1.),
        ]
        for source in (ROBUST, UNIFORM):
            cases.extend((source, (key,), replacement) for key, replacement in (
                ('returned_residual_upper', 1.),
                ('relative_score_error_upper', 1.),
                ('ordered_first_five_certified', False)))
        for source, path, replacement in cases:
            values, _, original_event = self.solutions[source]
            event = copy.deepcopy(original_event)
            target = event
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = replacement
            with self.subTest(source=source, field=path), self.assertRaises(ValueError):
                self.method.certify_saved(self.w, self.e, values, event, source=source)

    def test_saved_algorithm_and_input_binding_are_checked(self):
        values, _, event = self.solutions[ROBUST]
        with self.assertRaises((ValueError, AssertionError, np.linalg.LinAlgError)):
            self.method.certify_saved(self.w, self.e, values, event, source=UNIFORM)
        boundary = self.e.copy()
        boundary[0] *= 2.
        with self.assertRaises((ValueError, AssertionError, np.linalg.LinAlgError)):
            self.method.certify_saved(self.w, boundary, values, event, source=ROBUST)

    def test_unanchored_component_is_rejected(self):
        boundary = self.e.copy()
        boundary[:2] = 0.
        with self.assertRaises((ValueError, AssertionError, np.linalg.LinAlgError)):
            self.method.solve(self.w, boundary, source=ROBUST)

    def test_install_preserves_the_existing_ranker_score_and_event_interface(self):
        from MODEL.Paper_RFDP import PaperFullIterative as model
        original_solver = model.solve_hybrid_sparse
        existing_events = certified.EVENTS[:]
        values, residual, event = self.solutions[ROBUST]
        try:
            certified.EVENTS.clear()
            previous = self.method.install(ROBUST)
            self.assertIs(previous, original_solver)
            with patch.object(self.method, 'solve', return_value=(values, residual, event)) as solve:
                diagnostics = {}
                returned, reported = model.solve_hybrid_sparse(
                    self.w, self.e, alpha=5., beta=25., diagnostics=diagnostics)
            self.assertEqual(solve.call_count, 1)
            self.assertIs(returned, values)
            self.assertEqual(reported, residual)
            self.assertEqual(certified.EVENTS, [event])
            self.assertIn('direct_calls', diagnostics)
        finally:
            model.solve_hybrid_sparse = original_solver
            certified.EVENTS[:] = existing_events


if __name__ == '__main__':
    unittest.main()
