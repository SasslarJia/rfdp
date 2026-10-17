"""Synthetic metadata tests only; no feature loader, ranking or network calls."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import plan_uniform_suite as suite
import input_binding as runner_binding


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


class SyntheticSuite(unittest.TestCase):
    def setUp(self):
        # Every test write stays within the designated integration-draft tree.
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-plan-", dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root, self.output = self.base / "v2", self.base / "next"
        self.source = self.output / "source"
        self.source.mkdir(parents=True)
        (self.root / "repo").mkdir(parents=True)
        self.protocol = self.output / "PROTOCOL.md"
        self.protocol.write_text("synthetic uniform protocol\n")
        for name in ("run_uniform_candidate.py", "verify_uniform_control.py", "verify_experiment.py", "verify_numerics.py"):
            (self.source / name).write_text("# synthetic stub, never launched\n")
        (self.root / "PROTOCOL.md").write_text("synthetic frozen v2 protocol\n")
        self.old_protocol = suite.sha(self.root / "PROTOCOL.md")
        self.verifier_sha, self.recovery_sha = "b" * 64, "c" * 64
        self.entries = {}
        self.selected = dict(protocol_sha256=self.old_protocol, models={})
        for model in suite.MODELS:
            self.make_inputs(model)
            for split in ("val", "test"):
                for rho in (suite.RHOS if split == "val" else (.25, .5)):
                    self.add_report(model, split, None, rho, 0.)
                for mode in suite.MODES:
                    lambdas = suite.LAMBDAS if split == "val" else (.25 if mode == "ordinary" else .75,)
                    for lam in lambdas:
                        self.add_report(model, split, mode, .5, lam)
            self.selected["models"][model] = {
                "rho": self.entries[(model, "val", None, .25, 0.)].copy(),
                "ordinary": self.entries[(model, "val", "ordinary", .5, .25)].copy(),
                "cb": self.entries[(model, "val", "cb", .5, .75)].copy(),
            }
            dump(self.root / ("regression-" + model + ".json"), dict(status="passed", old_metrics_sha256="d" * 64,
                 new_verified_sha256=self.entries[(model, "test", None, .5, 0.)]["verified_sha256"], different_rankings=[]))
        self.selection_file = self.root / "selection.json"
        dump(self.selection_file, self.selected)
        self.selection_sha = suite.sha(self.selection_file)
        self.complete = dict(status="verified", selection_sha256=self.selection_sha,
                             development=[entry for key, entry in self.entries.items() if key[1] == "val"],
                             test=[entry for key, entry in self.entries.items() if key[1] == "test"])
        self.save_complete()

    def make_inputs(self, model, tag="original"):
        mat = self.root / "features" / ("test-" + model + ".mat")
        mat.parent.mkdir(parents=True, exist_ok=True)
        mat.write_text(model + " " + tag + " synthetic MAT\n")
        cache = self.root / "cache" / ("test-" + model)
        cache.mkdir(parents=True, exist_ok=True)
        for name in suite.CACHE_FILES[1:]:
            (cache / name).write_text(model + " " + tag + " synthetic " + name + "\n")
        dump(cache / "manifest.json", dict(feature_sha256=suite.sha(mat),
             files={name: suite.sha(cache / name) for name in suite.CACHE_FILES[1:]}))
        for mode in suite.MODES:
            support = self.root / "supports" / ("test-" + model + "-" + mode + ".npz")
            support.parent.mkdir(parents=True, exist_ok=True)
            support.write_text(model + " " + mode + " " + tag + " synthetic support\n")

    def add_report(self, model, split, mode, rho, lam):
        method = suite.method_name(mode, rho, lam)
        run = self.root / "runs" / split / model / method
        nq, ng = suite.DIMENSIONS[split]
        config = dict(method=method, rho=rho, support_mode=mode, **{"lambda": lam}, query_count=nq,
                      gallery_count=ng, depth=500, protocol_sha256=self.old_protocol, feature_sha256="e" * 64,
                      alpha=5, beta=25, step=5, k=20, gamma=.4, numerical_solver="same-float64-exact-rational-mp-v1")
        if split == "test":
            mat = self.root / "features" / ("test-" + model + ".mat")
            cache = self.root / "cache" / ("test-" + model)
            support = self.root / "supports" / ("test-" + model + "-" + str(mode) + ".npz") if mode else None
            config.update(feature_sha256=suite.sha(mat), support_sha256=suite.sha(support) if support else None,
                          support_file=str(support) if support else None, cache_manifest_path=str(cache / "manifest.json"),
                          inputs={str(cache / name): suite.sha(cache / name) for name in suite.CACHE_FILES})
        manifest_path = run / "run-manifest.json"
        dump(manifest_path, dict(config=config, signature=suite.canonical(config)))
        report = dict(status="verified", method=method, input=dict(query_count=nq, gallery_count=ng),
                      input_mat_sha256=config["feature_sha256"],
                      manifest=dict(path=str(manifest_path), signature=suite.canonical(config)),
                      checks=dict(full_permutation_queries=nq, independent_metric_queries=nq,
                                  diffusion_checkpoints=nq, no_diffusion=False, diffusion_depth=500,
                                  required_max_relative_residual=1e-8, max_relative_residual=1e-9),
                      files={str(manifest_path): dict(sha256=suite.sha(manifest_path)),
                             str(self.root / "source/verify_numerics.py"): dict(sha256=self.verifier_sha),
                             str(self.root / "source/high_precision_recovery.py"): dict(sha256=self.recovery_sha)})
        report_path = run / ("verified-" + method + ".json")
        dump(report_path, report)
        entry = dict(model=model, split=split, method=method, rho=rho, lambda_=lam, mode=mode,
                     verified_path=str(report_path), verified_sha256=suite.sha(report_path))
        self.entries[(model, split, mode, rho, lam)] = entry
        dump(run / "verified-numerical-evidence.json", dict(status="verified", method=method,
             query_checkpoints=nq, standard_report_sha256=entry["verified_sha256"], verifier_sha256=self.verifier_sha,
             recovery_sha256=self.recovery_sha, max_relative_residual=1e-9, arbitrary_precision_events=0, files={}))

    def save_complete(self):
        dump(self.root / "COMPLETE.json", self.complete)

    def replace_bound_test_manifest(self, entry, mutate):
        """Build internally consistent synthetic evidence for an invalid contract."""
        report_path = Path(entry["verified_path"])
        manifest_path = report_path.parent / "run-manifest.json"
        manifest = suite.load(manifest_path)
        mutate(manifest["config"])
        manifest["signature"] = suite.canonical(manifest["config"])
        dump(manifest_path, manifest)
        report = suite.load(report_path)
        report["manifest"]["signature"] = manifest["signature"]
        report["input_mat_sha256"] = manifest["config"]["feature_sha256"]
        report["files"][str(manifest_path)]["sha256"] = suite.sha(manifest_path)
        dump(report_path, report)
        entry["verified_sha256"] = suite.sha(report_path)
        numerical_path = report_path.parent / "verified-numerical-evidence.json"
        numerical = suite.load(numerical_path)
        numerical["standard_report_sha256"] = entry["verified_sha256"]
        dump(numerical_path, numerical)
        self.save_complete()

    def options(self):
        return dict(v2_root=self.root, output_root=self.output, selection_file=self.selection_file,
                    selection_sha256=self.selection_sha, protocol=self.protocol, source_dir=self.source)

    def plan(self):
        return suite.build_plan(**self.options())

    def binding_arguments(self, job):
        model, mode = job["model"], job["source_support_mode"]
        return dict(model=model, mode=mode, source_lambda=job["source_lambda"], rho=.5,
                    selection_sha256=self.selection_sha, mat=self.root / ("features/test-" + model + ".mat"),
                    cache=self.root / ("cache/test-" + model), support=self.root / ("supports/test-" + model + "-" + mode + ".npz"),
                    query_count=3368, gallery_count=19732, depth=500)

    def failed_gate(self, fragment, name="v2_complete_grid_and_bound_evidence"):
        plan = self.plan()
        self.assertFalse(plan["execution_ready"])
        gate = next(item for item in plan["gates"] if item["name"] == name)
        self.assertIn(fragment, gate.get("reason", json.dumps(gate)))
        with patch.object(suite.subprocess, "run") as run:
            with self.assertRaises(suite.GateError):
                suite.execute(plan)
            run.assert_not_called()

    def test_complete_grid_four_fixed_pairs_without_any_metrics(self):
        plan = self.plan()
        self.assertTrue(plan["execution_ready"])
        self.assertEqual(len(plan["jobs"]), 4)
        self.assertEqual([(j["model"], j["source_support_mode"], j["source_lambda"]) for j in plan["jobs"]],
                         [(m, mode, .25 if mode == "ordinary" else .75) for m in suite.MODELS for mode in suite.MODES])
        for job in plan["jobs"]:
            self.assertEqual(job["rho"], .5)
            runner = job["commands"][0]
            self.assertEqual(runner[runner.index("--selection-sha256") + 1], self.selection_sha)
            self.assertEqual(runner[runner.index("--expected-queries") + 1], "3368")
            self.assertEqual(runner[runner.index("--expected-gallery") + 1], "19732")
            self.assertEqual(runner[runner.index("--input-binding-sha256") + 1], suite.canonical(job["input_binding"]))
            self.assertEqual(runner[runner.index("--input-binding-file") + 1], job["input_binding_file"])
            self.assertEqual(job["input_binding"]["scope"], "frozen-v2-test")
            self.assertEqual(job["input_binding"]["model"], job["model"])
            self.assertEqual(set(job["input_binding"]["cache_files"]), set(suite.CACHE_FILES))
            self.assertEqual(len(job["commands"]), 4)

    def test_cli_default_only_prints_plan(self):
        args = []
        for key, value in self.options().items():
            args.extend(["--" + key.replace("_", "-"), str(value)])
        before = sorted(str(p) for p in self.base.rglob("*"))
        with patch.object(suite.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(suite.main(args), 0)
            run.assert_not_called()
        self.assertEqual(json.loads(stream.getvalue())["status"], "planned")
        self.assertEqual(before, sorted(str(p) for p in self.base.rglob("*")))

    def test_missing_complete_still_produces_blocked_four_job_plan(self):
        (self.root / "COMPLETE.json").unlink()
        self.failed_gate("COMPLETE.json")
        self.assertEqual(len(self.plan()["jobs"]), 4)

    def test_explicit_selection_sha_is_mandatory_and_exact(self):
        options = self.options()
        options["selection_sha256"] = "0" * 64
        with self.assertRaisesRegex(suite.GateError, "Explicit selection SHA"):
            suite.build_plan(**options)

    def test_complete_must_bind_frozen_selection(self):
        self.complete["selection_sha256"] = "0" * 64
        self.save_complete()
        self.failed_gate("COMPLETE selection SHA")

    def test_duplicate_development_cell_cannot_replace_missing_cell(self):
        self.complete["development"][-1] = self.complete["development"][0]
        self.save_complete()
        self.failed_gate("duplicate or unexpected development")

    def test_wrong_test_grid_same_count_rejected(self):
        self.complete["test"][0]["rho"] = .75
        self.save_complete()
        self.failed_gate("unexpected test grid")

    def test_partial_eight_test_grid_rejected(self):
        self.complete["test"].pop()
        self.save_complete()
        self.failed_gate("test configuration count")

    def test_changed_report_bytes_rejected(self):
        path = Path(self.complete["test"][0]["verified_path"])
        path.write_text(path.read_text() + "\n")
        self.failed_gate("Verified report SHA mismatch")

    def test_missing_numerics_and_wrong_binding_rejected(self):
        path = Path(self.complete["test"][0]["verified_path"]).parent / "verified-numerical-evidence.json"
        record = suite.load(path)
        record["standard_report_sha256"] = "0" * 64
        dump(path, record)
        self.failed_gate("Numerical verification is not bound")
        path.unlink()
        self.failed_gate("verified-numerical-evidence.json")

    def test_numerics_source_and_residual_must_remain_bound(self):
        path = Path(self.complete["test"][0]["verified_path"]).parent / "verified-numerical-evidence.json"
        record = suite.load(path)
        record["recovery_sha256"] = "0" * 64
        dump(path, record)
        self.failed_gate("Numerical verifier/source binding differs")
        record["recovery_sha256"] = self.recovery_sha
        record["max_relative_residual"] = 2e-8
        dump(path, record)
        self.failed_gate("unchanged threshold")

    def test_changed_manifest_rejected(self):
        path = Path(self.complete["test"][0]["verified_path"]).parent / "run-manifest.json"
        record = suite.load(path)
        record["config"]["rho"] = .75
        dump(path, record)
        self.failed_gate("Run manifest SHA mismatch")

    def test_both_historical_regressions_must_pass_and_bind_baseline(self):
        path = self.root / "regression-osnet.json"
        record = suite.load(path)
        record["status"] = "pending"
        dump(path, record)
        self.failed_gate("Historical CSW regression did not pass")
        record["status"] = "passed"
        record["new_verified_sha256"] = "0" * 64
        dump(path, record)
        self.failed_gate("not bound to the completed rho=.5")

    def test_metrics_values_cannot_change_pair_selection(self):
        initial = [(j["method"], j["source_lambda"]) for j in self.plan()["jobs"]]
        for entry in self.complete["test"]:
            entry["metrics"] = {"mAP": 1000000 if entry["mode"] == "cb" else -1000000}
        self.save_complete()
        self.assertEqual(initial, [(j["method"], j["source_lambda"]) for j in self.plan()["jobs"]])

    def test_execution_rechecks_evidence_before_any_subprocess(self):
        plan = self.plan()
        self.complete["status"] = "pending"
        self.save_complete()
        with patch.object(suite.subprocess, "run") as run:
            with self.assertRaisesRegex(suite.GateError, "changed before execution"):
                suite.execute(plan)
            run.assert_not_called()

    def test_execution_rechecks_supporting_source_bytes(self):
        helper = self.source / "uniform_control.py"
        helper.write_text("# original synthetic helper\n")
        plan = self.plan()
        helper.write_text("# changed synthetic helper\n")
        with patch.object(suite.subprocess, "run") as run:
            with self.assertRaisesRegex(suite.GateError, "changed before execution"):
                suite.execute(plan)
            run.assert_not_called()

    def test_frozen_v2_protocol_bytes_are_bound(self):
        (self.root / "PROTOCOL.md").write_text("changed protocol\n")
        self.failed_gate("Frozen v2 protocol SHA mismatch")

    def test_execute_only_invokes_local_argv_in_required_sequence(self):
        plan = self.plan()
        with patch.object(suite.subprocess, "run") as run:
            result = suite.execute(plan)
        self.assertEqual(result["status"], "executed")
        self.assertEqual(run.call_count, 16)
        self.assertEqual([call.args[0] for call in run.call_args_list],
                         [command for job in plan["jobs"] for command in job["commands"]])
        self.assertTrue(all(call.kwargs["check"] for call in run.call_args_list))
        for job in plan["jobs"]:
            payload = Path(job["input_binding_file"]).read_bytes()
            self.assertEqual(payload, suite.canonical_bytes(job["input_binding"]))
            self.assertFalse(payload.endswith(b"\n"))
            self.assertEqual(suite.sha(job["input_binding_file"]), job["input_binding_sha256"])

    def test_replacing_self_consistent_mat_cache_support_is_rejected(self):
        self.make_inputs("resnet50", tag="replacement")
        cache = self.root / "cache/test-resnet50"
        manifest = suite.load(cache / "manifest.json")
        self.assertEqual(manifest["feature_sha256"], suite.sha(self.root / "features/test-resnet50.mat"))
        self.assertTrue(all(manifest["files"][name] == suite.sha(cache / name) for name in suite.CACHE_FILES[1:]))
        self.failed_gate("Current MAT SHA differs", "current_inputs_match_frozen_paired_v2_tests")

    def test_replacing_cache_with_new_self_consistent_inventory_is_rejected(self):
        cache = self.root / "cache/test-resnet50"
        for name in suite.CACHE_FILES[1:]:
            (cache / name).write_text("replacement cache " + name)
        dump(cache / "manifest.json", dict(feature_sha256=suite.sha(self.root / "features/test-resnet50.mat"),
             files={name: suite.sha(cache / name) for name in suite.CACHE_FILES[1:]}))
        self.failed_gate("Current cache SHA differs", "current_inputs_match_frozen_paired_v2_tests")

    def test_replacing_only_support_is_rejected(self):
        (self.root / "supports/test-resnet50-cb.npz").write_text("replacement support")
        self.failed_gate("Current support SHA differs", "current_inputs_match_frozen_paired_v2_tests")

    def test_swapping_another_models_complete_inputs_is_rejected(self):
        for name in suite.CACHE_FILES:
            target = self.root / "cache/test-resnet50" / name
            target.write_bytes((self.root / "cache/test-osnet" / name).read_bytes())
        (self.root / "features/test-resnet50.mat").write_bytes((self.root / "features/test-osnet.mat").read_bytes())
        for mode in suite.MODES:
            target = self.root / ("supports/test-resnet50-" + mode + ".npz")
            target.write_bytes((self.root / ("supports/test-osnet-" + mode + ".npz")).read_bytes())
        self.failed_gate("Current MAT SHA differs", "current_inputs_match_frozen_paired_v2_tests")

    def test_self_consistent_paired_report_cannot_bind_other_models_paths(self):
        source = suite.load(Path(self.entries[("osnet", "test", "ordinary", .5, .25)]["verified_path"]).parent / "run-manifest.json")["config"]
        target = self.entries[("resnet50", "test", "ordinary", .5, .25)]
        fields = ("feature_sha256", "support_sha256", "support_file", "cache_manifest_path", "inputs")
        self.replace_bound_test_manifest(target, lambda config: config.update({key: source[key] for key in fields}))
        self.failed_gate("not the fixed model/mode input", "current_inputs_match_frozen_paired_v2_tests")

    def test_paired_report_path_must_identify_correct_backbone(self):
        target = self.entries[("resnet50", "test", "ordinary", .5, .25)]
        source = self.entries[("osnet", "test", "ordinary", .5, .25)]
        target.update(verified_path=source["verified_path"], verified_sha256=source["verified_sha256"])
        self.save_complete()
        self.failed_gate("Report path does not identify its grid cell")

    def test_fixed_v2_solver_parameters_cannot_be_changed_with_rehashed_evidence(self):
        target = self.entries[("resnet50", "test", "ordinary", .5, .25)]
        self.replace_bound_test_manifest(target, lambda config: config.update(alpha=6))
        self.failed_gate("fixed v2 model parameters")

    def test_fixed_v2_solver_and_tail_cannot_be_changed_with_rehashed_evidence(self):
        target = self.entries[("resnet50", "test", "ordinary", .5, .25)]
        self.replace_bound_test_manifest(target, lambda config: config.update(numerical_tail=True))
        self.failed_gate("v2 solver/no-tail policy")
        self.replace_bound_test_manifest(target, lambda config: config.update(numerical_tail=False, numerical_solver="different"))
        self.failed_gate("v2 solver/no-tail policy")

    def test_existing_different_binding_is_not_overwritten_or_executed(self):
        plan = self.plan()
        target = Path(plan["jobs"][-1]["input_binding_file"])
        target.parent.mkdir(parents=True)
        target.write_text("existing different bytes\n")
        with patch.object(suite.subprocess, "run") as run:
            with self.assertRaisesRegex(suite.GateError, "refusing to overwrite"):
                suite.execute(plan)
            run.assert_not_called()
        self.assertEqual(target.read_text(), "existing different bytes\n")
        self.assertEqual(list(target.parent.iterdir()), [target])

    def test_identical_compact_bindings_can_be_reused(self):
        plan = self.plan()
        suite.write_input_bindings(plan)
        before = {job["input_binding_file"]: Path(job["input_binding_file"]).stat().st_mtime_ns for job in plan["jobs"]}
        suite.write_input_bindings(plan)
        self.assertEqual(before, {name: Path(name).stat().st_mtime_ns for name in before})

    def test_execution_rechecks_input_bytes_before_any_subprocess(self):
        plan = self.plan()
        self.make_inputs("osnet", tag="changed after plan")
        with patch.object(suite.subprocess, "run") as run:
            with self.assertRaisesRegex(suite.GateError, "changed before execution"):
                suite.execute(plan)
            run.assert_not_called()

    def test_real_runner_binding_validator_accepts_all_four_synthetic_formal_contracts(self):
        plan = self.plan()
        suite.write_input_bindings(plan)
        for job in plan["jobs"]:
            with self.subTest(model=job["model"], mode=job["source_support_mode"]):
                binding, extra = runner_binding.validate_input_binding(job["input_binding_file"], job["input_binding_sha256"],
                                                                      **self.binding_arguments(job))
                self.assertEqual(binding, job["input_binding"])
                self.assertEqual(len(extra), 3)

    def test_real_runner_binding_validator_rejects_each_tampered_cache_file(self):
        plan = self.plan()
        suite.write_input_bindings(plan)
        job = plan["jobs"][0]
        arguments = self.binding_arguments(job)
        for name in suite.CACHE_FILES:
            with self.subTest(cache_file=name):
                path = arguments["cache"] / name
                before = path.read_bytes()
                try:
                    path.write_bytes(before + b"tampered")
                    with self.assertRaisesRegex(ValueError, "Paired cache SHA mismatch"):
                        runner_binding.validate_input_binding(job["input_binding_file"], job["input_binding_sha256"], **arguments)
                finally:
                    path.write_bytes(before)

    def test_real_runner_binding_validator_rejects_wrong_model_argument(self):
        plan = self.plan()
        suite.write_input_bindings(plan)
        job = plan["jobs"][0]
        arguments = self.binding_arguments(job)
        arguments["model"] = "osnet"
        with self.assertRaisesRegex(ValueError, "model/mode/parameter/selection differs"):
            runner_binding.validate_input_binding(job["input_binding_file"], job["input_binding_sha256"], **arguments)

    def test_real_runner_binding_validator_rejects_rehashed_wrong_model_contract(self):
        job = self.plan()["jobs"][0]
        binding = dict(job["input_binding"], model="osnet")
        path = self.output / "forged-binding.json"
        path.write_bytes(suite.canonical_bytes(binding))
        arguments = self.binding_arguments(job)
        arguments["model"] = "osnet"
        with self.assertRaisesRegex(ValueError, "wrong model path"):
            runner_binding.validate_input_binding(path, suite.canonical(binding), **arguments)

    def test_output_cannot_overwrite_frozen_root(self):
        options = self.options()
        options["output_root"] = self.root / "new-runs"
        with self.assertRaisesRegex(suite.GateError, "separate from frozen v2"):
            suite.build_plan(**options)


if __name__ == "__main__":
    unittest.main(verbosity=2)
