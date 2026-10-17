"""Plan four frozen, paired uniform controls; execute only with explicit --execute.

This module uses v2 report status, dimensions, contracts and hashes only. It never
uses development or test metrics to choose a parameter. Missing completion
evidence produces a blocked plan, not an attempted run. Original absolute report
paths must be available on the execution host; this is not a relocation adapter.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys


MODELS = ("resnet50", "osnet")
MODES = ("ordinary", "cb")
RHOS = (0.25, 0.5, 0.75, 1.0)
LAMBDAS = (0.25, 0.5, 0.75)
DIMENSIONS = {"val": (929, 3376), "test": (3368, 19732)}
SCHEMA = "uniform-paired-suite-plan-v1"
INPUT_BINDING_SCHEMA = "uniform-frozen-input-binding-v1"
CACHE_FILES = ("manifest.json", "features.npy", "affinity.npy", "graph.npz")


class GateError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise GateError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def digest(value):
    require(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value),
            "Expected a lowercase SHA-256 digest")
    return value


def load(path):
    def reject_constant(value):
        raise GateError("Non-finite JSON constant: " + value)
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def canonical(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def method_name(mode, rho, lam):
    token = lambda value: str(float(value)).replace(".", "p")
    return ("csw-r" + token(rho) if mode is None else
            mode + "-r0p5-l" + token(lam))


def identity(entry):
    for field in ("rho", "lambda_"):
        require(type(entry.get(field)) in (int, float), "Invalid parameter " + field)
    return (entry["model"], entry["split"], entry["mode"], entry["rho"], entry["lambda_"])


def selection_contract(path, expected_sha):
    digest(expected_sha)
    require(sha(path) == expected_sha, "Explicit selection SHA does not match selection file")
    selected = load(path)
    digest(selected["protocol_sha256"])
    require(set(selected["models"]) == set(MODELS), "Selection must contain exactly both backbones")
    for model in MODELS:
        choices = selected["models"][model]
        require(set(choices) == {"rho", *MODES}, "Unexpected selection choices")
        for choice, entry in choices.items():
            mode = None if choice == "rho" else choice
            key = identity(entry)
            require(key[:3] == (model, "val", mode), "Selection entry identity mismatch")
            rho, lam = key[3:]
            require((mode is None and rho in RHOS and lam == 0.) or
                    (mode in MODES and rho == .5 and lam in LAMBDAS),
                    "Selection parameter outside the frozen candidate grid")
            require(entry["method"] == method_name(mode, rho, lam), "Selection method mismatch")
            digest(entry["verified_sha256"])
    return selected


def expected_grids(selected):
    development, test = set(), set()
    for model in MODELS:
        development.update((model, "val", None, rho, 0.) for rho in RHOS)
        development.update((model, "val", mode, .5, lam) for mode in MODES for lam in LAMBDAS)
        test.add((model, "test", None, .5, 0.))
        test.add((model, "test", None, selected["models"][model]["rho"]["rho"], 0.))
        test.update((model, "test", mode, .5, selected["models"][model][mode]["lambda_"])
                    for mode in MODES)
    require(len(development) == 20 and len(test) == 8,
            "Frozen selection must imply the required 20-development / 8-test grid")
    return development, test


def report_contract(root, entry, protocol_sha):
    model, split, mode, rho, lam = identity(entry)
    method = method_name(mode, rho, lam)
    require(entry["method"] == method, "Index method/parameter mismatch")
    expected_path = (root / "runs" / split / model / method / ("verified-" + method + ".json")).resolve()
    claimed_path = Path(entry["verified_path"])
    path = (claimed_path if claimed_path.is_absolute() else root / claimed_path).resolve()
    require(path == expected_path and path.is_relative_to(root), "Report path does not identify its grid cell")
    report_sha = digest(entry["verified_sha256"])
    require(sha(path) == report_sha, "Verified report SHA mismatch: " + str(path))
    report = load(path)
    require(report["status"] == "verified" and report["method"] == method, "Report is not verified for its method")
    nq, ng = DIMENSIONS[split]
    require((report["input"]["query_count"], report["input"]["gallery_count"]) == (nq, ng),
            "Verified input dimensions differ from the complete split")
    checks = report["checks"]
    require(all(checks[key] == nq for key in ("full_permutation_queries", "independent_metric_queries", "diffusion_checkpoints")),
            "Incomplete per-query verification")
    require(checks["no_diffusion"] is False and checks["diffusion_depth"] == 500, "Wrong diffusion coverage/depth")
    require(checks["required_max_relative_residual"] == 1e-8, "Residual threshold differs from frozen v2")
    for key in ("required_max_relative_residual", "max_relative_residual"):
        require(type(checks[key]) in (int, float) and math.isfinite(checks[key]) and 0 <= checks[key] <= 1e-8,
                "Invalid residual verification threshold or result")
    manifest_path = path.parent / "run-manifest.json"
    require(Path(report["manifest"]["path"]).resolve() == manifest_path,
            "Report binds a different run manifest")
    require(sha(manifest_path) == report["files"][str(manifest_path)]["sha256"], "Run manifest SHA mismatch")
    manifest = load(manifest_path)
    config = manifest["config"]
    require(canonical(config) == manifest["signature"] == report["manifest"]["signature"],
            "Run signature is not bound to the verified report")
    require((config["method"], config["rho"], config["lambda"], config["support_mode"]) == (method, rho, lam, mode),
            "Verified run parameters differ from the indexed grid cell")
    require((config["query_count"], config["gallery_count"], config["depth"]) == (nq, ng, 500),
            "Run contract dimensions/depth differ")
    require((config["alpha"], config["beta"], config["step"], config["k"], config["gamma"]) == (5, 25, 5, 20, .4),
            "Run contract differs from fixed v2 model parameters")
    require(config["numerical_solver"] == "same-float64-exact-rational-mp-v1" and
            config.get("numerical_tail", False) is False, "Run contract differs from the v2 solver/no-tail policy")
    require(config["protocol_sha256"] == protocol_sha, "Run protocol differs from frozen selection")
    require(config["feature_sha256"] == report["input_mat_sha256"], "Run feature/report binding differs")
    numerical_path = path.parent / "verified-numerical-evidence.json"
    numerical = load(numerical_path)
    require(numerical["status"] == "verified" and numerical["method"] == method and
            numerical["standard_report_sha256"] == report_sha and numerical["query_checkpoints"] == nq,
            "Numerical verification is not bound to the complete standard report")
    for filename, field in (("verify_numerics.py", "verifier_sha256"),
                            ("high_precision_recovery.py", "recovery_sha256")):
        claimed = [record["sha256"] for name, record in report["files"].items() if Path(name).name == filename]
        require(claimed == [numerical[field]], "Numerical verifier/source binding differs: " + filename)
    maximum = numerical["max_relative_residual"]
    require(type(maximum) in (int, float) and math.isfinite(maximum) and 0 <= maximum <= 1e-8,
            "Numerical verification residual exceeds the unchanged threshold")
    require(maximum <= checks["max_relative_residual"] + 1e-16,
            "Numerical verification residual is absent from the standard report")
    return dict(model=model, split=split, method=method, report_path=str(path), report_sha256=report_sha,
                numerical_path=str(numerical_path), numerical_sha256=sha(numerical_path),
                manifest_sha256=sha(manifest_path))


def completion_gate(root, selected, selection_sha):
    complete_path = root / "COMPLETE.json"
    complete = load(complete_path)
    require(complete["status"] == "verified", "v2 COMPLETE.status must be verified")
    require(complete["selection_sha256"] == selection_sha, "COMPLETE selection SHA differs from explicit frozen SHA")
    require(sha(root / "PROTOCOL.md") == selected["protocol_sha256"], "Frozen v2 protocol SHA mismatch")
    expected = expected_grids(selected)
    cells, reports = {}, []
    for section, grid in zip(("development", "test"), expected):
        entries = complete[section]
        require(isinstance(entries, list) and len(entries) == len(grid), "Wrong " + section + " configuration count")
        keys = [identity(entry) for entry in entries]
        require(len(set(keys)) == len(keys) and set(keys) == grid, "Incomplete, duplicate or unexpected " + section + " grid")
        for entry, key in zip(entries, keys):
            reports.append(report_contract(root, entry, selected["protocol_sha256"]))
            cells[key] = entry
    for model in MODELS:
        for entry in selected["models"][model].values():
            require(cells[identity(entry)]["verified_sha256"] == entry["verified_sha256"],
                    "Selected development report is not the completed grid report")
    regressions = []
    for model in MODELS:
        path = root / ("regression-" + model + ".json")
        record = load(path)
        require(record["status"] == "passed", "Historical CSW regression did not pass: " + model)
        require(record["new_verified_sha256"] == cells[(model, "test", None, .5, 0.)]["verified_sha256"],
                "Historical CSW regression is not bound to the completed rho=.5 test report")
        digest(record["old_metrics_sha256"])
        changed = record["different_rankings"]
        require(isinstance(changed, list) and all(type(q) is int and 0 <= q < 3368 for q in changed)
                and len(set(changed)) == len(changed), "Invalid historical ranking-difference inventory")
        regressions.append(dict(model=model, path=str(path), sha256=sha(path), status="passed"))
    return dict(status="passed", complete_sha256=sha(complete_path), development_configurations=20,
                test_configurations=8, reports=reports, historical_regressions=regressions)


def disjoint(first, second):
    return not (first.is_relative_to(second) or second.is_relative_to(first))


def frozen_input_binding(*, completed_reports, model, mode, lam, selection_sha,
                         mat, cache, support, hash_file):
    """Bind current files to this exact completed test grid cell, never by metrics."""
    paired = [record for record in completed_reports
              if (record["model"], record["split"], record["method"]) ==
                 (model, "test", method_name(mode, .5, lam))]
    require(len(paired) == 1, "Expected exactly one verified paired v2 test report")
    paired = paired[0]
    report_path = Path(paired["report_path"])
    manifest_path = report_path.parent / "run-manifest.json"
    require(sha(report_path) == paired["report_sha256"] and
            sha(manifest_path) == paired["manifest_sha256"], "Paired v2 report/manifest changed during input binding")
    manifest = load(manifest_path)
    config = manifest["config"]
    require(canonical(config) == manifest["signature"], "Paired input manifest signature mismatch")
    require((config["method"], config["rho"], config["lambda"], config["support_mode"]) ==
            (method_name(mode, .5, lam), .5, lam, mode), "Paired input manifest parameters differ")
    feature_sha = digest(config["feature_sha256"])
    support_sha = digest(config["support_sha256"])
    require(Path(config["support_file"]).resolve() == support.resolve(),
            "Paired support path is not the fixed model/mode input")
    require(Path(config["cache_manifest_path"]).resolve() == (cache / "manifest.json").resolve(),
            "Paired cache path is not the fixed model input")
    require(hash_file(mat) == feature_sha, "Current MAT SHA differs from the frozen paired v2 test input: " + model)
    require(hash_file(support) == support_sha,
            "Current support SHA differs from the frozen paired v2 test input: " + model + "/" + mode)
    cache_hashes = {}
    for name in CACHE_FILES:
        candidates = [(Path(path), value) for path, value in config["inputs"].items() if Path(path).name == name]
        require(len(candidates) == 1, "Missing/ambiguous paired cache input: " + name)
        claimed_path, expected_sha = candidates[0]
        require(claimed_path.resolve() == (cache / name).resolve(),
                "Paired cache file is not the fixed model input: " + name)
        expected_sha = digest(expected_sha)
        require(hash_file(cache / name) == expected_sha,
                "Current cache SHA differs from the frozen paired v2 test input: " + model + "/" + name)
        cache_hashes[name] = expected_sha
    return dict(schema=INPUT_BINDING_SCHEMA, scope="frozen-v2-test", model=model,
                source_support_mode=mode, source_lambda=lam, rho=.5, selection_sha256=selection_sha,
                feature_sha256=feature_sha, support_sha256=support_sha, cache_files=cache_hashes,
                paired_standard_report_path=str(report_path), paired_standard_report_sha256=paired["report_sha256"],
                paired_run_manifest_path=str(manifest_path), paired_run_manifest_sha256=paired["manifest_sha256"])


def build_plan(*, v2_root, output_root, selection_file, selection_sha256, protocol,
               source_dir=None, repo=None, python=None):
    root, output = Path(v2_root).resolve(), Path(output_root).resolve()
    source = Path(source_dir or Path(__file__).parent).resolve()
    repository = Path(repo or root / "repo").resolve()
    selection_file, protocol = Path(selection_file).resolve(), Path(protocol).resolve()
    require(disjoint(root, output), "New output root must be separate from frozen v2")
    require(disjoint(repository, output), "New output root must be separate from the repository")
    require(protocol.is_relative_to(output), "New protocol must be inside the new output root")
    selected = selection_contract(selection_file, selection_sha256)
    gates = []
    try:
        gates.append(dict(name="v2_complete_grid_and_bound_evidence", **completion_gate(root, selected, selection_sha256)))
    except (OSError, ValueError, KeyError, TypeError) as error:
        gates.append(dict(name="v2_complete_grid_and_bound_evidence", status="blocked", reason=str(error)))
    required_files = {protocol, source / "run_uniform_candidate.py", source / "verify_uniform_control.py",
                      source / "verify_experiment.py", source / "verify_numerics.py"}
    jobs, input_checks, input_hashes = [], [], {}

    def input_sha(path):
        # ordinary/CB share large files; read each physical file once per audit.
        resolved = Path(path).resolve()
        if resolved not in input_hashes:
            input_hashes[resolved] = sha(resolved)
        return input_hashes[resolved]

    for model in MODELS:
        for mode in MODES:
            lam = selected["models"][model][mode]["lambda_"]
            method = "uniform-mass-" + mode
            out = output / "runs" / "test" / model / method
            mat = root / "features" / ("test-" + model + ".mat")
            cache = root / "cache" / ("test-" + model)
            support = root / "supports" / ("test-" + model + "-" + mode + ".npz")
            required_files.update((mat, support, cache / "manifest.json", cache / "features.npy", cache / "affinity.npy", cache / "graph.npz"))
            binding_file = output / "input-bindings" / (model + "-" + mode + ".json")
            binding, binding_sha = None, None
            try:
                require(gates[0]["status"] == "passed", "Completed v2 evidence gate has not passed")
                binding = frozen_input_binding(completed_reports=gates[0]["reports"], model=model, mode=mode,
                    lam=lam, selection_sha=selection_sha256, mat=mat, cache=cache, support=support, hash_file=input_sha)
                binding_sha = canonical(binding)
                input_checks.append(dict(model=model, source_support_mode=mode, status="passed", input_binding_sha256=binding_sha))
            except (OSError, ValueError, KeyError, TypeError) as error:
                input_checks.append(dict(model=model, source_support_mode=mode, status="blocked", reason=str(error)))
            prefix = [str(python or sys.executable), "-B"]
            common = ["--output-dir", str(out), "--method", method]
            runner = prefix + [str(source / "run_uniform_candidate.py"), "--repo", str(repository),
                "--mat-path", str(mat), "--cache-dir", str(cache), *common, "--protocol", str(protocol),
                "--rho", "0.5", "--source-lambda", str(lam), "--support-mode", mode,
                "--support-file", str(support), "--selection-file", str(selection_file),
                "--selection-sha256", selection_sha256, "--model", model, "--workers", "8",
                "--expected-queries", "3368", "--expected-gallery", "19732",
                "--input-binding-file", str(binding_file), "--input-binding-sha256",
                binding_sha or "BLOCKED_MISSING_FROZEN_INPUT_BINDING"]
            commands = [runner,
                prefix + [str(source / "verify_experiment.py"), "--mat-path", str(mat), *common,
                          "--expected-queries", "3368", "--expected-gallery", "19732"],
                prefix + [str(source / "verify_numerics.py"), *common],
                prefix + [str(source / "verify_uniform_control.py"), *common]]
            jobs.append(dict(model=model, split="test", method=method, source_support_mode=mode,
                             rho=.5, source_lambda=lam, output_dir=str(out), commands=commands,
                             input_binding=binding, input_binding_file=str(binding_file), input_binding_sha256=binding_sha,
                             paired_development_report_sha256=selected["models"][model][mode]["verified_sha256"]))
    missing = sorted(str(path) for path in required_files if not path.is_file())
    if not repository.is_dir():
        missing.append(str(repository))
    gates.append(dict(name="local_runner_and_inputs_present", status="blocked" if missing else "passed", missing=missing))
    gates.append(dict(name="current_inputs_match_frozen_paired_v2_tests",
                      status="passed" if all(item["status"] == "passed" for item in input_checks) else "blocked", jobs=input_checks))
    return dict(schema=SCHEMA, status="planned", execution_requested=False, v2_root=str(root), output_root=str(output),
                selection_file=str(selection_file), selection_sha256=selection_sha256,
                protocol=str(protocol), protocol_sha256=sha(protocol) if protocol.is_file() else None,
                source_files={str(path): sha(path) for path in sorted(source.glob("*.py")) if path.is_file()},
                source_dir=str(source), repo=str(repository), python=str(python or sys.executable),
                parameter_rule="Only frozen selection.models[model][support_mode].lambda_; rho fixed at 0.5",
                test_metric_policy="No metric/per-query values are accessed or used for selection",
                execution_ready=all(gate["status"] == "passed" for gate in gates), gates=gates, jobs=jobs)


def write_input_bindings(plan):
    """Persist exact compact bytes once; never replace a different contract."""
    output = Path(plan["output_root"])
    directory = output / "input-bindings"
    require(directory.resolve().is_relative_to(output), "Input binding directory escaped the new output root")
    pending = []
    for job in plan["jobs"]:
        path = Path(job["input_binding_file"])
        require(path == directory / (job["model"] + "-" + job["source_support_mode"] + ".json"),
                "Input binding destination differs from the paired job")
        payload = canonical_bytes(job["input_binding"])
        require(job["input_binding"] is not None and hashlib.sha256(payload).hexdigest() == job["input_binding_sha256"],
                "Input binding payload SHA mismatch")
        require(not path.is_symlink(), "Input binding file cannot be a symlink")
        if path.exists():
            require(path.read_bytes() == payload, "Existing input binding differs; refusing to overwrite: " + str(path))
        else:
            pending.append((path, payload))
    directory.mkdir(parents=True, exist_ok=True)
    for path, payload in pending:
        with path.open("xb") as stream:
            stream.write(payload)


def execute(plan):
    """Local, sequential subprocesses only; callers must explicitly opt in."""
    require(plan["execution_ready"], "Execution blocked: prerequisite evidence or local inputs are missing/invalid")
    options = {key: plan[key] for key in ("v2_root", "output_root", "selection_file", "selection_sha256", "protocol", "source_dir", "repo", "python")}
    fresh = build_plan(**options)
    require(fresh == plan, "Plan inputs or evidence changed before execution")
    output = Path(plan["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output / ".uniform-suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_input_bindings(plan)
        for job in plan["jobs"]:
            require(build_plan(**options) == plan, "Frozen evidence changed between paired runs")
            folder = Path(job["output_dir"])
            require(folder.resolve().is_relative_to(output), "Run output escaped the new output root")
            folder.mkdir(parents=True, exist_ok=True)
            for index, command in enumerate(job["commands"]):
                with (folder / ("suite-stage-" + str(index) + ".log")).open("a") as stream:
                    stream.write("COMMAND " + json.dumps(command) + "\n")
                    stream.flush()
                    subprocess.run(command, cwd=output, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return dict(schema=SCHEMA, status="executed", execution_requested=True, selection_sha256=plan["selection_sha256"],
                completed_command_groups=len(plan["jobs"]), scope="All runner and verifier commands exited successfully; retain their independent reports")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("v2-root", "output-root", "selection-file", "protocol"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true", help="Explicitly run locally after all evidence gates pass")
    args = vars(parser.parse_args(argv))
    requested = args.pop("execute")
    plan = build_plan(**args)
    print(json.dumps(execute(plan) if requested else plan, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GateError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(json.dumps(dict(schema=SCHEMA, status="blocked", error=str(error))), file=sys.stderr)
        raise SystemExit(2)
