#!/usr/bin/env python3
"""Independently audit full ReID rankings and diffusion evidence; never run a model.

Default coverage is the complete Market1501 test split (3368 x 19732).
For a complete development split pass its exact expected counts explicitly.
Only NumPy and SciPy are required; no runner/common/ReIDEval imports are used.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import uuid

import numpy as np
from scipy.io import loadmat


TOL = 1e-12
RESIDUAL_TOL = 1e-8
RANKS = (1, 5, 10, 20, 50, 100, 500)
LABEL_KEYS = ('query_label', 'query_cam', 'gallery_label', 'gallery_cam')
HASH_RE = re.compile(r'^[a-fA-F0-9]{64}$')
METHOD_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value, compact=True):
    kwargs = dict(sort_keys=True, allow_nan=False)
    if compact:
        kwargs['separators'] = (',', ':')
    return hashlib.sha256(json.dumps(value, **kwargs).encode()).hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f'Duplicate JSON key {key!r}: {path}')
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f'Non-finite JSON constant {value}: {path}')

    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def integer(value, context):
    require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)),
            f'Expected integer: {context}')
    return int(value)


def close(actual, expected, context):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(a.shape == b.shape, f'Metric shape mismatch: {context}: {a.shape} != {b.shape}')
    require(np.isfinite(a).all() and np.isfinite(b).all(), f'Non-finite metric: {context}')
    require(np.allclose(a, b, atol=TOL, rtol=0), f'Metric mismatch: {context}')


class Inventory:
    """Hash each evidence/source file once, then detect mutation during the audit."""
    def __init__(self):
        self.files = {}
        self.stats = {}

    def add(self, path, role, expected=None):
        path = Path(path).resolve()
        require(path.is_file(), f'Missing {role}: {path}')
        key = str(path)
        if key not in self.files:
            before = path.stat()
            digest = sha256(path)
            after = path.stat()
            stamp = lambda s: (s.st_size, s.st_mtime_ns, s.st_ino)
            require(stamp(before) == stamp(after), f'File changed while hashing: {path}')
            self.stats[key] = stamp(after)
            self.files[key] = dict(bytes=after.st_size, sha256=digest, roles=[])
        if role not in self.files[key]['roles']:
            self.files[key]['roles'].append(role)
        if expected is not None:
            require(isinstance(expected, str) and HASH_RE.fullmatch(expected),
                    f'Invalid expected SHA-256 for {path}')
            require(self.files[key]['sha256'] == expected.lower(), f'SHA-256 mismatch: {path}')
        return self.files[key]['sha256']

    def unchanged(self):
        for name, expected in self.stats.items():
            stat = Path(name).stat()
            require((stat.st_size, stat.st_mtime_ns, stat.st_ino) == expected,
                    f'Evidence changed during verification: {name}')


def walk(value, trail=()):
    yield trail, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk(child, trail + (str(key),))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from walk(child, trail + (str(i),))


def find_manifest(args):
    if args.manifest:
        return args.manifest.resolve()
    roots = [args.output_dir.resolve(), *list(args.output_dir.resolve().parents)[:4]]
    for root in roots:
        path = root / 'run-manifest.json'
        if path.is_file():
            return path
    raise ValueError('run-manifest.json not found; pass --manifest explicitly')


def source_claims(container):
    """Recognize path->hash and {path, sha256} lists/maps without a fixed file set."""
    if isinstance(container, dict):
        digest = container.get('sha256')
        if 'path' in container and isinstance(digest, str):
            yield str(container['path']), digest
            return
        for name, value in container.items():
            if isinstance(value, str) and HASH_RE.fullmatch(value):
                yield name, value
            elif isinstance(value, dict) and isinstance(value.get('sha256'), str):
                yield str(value.get('path', name)), value['sha256']
            elif isinstance(value, (dict, list)):
                yield from source_claims(value)
    elif isinstance(container, list):
        for value in container:
            yield from source_claims(value)


def resolve_source(name, args, manifest_path, manifest):
    path = Path(name)
    candidates = [path] if path.is_absolute() else []
    for mapping in args.path_map:
        old, separator, new = mapping.partition('=')
        require(separator and old and new, '--path-map must be OLD_PREFIX=NEW_PREFIX')
        try:
            candidates.append(Path(new) / path.relative_to(Path(old)))
        except ValueError:
            pass
    roots = [*args.source_root, manifest_path.parent, Path(__file__).resolve().parent, Path.cwd()]
    for trail, value in walk(manifest):
        if trail and trail[-1] in ('source_root', 'repo_root', 'repository_root', 'repo_path') and isinstance(value, str):
            roots.append(Path(value))
    if not path.is_absolute():
        candidates.extend(root / path for root in roots)
    existing = {candidate.resolve() for candidate in candidates if candidate.is_file()}
    require(existing, f'Source unavailable: {name}; pass --source-root or --path-map')
    # Multiple roots may contain the same content; reject ambiguity in content below.
    return sorted(existing)


def inspect_manifest(args, inventory, mat_hash, nq, ng):
    path = find_manifest(args)
    inventory.add(path, 'run_manifest')
    manifest = read_json(path)
    require(isinstance(manifest, dict), 'Manifest must be a JSON object')
    config = manifest.get('config', manifest)
    require(isinstance(config, dict), 'Manifest config must be an object')
    signature = manifest.get('signature')
    signature_check = 'not provided; manifest file hash recorded'
    if signature is not None:
        require(isinstance(signature, str) and HASH_RE.fullmatch(signature), 'Invalid manifest signature')
        require('config' in manifest, 'A signed manifest must include its config')
        matches = [label for label, compact in [('sorted-json-compact', True), ('sorted-json-default', False)]
                   if canonical_hash(config, compact) == signature.lower()]
        require(matches, 'Manifest config signature mismatch')
        signature_check = matches[0]
    # Interpret shared protocol fields only; do not hard-code method/feature lists or parameters.
    for key, expected in [('queries', nq), ('query_count', nq), ('gallery', ng), ('gallery_count', ng)]:
        if key in config and isinstance(config[key], (int, float)):
            require(config[key] == expected, f'Manifest coverage mismatch: {key}')
    if 'query_ids' in config:
        require(config['query_ids'] == list(range(nq)), 'Manifest query_ids omit/reorder queries')
    if not args.no_diffusion:
        if 'depth' in config:
            require(config['depth'] == args.depth, 'Manifest diffusion depth differs')
        if 'numerical_tail' in config:
            require(config['numerical_tail'] is False, 'Manifest enables numerical tail')
    if isinstance(config.get('methods'), list):
        require(args.method in config['methods'], 'Method not declared in manifest')
    if 'method' in config:
        require(config['method'] == args.method, 'Method differs from manifest')
    matched_features = []
    for trail, value in walk(manifest):
        if not isinstance(value, str) or value.lower() != mat_hash:
            continue
        context = '/'.join(trail).lower()
        if any(token in context for token in ('feature', 'mat', 'input')):
            matched_features.append('/'.join(trail))
    require(matched_features, 'MAT SHA-256 is not bound by manifest feature/mat/input provenance')
    checked_inputs = {}
    # The experiment document and a reliability support file are data inputs,
    # not Python sources. Bind the actual bytes rather than just echoing a hash.
    for field, option, path_fields in (
            ('protocol_sha256', args.protocol, ('protocol_path', 'protocol_file')),
            ('support_sha256', args.support_file, ('support_file', 'support_path'))):
        expected = config.get(field)
        if expected is None:
            require(option is None, f'Explicit input is not bound by manifest: {field}')
            continue
        require(isinstance(expected, str) and HASH_RE.fullmatch(expected), f'Invalid {field}')
        names = ([str(option)] if option else []) + [config[k] for k in path_fields
                                                    if isinstance(config.get(k), str)]
        if field == 'protocol_sha256' and not names:
            roots = [path.parent, *list(path.parents)[:4], Path(__file__).resolve().parent.parent]
            names = [str(root/'PROTOCOL.md') for root in roots if (root/'PROTOCOL.md').is_file()]
        candidates = []
        for name in names:
            candidates.extend(resolve_source(name, args, path, manifest))
        matching = [candidate for candidate in candidates if sha256(candidate) == expected.lower()]
        require(matching, f'Cannot verify {field}; provide --protocol/--support-file or a manifest path')
        inventory.add(matching[0], 'declared_input', expected)
        checked_inputs[field] = dict(resolved_path=str(matching[0]), sha256=expected.lower())
    for trail, value in walk(manifest):
        if not trail or trail[-1] not in ('inputs', 'input_files', 'input_hashes', 'input_fingerprints'):
            continue
        for name, expected in source_claims(value):
            candidates = resolve_source(name, args, path, manifest)
            matching = [candidate for candidate in candidates if sha256(candidate) == expected.lower()]
            require(matching, f'Declared input SHA-256 has no matching file: {name}')
            inventory.add(matching[0], 'declared_input', expected)
            checked_inputs[name] = dict(resolved_path=str(matching[0]), sha256=expected.lower())
    checked_sources = {}
    for trail, value in walk(manifest):
        if not trail or trail[-1] not in ('sources', 'source_files', 'source_hashes', 'source_fingerprints'):
            continue
        for name, expected in source_claims(value):
            candidates = resolve_source(name, args, path, manifest)
            valid = [candidate for candidate in candidates if sha256(candidate) == expected.lower()]
            require(valid, f'Declared source SHA-256 has no matching file: {name}')
            chosen = valid[0]
            inventory.add(chosen, 'declared_source', expected)
            if name in checked_sources:
                require(checked_sources[name]['sha256'] == expected.lower(), 'Conflicting source hash declarations')
            checked_sources[name] = dict(resolved_path=str(chosen), sha256=expected.lower())
    require(checked_sources, 'Manifest has no recognized, verifiable source hash declarations')
    return manifest, dict(path=str(path), signature=signature, signature_validation=signature_check,
                          feature_hash_claims=matched_features, checked_inputs=checked_inputs,
                          checked_sources=checked_sources)


def load_input(path, expected_queries, expected_gallery):
    data = loadmat(path, variable_names=['query_f', 'gallery_f', *LABEL_KEYS])
    require(all(key in data for key in ('query_f', 'gallery_f', *LABEL_KEYS)), 'MAT lacks required arrays')
    qf, gf = np.asarray(data['query_f']), np.asarray(data['gallery_f'])
    require(qf.ndim == gf.ndim == 2 and qf.shape[1] == gf.shape[1] and qf.shape[1] > 0,
            'Invalid feature matrix dimensions')
    nq, ng = len(qf), len(gf)
    require((nq, ng) == (expected_queries, expected_gallery),
            f'Expected {expected_queries} queries x {expected_gallery} gallery, got {nq} x {ng}')
    for feature in (qf, gf):
        require(feature.dtype.kind in 'fiu' and np.isfinite(feature).all(), 'Non-finite/non-real features')
        require(np.all(np.linalg.norm(feature, axis=1) > 0), 'Zero feature vector')
    labels = []
    for key, count in zip(LABEL_KEYS, (nq, nq, ng, ng)):
        values = np.asarray(data[key]).ravel()
        require(values.dtype.kind in 'fiu' and values.shape == (count,) and np.isfinite(values).all(),
                f'Invalid label/camera array: {key}')
        require(np.all(values == np.floor(values)), f'Noninteger IDs: {key}')
        labels.append(values.astype(np.int64))
    return labels, dict(query_count=nq, gallery_count=ng, feature_dimension=int(qf.shape[1]),
                        query_norm_min=float(np.linalg.norm(qf, axis=1).min()),
                        query_norm_max=float(np.linalg.norm(qf, axis=1).max()),
                        gallery_norm_min=float(np.linalg.norm(gf, axis=1).min()),
                        gallery_norm_max=float(np.linalg.norm(gf, axis=1).max()))


def evaluate_query(global_order, query, labels, max_rank):
    qp, qc, gp, gc = labels
    nq, ng = len(qp), len(gp)
    require(global_order.shape == (ng,) and global_order.dtype.kind in 'iu', 'Ranking shape/dtype invalid')
    # Casting after range checks prevents unsigned-wrap surprises.
    require(np.all(global_order >= nq) and np.all(global_order < nq + ng), 'Ranking uses wrong global offset/range')
    order = global_order.astype(np.int64) - nq
    require(np.array_equal(np.sort(order), np.arange(ng)), f'Not a full gallery permutation: query {query}')
    keep = (gp[order] != -1) & ~((gp[order] == qp[query]) & (gc[order] == qc[query]))
    valid_order = order[keep]
    hits = np.flatnonzero(gp[valid_order] == qp[query]) + 1
    expected_positive_count = int(np.count_nonzero((gp == qp[query]) & (gc != qc[query]) & (gp != -1)))
    require(expected_positive_count > 0, f'Query {query} has no eligible positive; full-protocol denominator cannot be silently reduced')
    require(len(hits) == expected_positive_count, f'Missing eligible positives: query {query}')
    precision_at_hits = np.arange(1, len(hits) + 1, dtype=float) / hits
    first, last = int(hits[0]), int(hits[-1])
    row = dict(query_id=query, pid=int(qp[query]), camid=int(qc[query]), valid=True,
               AP=float(precision_at_hits.mean()), INP=float(len(hits) / last),
               positive_count=len(hits), retrieved_positive_count=len(hits),
               first_positive_rank=first, last_positive_rank=last,
               rank1=float(first == 1), filtered_gallery_count=len(valid_order))
    row.update({f'rank-{k}': float(first <= k) for k in RANKS if k <= max_rank})
    return row, (np.arange(1, max_rank + 1) >= first).astype(np.int64)


def metric_object(value):
    require(isinstance(value, dict), 'Metrics JSON must be an object')
    result = value.get('metrics', value)
    require(isinstance(result, dict), 'metrics wrapper must contain an object')
    return result


def compare_metrics(actual, saved, rows, context):
    saved = metric_object(saved)
    for key in ('mAP', 'mINP'):
        require(key in saved, f'Missing {key}: {context}')
        close(actual[key], saved[key], (context, key))
    require(integer(saved.get('valid_queries'), f'{context}/valid_queries') == len(rows), 'Wrong query denominator')
    curves = [saved[k] for k in ('cmc', 'CMC') if k in saved]
    require(curves, f'Missing CMC: {context}')
    for curve in curves:
        curve = np.asarray(curve, dtype=float)
        require(curve.ndim == 1 and min(50, len(actual['CMC'])) <= len(curve) <= len(actual['CMC']),
                f'Wrong CMC shape/coverage: {context}')
        close(np.asarray(actual['CMC'])[:len(curve)], curve, (context, 'CMC'))
    for k in RANKS:
        field = f'rank-{k}'
        if field in saved:
            require(k <= len(actual['CMC']), f'Unsupported saved rank {k}')
            close(actual['CMC'][k - 1], saved[field], (context, field))
        elif k in (1, 5, 10, 20) and k <= len(actual['CMC']):
            raise ValueError(f'Missing existing-evaluator field {field}: {context}')
    for key in ('query_count', 'gallery_count'):
        if key in saved:
            require(integer(saved[key], key) == actual[key], f'Wrong {key}: {context}')
    if 'per_query' in saved:
        per = saved['per_query']
        require(isinstance(per, list) and len(per) == len(rows), f'Per-query coverage mismatch: {context}')
        index = {}
        for record in per:
            require(isinstance(record, dict), 'Invalid per-query metric record')
            q = integer(record.get('query_id', record.get('query')), 'per-query ID')
            require(q not in index, f'Duplicate per-query ID: {context}/{q}')
            index[q] = record
        require(set(index) == {r['query_id'] for r in rows}, f'Per-query IDs differ: {context}')
        for row in rows:
            record = index[row['query_id']]
            require(record.get('valid') is True, 'Saved query unexpectedly invalid')
            for field in ('AP', 'INP', 'rank1', 'positive_count', 'retrieved_positive_count'):
                require(field in record, f'Missing per-query field: {field}')
                close(row[field], record[field], (context, row['query_id'], field))
            for field in ('pid', 'camid', 'first_positive_rank', 'last_positive_rank'):
                if field in record:
                    close(row[field], record[field], (context, row['query_id'], field))


def checkpoint_hash(record, field):
    candidates = [record[field]] if field in record else []
    if isinstance(record.get('files'), dict) and field in record['files']:
        candidates.append(record['files'][field])
    require(candidates and all(v == candidates[0] for v in candidates), f'Missing/conflicting {field}')
    return candidates[0]


def verify_checkpoint(args, query, ranking, labels, row, cmc, manifest_info, inventory):
    folder = args.output_dir / args.method
    record_path = folder / f'{query:04d}.json'
    rp, pp = folder / f'{query:04d}.rankings.npy', folder / f'{query:04d}.prefix.npy'
    inventory.add(record_path, 'diffusion_diagnostic')
    record = read_json(record_path)
    require(integer(record.get('query', record.get('query_id')), 'checkpoint query') == query, 'Checkpoint query differs')
    require(record.get('mode', record.get('method')) == args.method, 'Checkpoint method differs')
    if manifest_info['signature'] is not None:
        require(record.get('signature') == manifest_info['signature'], 'Checkpoint config signature differs')
    inventory.add(rp, 'checkpoint_ranking', checkpoint_hash(record, 'rankings_sha256'))
    inventory.add(pp, 'diffusion_prefix', checkpoint_hash(record, 'prefix_sha256'))
    checkpoint = np.load(rp, allow_pickle=False)
    require(checkpoint.dtype.kind in 'iu' and checkpoint.shape == ranking.shape and np.array_equal(checkpoint, ranking),
            f'Checkpoint and aggregate ranking differ: {query}')
    prefix = np.load(pp, allow_pickle=False)
    nq, ng = len(labels[0]), len(labels[2])
    require(prefix.dtype.kind in 'iu' and prefix.shape == (args.depth,), 'Prefix shape/dtype invalid')
    require(int(prefix[0]) == query and len(np.unique(prefix)) == args.depth,
            f'Prefix must start with query and contain unique global IDs: {query}')
    require(np.all(prefix >= 0) and np.all(prefix < nq + ng), 'Prefix global ID out of range')
    gallery_prefix = prefix[prefix >= nq]
    require(np.array_equal(ranking[:len(gallery_prefix)], gallery_prefix), 'Ranking does not preserve diffusion prefix')
    diag = record.get('diagnostics')
    require(isinstance(diag, dict) and diag.get('status') == 'complete_diffusion', 'Diffusion did not complete')
    require(integer(diag.get('diffusion_nodes'), 'diffusion_nodes') == args.depth, 'Wrong diffusion node count')
    require(integer(diag.get('numerical_tail_count'), 'numerical_tail_count') == 0, 'Numerical tail used')
    for value in (record.get('numerical_tail', False), diag.get('numerical_tail', False)):
        require(value is False, 'Numerical tail enabled')
    residual = float(diag.get('max_relative_residual', float('nan')))
    require(math.isfinite(residual) and 0 <= residual <= RESIDUAL_TOL, 'Non-finite/excessive diagnostic residual')
    trace = record.get('trace')
    require(isinstance(trace, list) and (trace or args.depth == 1), 'Missing diffusion trace')
    selected, trace_residuals = [query], []
    for step in trace:
        require(isinstance(step, dict) and isinstance(step.get('selected'), list), 'Malformed trace step')
        require(step['selected'], 'Empty trace selection')
        selected.extend(integer(node, 'trace selected node') for node in step['selected'])
        value = float(step.get('residual', float('nan')))
        require(math.isfinite(value) and 0 <= value <= RESIDUAL_TOL, 'Non-finite/excessive trace residual')
        trace_residuals.append(value)
    require(np.array_equal(np.asarray(selected), prefix), 'Trace selections do not equal saved prefix')
    if 'iterations' in diag:
        require(integer(diag['iterations'], 'iterations') == len(trace), 'Trace iteration count differs')
    if trace_residuals:
        require(max(trace_residuals) <= residual + 1e-15, 'Diagnostics understate maximum trace residual')
    if 'metrics' in record:
        actual = dict(mAP=row['AP'], mINP=row['INP'], CMC=cmc.astype(float).tolist(),
                      query_count=1, gallery_count=ng)
        compare_metrics(actual, record['metrics'], [row], f'query {query}')
    return residual


def verify(args):
    require(METHOD_RE.fullmatch(args.method) and args.method not in ('.', '..'), 'Unsafe method name')
    args.output_dir = args.output_dir.resolve()
    args.mat_path = args.mat_path.resolve()
    require(args.output_dir.is_dir(), 'Output directory does not exist')
    require(args.expected_queries > 0 and args.expected_gallery > 0 and 1 <= args.depth,
            'Expected counts/depth must be positive')
    inventory = Inventory()
    mat_hash = inventory.add(args.mat_path, 'input_feature_mat')
    labels, input_info = load_input(args.mat_path, args.expected_queries, args.expected_gallery)
    nq, ng = input_info['query_count'], input_info['gallery_count']
    manifest, manifest_info = inspect_manifest(args, inventory, mat_hash, nq, ng)
    ranking_path = args.output_dir / f'{args.method}.rankings.npy'
    metric_path = args.output_dir / f'{args.method}.json'
    inventory.add(ranking_path, 'aggregate_rankings')
    inventory.add(metric_path, 'runner_metrics')
    complete_path = args.output_dir/'COMPUTE_COMPLETE.json'
    if complete_path.is_file():
        inventory.add(complete_path, 'compute_completion_marker')
        completion = read_json(complete_path)
        require(completion.get('status') == 'computed', 'Compute completion status differs')
        if manifest_info['signature'] is not None:
            require(completion.get('signature') == manifest_info['signature'], 'Compute completion signature differs')
        require(completion.get('query_count') == nq and completion.get('gallery_count') == ng,
                'Compute completion coverage differs')
        inventory.add(ranking_path, 'completion_declared_rankings', completion.get('rankings_sha256'))
        require('rankings_sha256' in completion, 'Compute completion lacks ranking hash')
    rankings = np.load(ranking_path, mmap_mode='r', allow_pickle=False)
    require(rankings.dtype.kind in 'iu' and rankings.shape == (nq, ng), 'Ranking matrix must cover every query and gallery')
    saved = read_json(metric_path)
    if isinstance(saved, dict):
        for key in ('feature_sha256', 'mat_sha256'):
            if key in saved:
                require(saved[key] == mat_hash, 'Metrics bind a different feature MAT')
        if 'rankings_sha256' in saved:
            inventory.add(ranking_path, 'metrics_declared_rankings', saved['rankings_sha256'])
        if 'query_ids' in saved:
            require(saved['query_ids'] == list(range(nq)), 'Metric query_ids omit/reorder queries')
    if not args.no_diffusion:
        folder = args.output_dir / args.method
        require(folder.is_dir(), 'Missing diffusion checkpoint directory')
        # Catch a stale aggregate, omitted query, or leftover error checkpoint.
        expected_names = {f'{q:04d}{suffix}' for q in range(nq)
                          for suffix in ('.json', '.prefix.npy', '.rankings.npy')}
        actual_names = {p.name for p in folder.iterdir() if p.is_file() and
                        (p.suffix == '.json' or p.name.endswith(('.prefix.npy', '.rankings.npy')))}
        require(actual_names == expected_names, f'Checkpoint coverage differs: missing={len(expected_names-actual_names)}, extra={len(actual_names-expected_names)}')
    max_rank = min(500, ng)
    rows, cmc_sum, residuals = [], np.zeros(max_rank, dtype=np.int64), []
    for query in range(nq):
        row, cmc = evaluate_query(rankings[query], query, labels, max_rank)
        rows.append(row)
        cmc_sum += cmc
        if not args.no_diffusion:
            residuals.append(verify_checkpoint(args, query, rankings[query], labels, row, cmc,
                                              manifest_info, inventory))
        if args.progress_every and (query + 1) % args.progress_every == 0:
            print(json.dumps(dict(stage='verifying', method=args.method, queries=query+1, total=nq)), flush=True)
    metrics = dict(mAP=float(np.mean([row['AP'] for row in rows])),
                   mINP=float(np.mean([row['INP'] for row in rows])), CMC=(cmc_sum/nq).tolist(),
                   valid_queries=nq, query_count=nq, gallery_count=ng)
    metrics.update({f'rank-{k}': metrics['CMC'][k-1] for k in RANKS if k <= max_rank})
    compare_metrics(metrics, saved, rows, args.method)
    verifier_hash = inventory.add(Path(__file__), 'independent_verifier_source')
    protocol = dict(gallery_ids='global indices [query_count, query_count+gallery_count)',
                    coverage='all MAT queries; every gallery exactly once before junk removal',
                    junk='gallery pid=-1 OR same pid AND same camera as query',
                    AP='mean precision at every eligible positive rank; full gallery denominator',
                    INP='eligible positive count / last eligible positive rank',
                    CMC='first eligible positive rank <= k, averaged over all queries',
                    invalid_queries='rejected; no denominator reduction', max_rank=max_rank,
                    metric_units='fractions', tolerance=TOL)
    inventory.unchanged()
    return dict(schema_version=1, status='verified', method=args.method,
                checked_utc=datetime.now(timezone.utc).isoformat(), input=input_info,
                input_mat_sha256=mat_hash, evaluator_protocol=protocol,
                evaluator_protocol_sha256=canonical_hash(protocol), manifest=manifest_info,
                verifier_sha256=verifier_hash, metrics=metrics, per_query=rows,
                checks=dict(full_permutation_queries=nq, independent_metric_queries=nq,
                            diffusion_checkpoints=0 if args.no_diffusion else nq,
                            no_diffusion=args.no_diffusion,
                            diffusion_depth=None if args.no_diffusion else args.depth,
                            required_max_relative_residual=None if args.no_diffusion else RESIDUAL_TOL,
                            max_relative_residual=None if args.no_diffusion else max(residuals)),
                fingerprint_scope='All consumed evidence, all declared sources, and verifier; declared source hashes checked. Report SHA-256 is in adjacent .sha256 sidecar.',
                files=inventory.files,
                limitations=['Read-only audit of saved evidence, not rerun of feature extraction/diffusion.',
                             'Hash agreement proves byte identity, not scientific validity or training/test separation.',
                             'Original-similarity completion beyond diffusion prefix is not classified as a numerical fallback.',
                             'Cross-run baseline reuse additionally requires matching feature/extraction and experiment protocol provenance.'])


def write_report(output_dir, method, report):
    """Atomically publish a new report, using a versioned name if history exists."""
    root = Path(output_dir)
    target = root / f'verified-{method}.json'
    if target.exists():
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        target = root / f'verified-{method}.{stamp}-{uuid.uuid4().hex[:8]}.json'
    encoded = (json.dumps(report, indent=2, allow_nan=False) + '\n').encode()
    digest = hashlib.sha256(encoded).hexdigest()
    with tempfile.NamedTemporaryFile(prefix='.verification-', suffix='.tmp', dir=root, delete=False) as stream:
        temp = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temp, target)  # No replacement; fails safely if another process won the name.
    finally:
        temp.unlink(missing_ok=True)
    with target.with_name(target.name + '.sha256').open('x') as stream:
        stream.write(f'{digest}  {target.name}\n')
    return target, digest


def self_test():
    """Hand-calculated retrieval case plus corruption guards; no real experiment."""
    from scipy.io import savemat
    checks = []

    def rejects(name, callback):
        try:
            callback()
        except (ValueError, KeyError, TypeError):
            checks.append(name)
        else:
            raise AssertionError('Expected rejection: ' + name)

    labels = [np.array([7, 9]), np.array([1, 1]),
              np.array([-1, 7, 9, 7, 7, 7, 9, 7]), np.array([2, 1, 1, 2, 3, 1, 2, 2])]
    rank = np.arange(2, 10, dtype=np.int32)
    row, cmc = evaluate_query(rank, 0, labels, 8)
    close(row['AP'], 53/90, 'hand AP')
    close(row['INP'], 3/5, 'hand INP')
    require(cmc[0] == 0 and np.all(cmc[1:] == 1), 'Hand CMC failure')
    checks.append('AP=53/90; INP=3/5; same-camera and pid=-1 filtered; first match rank2')
    rejects('duplicate gallery', lambda: evaluate_query(np.array([2,3,4,5,6,7,8,8]), 0, labels, 8))
    rejects('wrong global offset', lambda: evaluate_query(np.arange(8), 0, labels, 8))
    rejects('float ranks', lambda: evaluate_query(rank.astype(float), 0, labels, 8))
    rejects('nonfinite metric', lambda: close(np.nan, 0, 'NaN'))
    rejects('shape broadcasting', lambda: close([1,1], [1], 'shape'))
    with tempfile.TemporaryDirectory(prefix='.verifier-selftest-', dir=Path(__file__).parent) as folder:
        root = Path(folder)
        mat = root/'features.mat'
        data = dict(zip(LABEL_KEYS, labels))
        data.update(query_f=np.ones((2,2)), gallery_f=np.ones((8,2)))
        savemat(mat, data)
        protocol_path, support_path = root/'PROTOCOL.md', root/'support.npz'
        protocol_path.write_text('Synthetic fixture protocol only.\n')
        support_path.write_bytes(b'Synthetic input hash fixture; not model data.\n')
        config = dict(query_count=2, gallery_count=8, depth=3, methods=['sample'], method='sample',
                      protocol_sha256=sha256(protocol_path), protocol_path=str(protocol_path),
                      support_sha256=sha256(support_path), support_file=str(support_path),
                      feature_sha256=sha256(mat), sources={str(Path(__file__).resolve()):sha256(__file__)})
        manifest = dict(config=config, signature=canonical_hash(config, compact=False))
        (root/'run-manifest.json').write_text(json.dumps(manifest))
        np.save(root/'sample.rankings.npy', np.stack([rank,rank]))
        rows, cmcs = zip(*(evaluate_query(rank, q, labels, 8) for q in range(2)))
        metrics = dict(mAP=np.mean([r['AP'] for r in rows]), mINP=np.mean([r['INP'] for r in rows]),
                       valid_queries=2, cmc=np.mean(cmcs,axis=0).tolist(), per_query=list(rows))
        metrics.update({f'rank-{k}':metrics['cmc'][k-1] for k in (1,5)})
        (root/'sample.json').write_text(json.dumps({'metrics':metrics}))
        (root/'sample').mkdir()
        for q in range(2):
            rp, pp = root/'sample'/f'{q:04d}.rankings.npy', root/'sample'/f'{q:04d}.prefix.npy'
            np.save(rp, rank)
            np.save(pp, np.array([q,2,3], dtype=np.int32))
            record = dict(query=q, mode='sample', signature=manifest['signature'],
                          files=dict(rankings_sha256=sha256(rp),prefix_sha256=sha256(pp)),
                          diagnostics=dict(status='complete_diffusion', diffusion_nodes=3,
                                           numerical_tail_count=0,max_relative_residual=1e-12,iterations=1),
                          trace=[dict(selected=[2,3],residual=1e-12)])
            (root/'sample'/f'{q:04d}.json').write_text(json.dumps(record))
        args = argparse.Namespace(output_dir=root, mat_path=mat, method='sample', manifest=None,
                                  source_root=[],path_map=[],expected_queries=2,expected_gallery=8,
                                  depth=3,no_diffusion=False,progress_every=0,protocol=None,support_file=None)
        report = verify(args)
        require(report['checks']['diffusion_checkpoints'] == 2, 'Synthetic integration failed')
        checks.append('complete synthetic integration with nested checkpoint hashes and wrapped metrics')
        support_path.write_bytes(b'corrupted')
        rejects('support input hash mismatch', lambda: verify(args))
        support_path.write_bytes(b'Synthetic input hash fixture; not model data.\n')
        protocol_path.write_text('Changed protocol\n')
        rejects('protocol input hash mismatch', lambda: verify(args))
        protocol_path.write_text('Synthetic fixture protocol only.\n')
        first, _ = write_report(root,'sample',report)
        second, _ = write_report(root,'sample',report)
        require(first != second and first.exists() and second.exists(), 'Report history overwritten')
        checks.append('immutable report versioning and SHA sidecar')
        checkpoint = root/'sample'/'0000.json'
        original = checkpoint.read_text()
        value = json.loads(original)
        value['diagnostics']['max_relative_residual'] = float('nan')
        checkpoint.write_text(json.dumps(value))
        rejects('NaN residual', lambda: verify(args))
        value = json.loads(original)
        value['diagnostics']['numerical_tail_count'] = 1
        checkpoint.write_text(json.dumps(value))
        rejects('numerical fallback', lambda: verify(args))
        checkpoint.write_text(original)
        value = json.loads(original)
        value['trace'][0]['selected'] = [3,2]
        checkpoint.write_text(json.dumps(value))
        rejects('trace and prefix mismatch', lambda: verify(args))
        checkpoint.write_text(original)
        value = json.loads(original)
        value['files']['rankings_sha256'] = '0'*64
        checkpoint.write_text(json.dumps(value))
        rejects('checkpoint ranking hash mismatch', lambda: verify(args))
        checkpoint.write_text(original)
        args.no_diffusion = True
        require(verify(args)['checks']['diffusion_checkpoints'] == 0, 'No-diffusion path failed')
        checks.append('rank-matrix-only no-diffusion mode')
        args.expected_queries = 3
        rejects('incomplete requested coverage', lambda: verify(args))
    return dict(status='passed', count=len(checks), checks=checks, real_experiment_data_read=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mat-path', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--method')
    parser.add_argument('--manifest', type=Path, help='Default: nearest run-manifest.json in output directory or four parents')
    parser.add_argument('--protocol', type=Path, help='Resolve declared protocol_sha256; otherwise use manifest path or nearby PROTOCOL.md')
    parser.add_argument('--support-file', type=Path, help='Resolve declared support_sha256; otherwise use manifest support_file/path')
    parser.add_argument('--source-root', action='append', type=Path, default=[], help='Resolve relative source hash paths; repeatable')
    parser.add_argument('--path-map', action='append', default=[], metavar='OLD_PREFIX=NEW_PREFIX', help='Relocate declared absolute source paths without changing their hashes')
    parser.add_argument('--expected-queries', type=int, default=3368)
    parser.add_argument('--expected-gallery', type=int, default=19732)
    parser.add_argument('--depth', type=int, default=500)
    parser.add_argument('--no-diffusion', action='store_true', help='Original/classic controls: require full rank matrix and metrics, skip checkpoint diagnostics')
    parser.add_argument('--progress-every', type=int, default=250)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    for name in ('mat_path','output_dir','method'):
        if getattr(args,name) is None:
            parser.error('--'+name.replace('_','-')+' is required unless --self-test is used')
    report = verify(args)
    path, digest = write_report(args.output_dir, args.method, report)
    print(json.dumps(dict(status='verified',method=args.method,report=str(path),sha256=digest,
                          queries=report['metrics']['query_count'],gallery=report['metrics']['gallery_count'],
                          mAP=report['metrics']['mAP'],mINP=report['metrics']['mINP']),indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f'VERIFICATION FAILED: {error}', file=sys.stderr)
        sys.exit(1)
