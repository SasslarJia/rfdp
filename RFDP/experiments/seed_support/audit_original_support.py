#!/usr/bin/env python3
"""Read-only, original-trajectory diagnostic of per-camera maximum seed support.

No scipy, training, diffusion solve, new ranking, or candidate rollout is used.
Only verified archive members are copied into a temporary working directory.
The primary population is original selected nodes, not the complete frontier.
Saved full frontiers/scores are used only where explicitly present and reported.
"""
import argparse
import bisect
import concurrent.futures
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

NAMES = ('market-osnet', 'market-transreid', 'duke-osnet', 'duke-transreid')
POPULATIONS = {'market': (3368, 19732), 'duke': (2228, 17661)}


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def log(**values):
    print(json.dumps(values, ensure_ascii=False), flush=True)


class SplitReader(io.RawIOBase):
    """Seekable view of the existing uncompressed split tar; no concatenation."""
    def __init__(self, folder):
        self.folder = Path(folder)
        metadata = json.loads((self.folder / 'INDEX.json').read_text())
        assert metadata['format'] == 'split-tar-v1'
        self.files = []
        self.ends = []
        total = 0
        for part in metadata['parts']:
            path = self.folder / part['name']
            assert path.stat().st_size == part['size']
            self.files.append(open(path, 'rb'))
            total += part['size']
            self.ends.append(total)
        assert total == metadata['tar_bytes']
        self.size, self.position = total, 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else (self.position + offset if whence == 1 else self.size + offset)
        if position < 0: raise ValueError('negative seek')
        self.position = position
        return position

    def read(self, count=-1):
        if count < 0: count = self.size - self.position
        count = min(count, self.size - self.position)
        chunks = []
        while count > 0:
            part = bisect.bisect_right(self.ends, self.position)
            start = 0 if part == 0 else self.ends[part - 1]
            take = min(count, self.ends[part] - self.position)
            self.files[part].seek(self.position - start)
            chunk = self.files[part].read(take)
            if len(chunk) != take: raise EOFError('short split-tar read')
            chunks.append(chunk)
            self.position += take
            count -= take
        return b''.join(chunks)

    def close(self):
        for stream in self.files: stream.close()
        super().close()


def source_manifest(spec):
    data = Path(spec['archive_manifest']).read_bytes()
    assert sha_bytes(data) == spec['archive_manifest_sha256']
    if spec['archive_manifest'].endswith('.jsonl'):
        rows = [json.loads(line) for line in data.decode().splitlines()]
        return {r['path']: {'bytes': r['size'], 'sha256': r['sha256']}
                for r in rows if r.get('type') == 'file'}
    return json.loads(data)['members']


def verified_extract(archive, wanted, ledger, label):
    """wanted maps exact member names to (destination, member SHA/size)."""
    outstanding = {}
    for member, (target, expected) in wanted.items():
        target = Path(target)
        if target.exists() and target.stat().st_size == expected['bytes'] and sha_file(target) == expected['sha256']:
            ledger.append({'member': member, 'bytes': expected['bytes'], 'sha256': expected['sha256'], 'local': str(target)})
        else:
            outstanding[member] = (target, expected)
    if not outstanding:
        log(action='verified_cached_members', source=label, members=len(wanted))
        return
    reader = SplitReader(archive) if Path(archive).is_dir() else None
    tf = tarfile.open(fileobj=reader, mode='r:') if reader else tarfile.open(archive, 'r|gz')
    count = 0
    try:
        for member in tf:
            name = member.name.removeprefix('./')
            if name not in outstanding: continue
            target, expected = outstanding.pop(name)
            assert member.isfile() and member.size == expected['bytes'], name
            data = tf.extractfile(member).read()
            assert len(data) == expected['bytes'] and sha_bytes(data) == expected['sha256'], name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            ledger.append({'member': name, 'bytes': len(data), 'sha256': expected['sha256'], 'local': str(target)})
            count += 1
            if count % 1500 == 0:
                log(action='extract_progress', source=label, copied=count, remaining=len(outstanding))
            if not outstanding: break
    finally:
        tf.close()
        if reader: reader.close()
    assert not outstanding, list(outstanding)[:5]
    log(action='extracted_verified_members', source=label, members=count)


def prepare(index_path, work):
    index = json.loads(Path(index_path).read_text())
    ledger = []
    labels = index['label_archive']
    wanted = {name: (work / 'labels' / name.split('/')[0].replace('-features-v1', '') / 'image-manifest.json', metadata)
              for name, metadata in labels['manifest_copy'].items()}
    verified_extract(labels['archive'], wanted, ledger, 'label-manifests')
    for name in NAMES:
        specification = index['data_archives'][name]
        members = source_manifest(specification)
        frozen = index['frozen_inputs'][name]
        manifest_member = [p for p, m in members.items() if p.endswith('/run-manifest.json') and m['sha256'] == frozen['run_manifest_sha256']]
        assert len(manifest_member) == 1, (name, manifest_member)
        run_member = manifest_member[0]
        prefix = run_member.rsplit('/', 1)[0] + '/queries/' if name != 'duke-osnet' else run_member.rsplit('/', 1)[0] + '/osnet/ihyrdp/'
        wanted = {run_member: (work / name / 'run-manifest.json', members[run_member])}
        qjson = prefixes = 0
        for member, metadata in members.items():
            if not member.startswith(prefix): continue
            tail = member[len(prefix):]
            if re.fullmatch(r'\d{4}\.json', tail): qjson += 1
            elif re.fullmatch(r'\d{4}\.prefix\.npy', tail): prefixes += 1
            else: continue
            wanted[member] = (work / name / 'queries' / tail, metadata)
        qn, _ = POPULATIONS[name.split('-')[0]]
        assert qjson == prefixes == qn, (name, qjson, prefixes)
        graph_members = [p for p, m in members.items() if p.endswith('/graph.npz') and m['sha256'] == frozen['graph_sha256']]
        if name != 'duke-osnet':
            assert len(graph_members) == 1
            gm = graph_members[0]
            wanted[gm] = (work / name / 'graph.npz', members[gm])
        verified_extract(specification['archive'], wanted, ledger, name)
    early = Path(index['baseline_graph_archive_for_duke_osnet']['archive'])
    early_manifest = early.with_name(early.name.replace('.tar.gz', '.manifest.json'))
    early_members = json.loads(early_manifest.read_text())['members']
    expected_sha = index['frozen_inputs']['duke-osnet']['graph_sha256']
    matches = [p for p, m in early_members.items() if p.endswith('/graph.npz') and m['sha256'] == expected_sha]
    assert len(matches) == 1
    gm = matches[0]
    verified_extract(early, {gm: (work / 'duke-osnet' / 'graph.npz', early_members[gm])}, ledger, 'duke-identical-graph-only')
    ledger.sort(key=lambda r: (r['local'], r['member']))
    encoded = ''.join(json.dumps(r, sort_keys=True, ensure_ascii=False) + '\n' for r in ledger).encode()
    (work / 'verified-members.jsonl').write_bytes(encoded)
    provenance = dict(index_path=str(index_path), index_sha256=sha_file(index_path),
                      member_count=len(ledger), verified_member_bytes=sum(r['bytes'] for r in ledger),
                      member_ledger_sha256=sha_bytes(encoded),
                      verification='Each used member SHA and size checked; no whole-archive restoration or full-part hashing.',
                      alternate_method_content_used='Only Duke OSNet graph matching the original graph SHA; no alternate selections/scores.')
    dump(work / 'extraction-provenance.json', provenance)
    return provenance


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    if len(a) == 0: return {'n': 0}
    assert np.isfinite(a).all()
    levels = [0, .25, .5, .75, .9, .95, 1]
    return {'n': len(a), 'mean': float(a.mean()),
            'quantiles': {str(p): float(v) for p, v in zip(levels, np.quantile(a, levels))}}


class Group:
    def __init__(self):
        self.count = self.direct = self.repeated = self.single_camera_multiple_nodes = 0
        self.values = {k: [] for k in ('e_sum', 'e_camera_max', 'removed_fraction', 'support_nodes', 'support_cameras', 'delta_p_score_covered', 'local_rhs_shift_score_covered')}

    def add(self, positions, features):
        for k in positions:
            e, capped, nodes, cameras = (features[a][k] for a in ('e', 'cap', 'nodes', 'cameras'))
            self.count += 1
            self.direct += int(e > 0)
            self.repeated += int(nodes > cameras)
            self.single_camera_multiple_nodes += int(nodes >= 2 and cameras == 1)
            for key, value in [('e_sum', e), ('e_camera_max', capped), ('support_nodes', nodes), ('support_cameras', cameras)]:
                self.values[key].append(float(value))
            if e > 0: self.values['removed_fraction'].append(float(features['R'][k]))
            if features['d'] is not None:
                self.values['delta_p_score_covered'].append(float(features['dp'][k]))
                self.values['local_rhs_shift_score_covered'].append(float(features['d'][k]))

    def result(self):
        return dict(count=self.count, direct_count=self.direct, zero_boundary_count=self.count-self.direct,
                    repeated_same_camera_support_count=self.repeated,
                    multiple_nodes_single_camera_count=self.single_camera_multiple_nodes,
                    **{k: distribution(v) for k, v in self.values.items()})


class PairGroup:
    def __init__(self):
        self.total = self.with_both_direct = self.n_more = self.g_more = self.equal = 0
        self.delta_r = []
        self.delta_d = []

    def add(self, g, n, features):
        self.total += 1
        if np.isfinite(features['R'][g]) and np.isfinite(features['R'][n]):
            delta = float(features['R'][n] - features['R'][g])
            self.with_both_direct += 1
            self.n_more += int(delta > 1e-12)
            self.g_more += int(delta < -1e-12)
            self.equal += int(abs(delta) <= 1e-12)
            self.delta_r.append(delta)
        if features['d'] is not None:
            self.delta_d.append(float(features['d'][n] - features['d'][g]))

    def result(self):
        return dict(pairs=self.total, both_direct_pairs=self.with_both_direct,
                    negative_larger_removed_fraction=self.n_more,
                    positive_larger_removed_fraction=self.g_more, equal_within_1e_12=self.equal,
                    negative_minus_positive_R=distribution(self.delta_r),
                    negative_minus_positive_local_rhs_shift_score_covered=distribution(self.delta_d))


def analyze_one(arguments):
    name, index_path, evidence_path, work_str = arguments
    started = time.monotonic()
    work = Path(work_str)
    index = json.loads(Path(index_path).read_text())
    evidence = json.loads(Path(evidence_path).read_text())
    family = name.split('-')[0]
    qn, gn = POPULATIONS[family]
    frozen = index['frozen_inputs'][name]
    labels = json.loads((work / 'labels' / family / 'image-manifest.json').read_text())
    assert len(labels) == qn + gn
    assert all(r['split'] == 'query' for r in labels[:qn])
    assert all(r['split'] != 'query' for r in labels[qn:])
    pids = np.asarray([r['pid'] for r in labels], dtype=np.int64)
    cameras_raw = np.asarray([r['camid'] for r in labels], dtype=np.int64)
    camera_labels, camera_codes = np.unique(cameras_raw, return_inverse=True)
    n, ncamera = len(labels), len(camera_labels)
    graph_path = work / name / 'graph.npz'
    assert sha_file(graph_path) == frozen['graph_sha256']
    with np.load(graph_path, allow_pickle=False) as g:
        assert g['format'].item() in (b'csr', 'csr')
        assert g['shape'].tolist() == [n, n]
        values = g['data'].astype(np.float64)
        indices = g['indices'].astype(np.int64)
        indptr = g['indptr'].astype(np.int64)
    assert np.isfinite(values).all() and np.all(values > 0)
    neighbor_ids = [indices[indptr[i]:indptr[i+1]] for i in range(n)]
    neighbor_weights = [values[indptr[i]:indptr[i+1]] for i in range(n)]
    assert all(np.all(np.diff(ids) > 0) for ids in neighbor_ids)
    groups = {key: Group() for key in ('all/positive', 'all/harmful_negative',
              'post_initial_nonfirst/positive', 'post_initial_nonfirst/harmful_negative',
              'initial/positive', 'initial/harmful_negative',
              'post_initial_nonfirst/positive_without_query_graph_edge')}
    pairs = {}
    coverage = {k: 0 for k in ('queries', 'stages', 'selected_nodes', 'full_frontier_stages',
                'full_frontier_and_scores_stages', 'full_frontier_queries',
                'full_frontier_and_scores_queries', 'score_covered_selected_nodes',
                'bridge_stages', 'bridge_edges', 'recovered_bridge_stages', 'bridge_selected_nodes')}
    locked = dict(queries_with_valid_gallery_in_first_batch=0,
                  queries_with_wrong_first_valid_gallery_in_first_batch=0,
                  queries_with_correct_first_valid_gallery_in_first_batch=0,
                  first_valid_gallery_offset_counts={}, wrong_query_ids=[],
                  queries_with_valid_gallery_at_first_nomination=0,
                  wrong_first_nomination_queries=0)
    r1_wrong_observed = 0
    trace_ledger = hashlib.sha256()
    source_members = source_manifest(index['data_archives'][name])
    member_by_query = {Path(k).name: v for k, v in source_members.items()
                       if (('/queries/' in k and name != 'duke-osnet') or ('/evidence/osnet/ihyrdp/' in k and name == 'duke-osnet'))
                       and (re.fullmatch(r'\d{4}\.json', Path(k).name) or re.fullmatch(r'\d{4}\.prefix\.npy', Path(k).name))}
    for query in range(qn):
        qpath = work / name / 'queries' / f'{query:04d}.json'
        ppath = qpath.with_name(f'{query:04d}.prefix.npy')
        qb, pb = qpath.read_bytes(), ppath.read_bytes()
        qsha, psha = sha_bytes(qb), sha_bytes(pb)
        assert qsha == member_by_query[qpath.name]['sha256']
        assert psha == member_by_query[ppath.name]['sha256']
        row = json.loads(qb)
        assert row['query'] == query
        if 'signature' in row: assert row['signature'] == frozen['run_signature']
        if 'prefix_sha256' in row: assert row['prefix_sha256'] == psha
        prefix = np.load(io.BytesIO(pb), allow_pickle=False).astype(np.int64)
        assert len(prefix) == 500 and prefix[0] == query and len(set(prefix)) == 500
        stages = row['trace']
        replay = [query] + [int(i) for t in stages for i in t['selected']]
        assert replay == prefix.tolist()
        trace_ledger.update(f'{query}:{qsha}:{psha}\n'.encode())
        valid = (np.arange(n) >= qn) & (pids != -1) & ~((pids == pids[query]) & (cameras_raw == cameras_raw[query]))
        positive = valid & (pids == pids[query])
        selected_positive_positions = np.flatnonzero(positive[prefix])
        unselected_positives = int(positive.sum()) - len(selected_positive_positions)
        harmful = np.zeros(n, dtype=bool)
        if unselected_positives > 0:
            harmful[prefix] = valid[prefix] & ~positive[prefix]
        elif len(selected_positive_positions):
            ids = prefix[:int(selected_positive_positions[-1])]
            harmful[ids] = valid[ids] & ~positive[ids]
        eligible = prefix[valid[prefix]]
        assert len(eligible) > 0
        r1_wrong_observed += int(not positive[eligible[0]])
        initial = np.asarray(stages[0]['selected'], dtype=np.int64)
        offsets = np.flatnonzero(valid[initial])
        if len(offsets):
            offset = int(offsets[0]); first = int(initial[offset])
            wrong = not bool(positive[first])
            locked['queries_with_valid_gallery_in_first_batch'] += 1
            locked['queries_with_wrong_first_valid_gallery_in_first_batch'] += int(wrong)
            locked['queries_with_correct_first_valid_gallery_in_first_batch'] += int(not wrong)
            key = str(offset)
            locked['first_valid_gallery_offset_counts'][key] = locked['first_valid_gallery_offset_counts'].get(key, 0) + 1
            if wrong: locked['wrong_query_ids'].append(query)
            if offset == 0:
                locked['queries_with_valid_gallery_at_first_nomination'] += 1
                locked['wrong_first_nomination_queries'] += int(wrong)
        e_all = np.zeros(n, dtype=np.float64)
        maxima = np.zeros((n, ncamera), dtype=np.float64)
        source_counts = np.zeros(n, dtype=np.int16)
        def add_seed(seed):
            ids, weights = neighbor_ids[seed], neighbor_weights[seed]
            e_all[ids] += weights
            source_counts[ids] += 1
            cam = camera_codes[seed]
            maxima[ids, cam] = np.maximum(maxima[ids, cam], weights)
        add_seed(query)
        labeled = [query]
        qneighbors = set(map(int, neighbor_ids[query]))
        has_u = has_scores = False
        recovery = evidence['bridge_metadata']['queries'].get(str(query)) if name == 'duke-osnet' else None
        if recovery: assert recovery['original_record_sha256'] == qsha
        for stage_number, stage in enumerate(stages):
            assert stage.get('iteration', stage_number) == stage_number
            if 'labeled' in stage: assert stage['labeled'] == labeled
            if 'labeled_count' in stage: assert stage['labeled_count'] == len(labeled)
            selected = np.asarray(stage['selected'], dtype=np.int64)
            assert 1 <= len(selected) <= 5
            bridges = stage.get('bridges', [])
            if recovery and recovery['iteration'] == stage_number:
                assert stage['selected'] == recovery['original_selected']
                if bridges: assert bridges == recovery['bridges']
                else:
                    bridges = recovery['bridges']
                    coverage['recovered_bridge_stages'] += 1
            coverage['bridge_stages'] += int(bool(bridges))
            coverage['bridge_edges'] += len(bridges)
            e = e_all[selected].copy()
            mx = maxima[selected].copy()
            nodes = source_counts[selected].astype(np.int64)
            for outside, inside, weight in bridges:
                assert inside in labeled and outside not in labeled and weight > 0
                positions = np.flatnonzero(selected == outside)
                if len(positions):
                    k = int(positions[0])
                    assert e[k] == 0
                    e[k] += weight
                    mx[k, camera_codes[inside]] = max(mx[k, camera_codes[inside]], weight)
                    nodes[k] += 1
                    coverage['bridge_selected_nodes'] += 1
            capped = mx.sum(axis=1)
            camera_counts = np.count_nonzero(mx, axis=1)
            assert np.array_equal(e > 0, capped > 0)
            assert np.all(capped <= e + 1e-12 * np.maximum(e, 1e-200))
            ratio = np.full(len(selected), np.nan)
            np.divide(capped, e, out=ratio, where=e > 0)
            removed = np.clip(1-ratio, 0, 1)
            if stage_number == 0: assert np.allclose(e, capped, rtol=1e-14, atol=0)
            features = dict(e=e, cap=capped, R=removed, nodes=nodes, cameras=camera_counts, dp=None, d=None)
            u, scores = stage.get('unlabeled'), stage.get('scores')
            if u is not None:
                u = np.asarray(u, dtype=np.int64)
                assert len(u) == stage.get('size', len(u)) and len(set(u)) == len(u)
                assert np.all(np.diff(u) > 0)
                has_u = True; coverage['full_frontier_stages'] += 1
                if scores is not None:
                    scores = np.asarray(scores, dtype=np.float64)
                    assert len(scores) == len(u) and np.isfinite(scores).all()
                    local = np.searchsorted(u, selected)
                    assert np.array_equal(u[local], selected)
                    umask = np.zeros(n, dtype=bool); umask[u] = True
                    h = np.asarray([neighbor_weights[i][umask[neighbor_ids[i]]].sum() for i in selected])
                    assert np.all(h > 0)
                    dp = e/(h+e)-capped/(h+capped)
                    features['dp'] = dp
                    features['d'] = 5. * dp * scores[local]
                    has_scores = True
                    coverage['full_frontier_and_scores_stages'] += 1
                    coverage['score_covered_selected_nodes'] += len(selected)
            gp = np.flatnonzero(positive[selected])
            gn_positions = np.flatnonzero(harmful[selected])
            groups['all/positive'].add(gp, features)
            groups['all/harmful_negative'].add(gn_positions, features)
            if stage_number == 0:
                groups['initial/positive'].add(gp, features)
                groups['initial/harmful_negative'].add(gn_positions, features)
            else:
                groups['post_initial_nonfirst/positive'].add(gp[gp > 0], features)
                groups['post_initial_nonfirst/harmful_negative'].add(gn_positions[gn_positions > 0], features)
                without_q = [int(k) for k in gp if k > 0 and int(selected[k]) not in qneighbors]
                groups['post_initial_nonfirst/positive_without_query_graph_edge'].add(without_q, features)
            # Pairs include every officially eligible negative in the batch,
            # even one originally after the last positive: promoting that item
            # could damage an originally correct order. Marginal groups above
            # continue to use the separately defined harmful-negative category.
            pair_negatives = np.flatnonzero(valid[selected] & ~positive[selected])
            for g in gp:
                for negative in pair_negatives:
                    order = 'original_wrong' if negative < g else 'original_correct'
                    keys = ['all_slots/' + order]
                    if stage_number > 0 and g > 0 and negative > 0: keys.append('post_initial_nonfirst/' + order)
                    if nodes[g] == nodes[negative] and camera_counts[g] == camera_counts[negative]:
                        keys.append('matched_support_node_and_camera_counts/' + order)
                    for key in keys:
                        pairs.setdefault(key, PairGroup()).add(int(g), int(negative), features)
            coverage['stages'] += 1
            coverage['selected_nodes'] += len(selected)
            for seed in selected: add_seed(int(seed))
            labeled.extend(selected.tolist())
        assert labeled == prefix.tolist()
        coverage['queries'] += 1
        coverage['full_frontier_queries'] += int(has_u)
        coverage['full_frontier_and_scores_queries'] += int(has_scores)
        if (query + 1) % 250 == 0 or query + 1 == qn:
            log(action='analyze_progress', input=name, queries=query+1, total=qn, seconds=round(time.monotonic()-started, 1))
    expected = evidence['admission_support'][name]['counts']
    assert coverage['stages'] == expected['saved_stages']
    assert coverage['selected_nodes'] == expected['all/selected_nodes']
    assert groups['all/positive'].count == expected['all/positive']
    assert groups['all/harmful_negative'].count == expected['all/harmful_negative']
    assert groups['post_initial_nonfirst/positive'].count == expected['post_initial_nonfirst/positive']
    assert groups['post_initial_nonfirst/harmful_negative'].count == expected['post_initial_nonfirst/harmful_negative']
    assert r1_wrong_observed == expected['rank1_errors']
    assert locked['queries_with_valid_gallery_at_first_nomination'] == expected['initial_first_is_official_valid_gallery']
    assert locked['wrong_first_nomination_queries'] == expected['rank1_errors_fixed_by_initial_first_item']
    for category in ('positive', 'harmful_negative'):
        key = 'post_initial_nonfirst/' + category
        assert groups[key].single_camera_multiple_nodes == expected[key+'_two_nodes'] - expected[key+'_pass']
    matched_expected = evidence['matched_pairs'][name]['counts']
    for order in ('original_wrong', 'original_correct'):
        expected_pair_count = sum(v for k, v in matched_expected.items() if k.startswith(order+'/'))
        assert pairs['post_initial_nonfirst/'+order].total == expected_pair_count
    result = dict(input=name, official_queries=qn, official_gallery=gn,
                  graph_sha256=frozen['graph_sha256'], run_signature=frozen['run_signature'],
                  verified_query_prefix_ledger_sha256=trace_ledger.hexdigest(),
                  source_coverage=coverage, groups={k: v.result() for k, v in groups.items()},
                  same_query_same_stage_pairs={k: v.result() for k, v in pairs.items()},
                  first_batch_rank1_locked=locked, original_rank1_errors_reproduced=r1_wrong_observed,
                  prior_compact_counts_reproduced=True, seconds=time.monotonic()-started)
    dump(work / f'{name}-summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True, help='Local frozen-input/archive index; never checked into Git.')
    parser.add_argument('--evidence', type=Path, required=True, help='Local original-state count and bridge evidence JSON.')
    parser.add_argument('--work', type=Path, required=True, help='Temporary extraction/output directory outside the checkout.')
    parser.add_argument('--action', choices=('extract', 'analyze', 'all'), default='all')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if args.action in ('extract', 'all'): prepare(args.index, args.work)
    if args.action == 'extract': return
    jobs = [(name, str(args.index), str(args.evidence), str(args.work)) for name in NAMES]
    if args.workers == 1:
        outputs = [analyze_one(j) for j in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            outputs = list(executor.map(analyze_one, jobs))
    summary = dict(scope='Original selected-node static analysis only. No new solver calls, rankings, training, or counterfactual rollout.',
        candidate='e_camera_max(i) = sum over source cameras c of max G(i,s), s in old L and camera(s)=c; temporary original bridges included only for their original stage.',
        primary_population='Original accepted nodes from all original saved stages. Official filtering is applied only to diagnostic categories.',
        harmful_negative='Officially eligible negative in the original prefix that precedes at least one eligible positive, including positives left for the original feature tail.',
        matched_pairs='All official eligible positive-negative pairs in the same original query and batch, including negatives originally after the final positive. Marginal harmful-negative statistics have the narrower definition above.',
        missing_scores='No score or full frontier is fabricated. delta_p and local_rhs_shift are restricted to stages retaining both complete U and original scores; coverage is explicit.',
        local_rhs_shift='5 * (p_sum-p_camera_max) * original_score. This is a local equation perturbation, not a predicted new score or ranking change.',
        camera_interpretation='Source camera grouping, not identity probability or independent evidence.',
        first_batch_interpretation='With L={q}, the sum and per-camera maximum are exactly equal. If the original first batch contains an official eligible gallery item, its first such item and Rank-1 outcome are invariant for this proposed aggregation.',
        new_solver_calls=0, new_rankings=0, new_rollouts=0, training_runs=0,
        script_sha256=sha_file(__file__), index_sha256=sha_file(args.index), evidence_sha256=sha_file(args.evidence),
        extraction=json.loads((args.work/'extraction-provenance.json').read_text()),
        inputs={r['input']: r for r in outputs})
    dump(args.work / 'static-seed-camera-max-summary.json', summary)
    log(action='complete', result=str(args.work / 'static-seed-camera-max-summary.json'))


if __name__ == '__main__': main()
