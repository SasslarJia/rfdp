"""Locate original Rank-1 failures; never solve or construct a new trajectory."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def analyze(work, name):
    family = name.split('-')[0]
    qn = 3368 if family == 'market' else 2228
    labels = json.loads((work/'labels'/family/'image-manifest.json').read_text())
    pid = np.array([x['pid'] for x in labels])
    cam = np.array([x['camid'] for x in labels])
    n = len(labels)
    graph = work/name/'graph.npz'
    with np.load(graph, allow_pickle=False) as f:
        indices, indptr = f['indices'], f['indptr']
        rows = [set(map(int, indices[indptr[i]:indptr[i+1]])) for i in range(n)]
    count = Counter()
    examples = {}
    for q in range(qn):
        record = json.loads((work/name/'queries'/f'{q:04d}.json').read_text())
        stage = record['trace'][0]
        batch = stage['selected']
        eligible = (np.arange(n) >= qn) & (pid != -1) & ~((pid == pid[q]) & (cam == cam[q]))
        positives = set(map(int, np.flatnonzero(eligible & (pid == pid[q]))))
        a = rows[q] - {q}
        for outside, inside, weight in stage.get('bridges', []):
            assert inside == q and weight > 0
            a.add(outside)
        u = (a | set().union(*(rows[i] for i in a))) - {q}
        if 'unlabeled' in stage:
            assert set(stage['unlabeled']) == u
        assert set(batch) <= u
        count['queries'] += 1
        count['queries_with_positive_in_initial_A'] += bool(a & positives)
        count['queries_with_positive_in_initial_U'] += bool(u & positives)
        eligible_offsets = [i for i, node in enumerate(batch) if eligible[node]]
        if not eligible_offsets:
            count['first_batch_without_eligible_gallery'] += 1
            continue
        count['first_batch_with_eligible_gallery'] += 1
        offset = eligible_offsets[0]
        wrong = batch[offset]
        if wrong in positives:
            count['first_batch_rank1_correct'] += 1
            continue
        count['first_batch_rank1_wrong'] += 1
        count['wrong_offset_' + str(offset)] += 1
        count['wrong_direct_A' if wrong in a else 'wrong_indirect_B'] += 1
        count['wrong_same_camera_as_query'] += cam[wrong] == cam[q]
        if a & positives:
            count['wrong_has_positive_in_A'] += 1
        elif u & positives:
            count['wrong_positive_only_in_B'] += 1
        else:
            count['wrong_without_positive_in_U'] += 1
        earlier = batch[:offset]
        correct_earlier = [s for s in earlier if pid[s] == pid[q]]
        count['wrong_with_earlier_same_identity_in_batch'] += bool(correct_earlier)
        count['wrong_with_earlier_same_camera_same_identity_in_batch'] += any(cam[s] == cam[q] for s in correct_earlier)
        support = (set().union(*(rows[s] for s in correct_earlier)) & positives) if correct_earlier else set()
        count['wrong_earlier_correct_nodes_support_eligible_positive'] += bool(support)
        count['wrong_earlier_correct_nodes_support_positive_without_q_edge'] += bool(support-a)
        count['wrong_earlier_correct_nodes_support_positive_outside_U'] += bool(support-u)
        count['wrong_earlier_correct_nodes_also_support_wrong'] += any(wrong in rows[s] for s in correct_earlier)
        if support and all(wrong not in rows[s] for s in correct_earlier):
            count['wrong_earlier_correct_nodes_support_positive_but_not_wrong'] += 1
            examples.setdefault('positive_support_not_wrong', []).append(dict(query=q, batch=batch,
                wrong=wrong, correct_earlier=correct_earlier, supported_positives=sorted(support)[:10],
                positive_in_A=len(a & positives), positive_in_B=len((u-a) & positives)))
    old = json.loads((work/f'{name}-summary.json').read_text())['first_batch_rank1_locked']
    assert count['first_batch_rank1_wrong'] == old['queries_with_wrong_first_valid_gallery_in_first_batch']
    assert count['first_batch_rank1_correct'] == old['queries_with_correct_first_valid_gallery_in_first_batch']
    return dict(counts={k:int(v) for k,v in count.items()}, examples={k:v[:4] for k,v in examples.items()},
                graph_sha256=hashlib.sha256(graph.read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = dict(scope='Original first-batch topology and label audit only; no recomputed scores or new selections.',
        interpretation='Earlier correct nodes are identified post hoc using labels, never used by a proposed method. Graph support does not establish a score flip.',
        inputs={name:analyze(args.work,name) for name in ('market-osnet','market-transreid','duke-osnet','duke-transreid')},
        new_solver_calls=0,new_rankings=0,new_rollouts=0,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for name, row in result['inputs'].items():
        print(name,json.dumps(row['counts'],ensure_ascii=False))


if __name__ == '__main__':
    main()
