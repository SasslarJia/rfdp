#!/usr/bin/env python3
"""Prepare identity-disjoint Market training development features.

Split-only mode imports only Python's standard library. Full extraction requires
NumPy, SciPy, Pillow, torch and torchvision on the existing GPU environment.
No downloads, backbone fitting, test features or test metrics are used.
"""

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import io
import json
import platform
import random
import re
import time
from pathlib import Path
import zipfile


SEED = 20261015
EXPECTED_IMAGES = 12936
EXPECTED_IDENTITIES = 751
FIT_IDENTITIES = 500
ARCHIVE_SHA256 = '416bb77b5a2449b32e936f623cbee58becf1a9e7e936f36380cb8f9ab928fe96'
PROTOCOL = 'market-train-id500-251-multicamera-query-20261015-v2'
PREPROCESSING = {
    'color': 'RGB', 'resize_hw': [256, 128], 'interpolation': 'bilinear',
    'antialias': True, 'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225], 'flip': False, 'autocast': False,
    'output_dtype': 'float32', 'normalization': 'one L2 normalization after inference',
}
MODELS = {
    'resnet50': {
        'constructor': 'resnet50', 'feature_dim': 2048, 'torch_threads': 4,
        'checkpoint_sha256': '15d3fab6912688516ed2c281961eaf7ee8590ff51dfd6f61ba64daba90325c31',
        'model_source_sha256': '6b871338b477421bbc23bc7b3028735fc67ab4f551d979c5ae8b2edf2d02c168',
        'historical_test_feature_sha256': 'af7ee69c3ee36d67a8a62e1d9faa178b2ebd11382f8eec2d5a7bcfb7cec1cbd2',
    },
    'osnet': {
        'constructor': 'osnet_x1_0', 'feature_dim': 512, 'torch_threads': 2,
        'checkpoint_sha256': '2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154',
        'model_source_sha256': 'c7c1c29187d6330f859c91da229271531920464c7011aec13842a086b2263cae',
        'historical_test_feature_sha256': '9f603d7149c99e5f4ba44934829d03d3fa1e25f737924c9bb6fa81f37c2d9c31',
    },
}
NAME_PATTERN = re.compile(r'(?P<pid>-?\d+)_c(?P<camera>[1-6])s\d+_\d+_\d+\.jpg')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(data):
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + '\n'


def write_once(path, content):
    """Permit identical split reuse, but never silently overwrite evidence."""
    path = Path(path)
    if path.exists():
        if path.read_text(encoding='utf-8') != content:
            raise FileExistsError(f'Existing output differs: {path}; choose a new output directory')
        return
    with path.open('x', encoding='utf-8') as stream:
        stream.write(content)


def record_from_name(name, source=None, digest=None):
    match = NAME_PATTERN.fullmatch(name)
    if not match:
        raise ValueError(f'Unexpected Market training filename: {name}')
    pid, camera = int(match['pid']), int(match['camera'])
    if pid <= 0:
        raise ValueError(f'Junk/distractor identity is invalid in the training split: {name}')
    row = {'name': name, 'pid': pid, 'camera': camera}
    if source is not None:
        row['source'] = source
    if digest is not None:
        row['sha256'] = digest
    return row


def scan_training_images(archive=None, dataset=None):
    rows = []
    if archive is not None:
        archive = Path(archive).resolve(strict=True)
        actual_hash = sha256(archive)
        if actual_hash != ARCHIVE_SHA256:
            raise ValueError(f'Market archive SHA256 differs from the historical test archive: {actual_hash}')
        with zipfile.ZipFile(archive) as z:
            members = [name for name in z.namelist()
                       if len(Path(name).parts) >= 2
                       and Path(name).parts[-2] == 'bounding_box_train'
                       and name.endswith('.jpg') and not Path(name).name.startswith('._')]
            if len(set(members)) != len(members):
                raise ValueError('Duplicate training members in archive')
            for member in sorted(members, key=lambda value: Path(value).name):
                raw = z.read(member)
                rows.append(record_from_name(Path(member).name, member, hashlib.sha256(raw).hexdigest()))
        origin = {'kind': 'zip', 'path': str(archive), 'archive_sha256': actual_hash}
    else:
        dataset = Path(dataset).resolve(strict=True)
        folder = dataset if dataset.name == 'bounding_box_train' else dataset / 'bounding_box_train'
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for path in sorted(folder.glob('*.jpg')):
            if not path.name.startswith('._'):
                rows.append(record_from_name(path.name, path.name, sha256(path)))
        origin = {'kind': 'directory', 'path': str(folder), 'archive_sha256': None,
                  'note': 'Per-image hashes recorded; unpacked source is not archive-hash authenticated.'}
    if not rows:
        raise ValueError('No training images found')
    return rows, origin


def build_split(records, seed=SEED, fit_count=FIT_IDENTITIES,
                expected_identities=EXPECTED_IDENTITIES, expected_images=EXPECTED_IMAGES):
    """One first image from each camera with >=2 images is query; retain the rest.

    If every camera has just one image, query the camera at pid % camera_count
    in sorted camera order. Reject identities without cross-camera positives.
    The production CLI fixes the identity counts and seed above.
    """
    rows = sorted(records, key=lambda row: row['name'])
    names = [row['name'] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate image basenames')
    if len(rows) != expected_images:
        raise ValueError(f'Expected {expected_images} training images, found {len(rows)}')
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        parsed = record_from_name(row['name'])
        if (parsed['pid'], parsed['camera']) != (row['pid'], row['camera']):
            raise ValueError(f'Filename/metadata mismatch: {row["name"]}')
        grouped[row['pid']].append(index)
    ids = sorted(grouped)
    if len(ids) != expected_identities:
        raise ValueError(f'Expected {expected_identities} identities, found {len(ids)}')
    if not 0 < fit_count < len(ids):
        raise ValueError('Both fit and validation identity sets must be nonempty')
    single_camera = [pid for pid in ids if len({rows[i]['camera'] for i in grouped[pid]}) < 2]
    if single_camera:
        raise ValueError(f'Cannot cover every identity with a cross-camera query: {single_camera}')
    shuffled = ids.copy()
    random.Random(seed).shuffle(shuffled)
    partition_ids = {'fit': sorted(shuffled[:fit_count]), 'val': sorted(shuffled[fit_count:])}
    partitions = {}
    assigned = set()
    for name, selected_ids in partition_ids.items():
        selected = set(selected_ids)
        query = []
        fallback_ids = []
        for pid in selected_ids:
            by_camera = defaultdict(list)
            for index in grouped[pid]:
                by_camera[rows[index]['camera']].append(index)
            chosen = [indices[0] for camera, indices in sorted(by_camera.items()) if len(indices) >= 2]
            if not chosen:
                cameras = sorted(by_camera)
                chosen = [by_camera[cameras[pid % len(cameras)]][0]]
                fallback_ids.append(pid)
            query.extend(chosen)
        query.sort()
        query_set = set(query)
        gallery = [i for i, row in enumerate(rows) if row['pid'] in selected and i not in query_set]
        gallery_set = set(gallery)
        if query_set & gallery_set or assigned & (query_set | gallery_set):
            raise AssertionError('Overlapping query/gallery or identity partitions')
        positives = [sum(rows[j]['pid'] == rows[i]['pid'] and rows[j]['camera'] != rows[i]['camera']
                         for j in grouped[rows[i]['pid']] if j in gallery_set) for i in query]
        if not all(count > 0 for count in positives):
            raise AssertionError('Query without a cross-camera gallery positive')
        assigned |= query_set | gallery_set
        partitions[name] = {
            'identity_ids': selected_ids, 'query_indices': query, 'gallery_indices': gallery,
            'query_paths': [rows[i]['name'] for i in query],
            'gallery_paths': [rows[i]['name'] for i in gallery],
            'identity_count': len(selected_ids), 'query_count': len(query),
            'gallery_count': len(gallery), 'image_count': len(query) + len(gallery),
            'all_singleton_camera_fallback_ids': fallback_ids,
            'query_cross_camera_positive_counts': positives,
            'query_camera_counts': {str(c): sum(rows[i]['camera'] == c for i in query) for c in range(1, 7)},
            'gallery_camera_counts': {str(c): sum(rows[i]['camera'] == c for i in gallery) for c in range(1, 7)},
        }
    if assigned != set(range(len(rows))):
        raise AssertionError('Not every training image is covered exactly once')
    if set(partition_ids['fit']) & set(partition_ids['val']):
        raise AssertionError('Identity leakage between fit and validation')
    return rows, {
        'protocol': PROTOCOL, 'seed': seed, 'split_rng': 'Python random.Random(seed).shuffle(sorted identities)',
        'query_rule': 'First basename per identity-camera with >=2 images; if all cameras are singletons, choose camera at pid % camera_count in sorted camera order; all other images are gallery',
        'image_order': 'All training basenames sorted lexicographically; subset indices preserve this order',
        'image_count': len(rows), 'identity_count': len(ids), 'partitions': partitions,
        'checks': {'identity_disjoint': True, 'query_gallery_disjoint': True,
                   'all_training_images_covered_once': True, 'all_queries_have_cross_camera_positive': True},
        'supervision_boundary': {
            'backbones_frozen': True, 'backbones_trained_on_all_751_official_train_identities': True,
            'backbone_has_seen_both_fit_and_val_identities': True,
            'reranker_identity_holdout': True, 'end_to_end_unseen_identity_validation': False,
            'test_split_used_for_parameter_selection': False,
        },
    }


def write_split(output, rows, split, origin):
    output.mkdir(parents=True, exist_ok=True)
    row_copy = [dict(row) for row in rows]
    for partition, info in split['partitions'].items():
        for role in ('query', 'gallery'):
            for index in info[role + '_indices']:
                row_copy[index].update(partition=partition, role=role)
            write_once(output / f'{partition}.{role}.txt', '\n'.join(info[role + '_paths']) + '\n')
        write_once(output / f'{partition}.ids.txt', '\n'.join(map(str, info['identity_ids'])) + '\n')
    manifest = dict(split, source=origin, records=row_copy,
                    extraction_script_sha256=sha256(__file__), python=platform.python_version())
    write_once(output / 'split-manifest.json', canonical_json(manifest))
    return manifest


class Images:
    """DataLoader-compatible, pickleable dataset; no model imports in split mode."""
    def __init__(self, rows, origin, transform):
        self.rows, self.origin, self.transform = rows, origin, transform
        self._zip = None

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_zip'] = None
        return state

    def __getitem__(self, index):
        from PIL import Image
        row = self.rows[index]
        if self.origin['kind'] == 'zip':
            if self._zip is None:
                self._zip = zipfile.ZipFile(self.origin['path'])
            source = io.BytesIO(self._zip.read(row['source']))
        else:
            source = Path(self.origin['path']) / row['source']
        with Image.open(source) as image:
            return self.transform(image.convert('RGB'))


def validate_model_files(args, model_name):
    spec = MODELS[model_name]
    prefix = 'resnet' if model_name == 'resnet50' else 'osnet'
    source = getattr(args, prefix + '_source')
    checkpoint = getattr(args, prefix + '_checkpoint')
    if source is None or checkpoint is None:
        raise ValueError(f'{model_name} requires --{prefix}-source and --{prefix}-checkpoint')
    source, checkpoint = source.resolve(strict=True), checkpoint.resolve(strict=True)
    for path, key in ((source, 'model_source_sha256'), (checkpoint, 'checkpoint_sha256')):
        actual = sha256(path)
        if actual != spec[key]:
            raise ValueError(f'{model_name} {key} differs from historical Market extraction: {actual}')
    historical = getattr(args, prefix + '_history_manifest')
    historical_evidence = None
    if historical is not None:
        historical = historical.resolve(strict=True)
        data = json.loads(historical.read_text())
        for key in ('model_source_sha256', 'checkpoint_sha256'):
            if data.get(key) != spec[key]:
                raise ValueError(f'Historical manifest {key} mismatch: {historical}')
        if data.get('feature_sha256') != spec['historical_test_feature_sha256']:
            raise ValueError(f'Historical test feature SHA mismatch: {historical}')
        historical_evidence = {'path': str(historical), 'sha256': sha256(historical),
                               'preprocessing': data.get('preprocessing')}
    return source, checkpoint, historical_evidence


def extract_model(args, model_name, rows, split, origin, model_files):
    import numpy as np
    import scipy
    from scipy.io import savemat, loadmat
    import torch
    import torchvision
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import PIL

    spec = MODELS[model_name]
    source, checkpoint, historical_evidence = model_files
    output = args.output_dir / model_name
    if output.exists():
        raise FileExistsError(f'Model output exists: {output}; use a fresh output directory or select only unfinished models')
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Full feature extraction requires a CUDA GPU; split-only does not')
    device = torch.device('cuda:0' if args.device == 'cuda' else args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    torch.set_num_threads(spec['torch_threads'])
    torch.backends.cudnn.benchmark = False
    module_spec = importlib.util.spec_from_file_location('market_frozen_' + model_name, source)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    # Match the historical safe/trusted loading policy. ResNet's legacy pickle
    # metadata is permitted only AFTER the exact official checkpoint SHA check.
    if model_name == 'resnet50':
        loaded = torch.load(checkpoint, map_location='cpu', weights_only=False, encoding='latin1')
    else:
        loaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = loaded.get('state_dict', loaded)
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    if tuple(state['classifier.weight'].shape) != (EXPECTED_IDENTITIES, spec['feature_dim']):
        raise ValueError(f'Wrong classifier dimensions for {model_name}')
    model = getattr(module, spec['constructor'])(num_classes=EXPECTED_IDENTITIES, pretrained=False)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    transform = transforms.Compose([
        transforms.Resize((256, 128), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(PREPROCESSING['mean'], PREPROCESSING['std']),
    ])
    loader = DataLoader(Images(rows, origin, transform), batch_size=args.batch_size,
                        num_workers=args.workers, shuffle=False, pin_memory=True)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    features = np.empty((len(rows), spec['feature_dim']), dtype=np.float32)
    completed = 0
    with torch.inference_mode():
        for images in loader:
            raw = model(images.to(device, non_blocking=True))
            if raw.ndim != 2 or raw.shape[1] != spec['feature_dim'] or not bool(torch.isfinite(raw).all()):
                raise ValueError('Invalid feature dimensions or nonfinite values')
            if bool((torch.linalg.vector_norm(raw, dim=1) <= 1e-12).any()):
                raise ValueError('Zero feature vector')
            block = torch.nn.functional.normalize(raw, p=2, dim=1).cpu().numpy().astype(np.float32, copy=False)
            features[completed:completed + len(block)] = block
            completed += len(block)
            print(f'{model_name}: {completed}/{len(rows)}', flush=True)
    if completed != len(rows) or not np.isfinite(features).all():
        raise AssertionError('Incomplete feature extraction')
    if not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5, rtol=0):
        raise AssertionError('Features are not L2 normalized')
    extraction_seconds = time.monotonic() - start
    labels = np.asarray([row['pid'] for row in rows], dtype=np.int32)
    cameras = np.asarray([row['camera'] for row in rows], dtype=np.int32)
    names = np.asarray([row['name'] for row in rows])
    output.mkdir()
    arrays = {'features': features, 'labels': labels, 'cameras': cameras, 'paths': names,
              'image_sha256': np.asarray([row['sha256'] for row in rows])}
    for partition, info in split['partitions'].items():
        for role in ('query', 'gallery'):
            arrays[partition + '_' + role + '_indices'] = np.asarray(info[role + '_indices'], dtype=np.int64)
    npz_temp = output / 'train-features.partial.npz'
    np.savez_compressed(npz_temp, **arrays)
    # Verify persisted feature values and indices before publishing the artifacts.
    with np.load(npz_temp, allow_pickle=False) as saved:
        for key, value in arrays.items():
            if not np.array_equal(saved[key], value):
                raise AssertionError(f'NPZ round-trip mismatch: {key}')
    artifact_records = {}
    for partition, info in split['partitions'].items():
        fields = {}
        for role in ('query', 'gallery'):
            indices = np.asarray(info[role + '_indices'], dtype=np.int64)
            fields[role + '_f'] = features[indices]
            fields[role + '_label'] = labels[indices]
            fields[role + '_cam'] = cameras[indices]
        temp = output / (partition + '.partial.mat')
        savemat(temp, fields, do_compression=False, appendmat=False)
        saved = loadmat(temp)
        for key, value in fields.items():
            expected = value if value.ndim == 2 else value.reshape(1, -1)
            if not np.array_equal(saved[key], expected):
                raise AssertionError(f'MAT round-trip mismatch: {partition}/{key}')
        final = output / (partition + '.mat')
        temp.rename(final)
        artifact_records[partition] = {
            'path': str(final), 'sha256': sha256(final),
            'identity_ids': info['identity_ids'], 'query_paths': info['query_paths'],
            'gallery_paths': info['gallery_paths'],
            'query_count': info['query_count'], 'gallery_count': info['gallery_count'],
            'shapes': {key: list(value.shape) for key, value in fields.items()},
        }
    npz_final = output / 'train-features.npz'
    npz_temp.rename(npz_final)
    manifest = {
        'protocol': PROTOCOL, 'model': model_name, 'constructor': spec['constructor'],
        'num_classes': EXPECTED_IDENTITIES, 'feature_dim': spec['feature_dim'],
        'model_source': str(source), 'checkpoint': str(checkpoint),
        'model_source_sha256': sha256(source), 'checkpoint_sha256': sha256(checkpoint),
        'script_sha256': sha256(__file__),
        'split_manifest_sha256': sha256(args.output_dir / 'split-manifest.json'),
        'historical_test_feature_sha256': spec['historical_test_feature_sha256'],
        'historical_manifest_evidence': historical_evidence,
        'upstream_commit': 'f8cd150fdf77e8d9e1ed143b7f308c2c609ded50',
        'preprocessing': PREPROCESSING, 'supervision_boundary': split['supervision_boundary'],
        'train_features': {'path': str(npz_final), 'sha256': sha256(npz_final),
                           'shape': list(features.shape), 'npz_keys': sorted(arrays)},
        'partitions': artifact_records,
        'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                        'torchvision': str(torchvision.__version__), 'cuda': torch.version.cuda,
                        'numpy': np.__version__, 'scipy': scipy.__version__, 'pillow': PIL.__version__,
                        'gpu': torch.cuda.get_device_name(device), 'device': str(device)},
        'batch_size': args.batch_size, 'workers': args.workers,
        'torch_threads': spec['torch_threads'], 'extraction_seconds': extraction_seconds,
        'total_seconds': time.monotonic() - start,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'validation': {'all_features_finite': True, 'all_l2_norms_unit_atol_1e-5': True,
                       'mat_npz_roundtrip_exact': True, 'no_training_or_test_metrics': True},
    }
    write_once(output / 'manifest.json', canonical_json(manifest))
    for partition, record in artifact_records.items():
        write_once(output / (partition + '.manifest.json'), canonical_json({
            'model': model_name, 'protocol': PROTOCOL, 'partition': partition,
            **record, 'feature_sha256': record['sha256'], 'preprocessing': PREPROCESSING,
            'checkpoint_sha256': spec['checkpoint_sha256'],
            'model_source_sha256': spec['model_source_sha256'],
            'split_manifest_sha256': manifest['split_manifest_sha256'],
            'supervision_boundary': split['supervision_boundary'],
        }))
    print(json.dumps({'model': model_name, 'status': 'complete', 'features': list(features.shape),
                      'seconds': extraction_seconds, 'manifest': str(output / 'manifest.json')}, indent=2), flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--archive', type=Path, help='Historical Market-1501-v15.09.15.zip (SHA pinned)')
    group.add_argument('--dataset', type=Path, help='Unpacked Market root or bounding_box_train directory')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--split-only', action='store_true', help='Validate/hash/split using only standard-library dependencies')
    parser.add_argument('--models', nargs='+', choices=sorted(MODELS), default=['resnet50', 'osnet'])
    for prefix in ('resnet', 'osnet'):
        parser.add_argument(f'--{prefix}-source', type=Path, help='Exact model .py source used for historical test extraction')
        parser.add_argument(f'--{prefix}-checkpoint', type=Path, help='Exact frozen Market checkpoint (SHA pinned)')
        parser.add_argument(f'--{prefix}-history-manifest', type=Path, help='Optional archived test manifest for additional cross-check')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.batch_size <= 0 or args.workers < 0 or len(set(args.models)) != len(args.models):
        parser.error('Require positive batch size, nonnegative workers, and unique model names')
    model_files = {} if args.split_only else {name: validate_model_files(args, name) for name in args.models}
    rows, origin = scan_training_images(args.archive, args.dataset)
    rows, split = build_split(rows)
    write_split(args.output_dir, rows, split, origin)
    print(json.dumps({'protocol': PROTOCOL, 'training_images': len(rows),
                      'partitions': {key: {field: value[field] for field in
                                         ('identity_count', 'query_count', 'gallery_count', 'image_count')}
                                     for key, value in split['partitions'].items()},
                      'split_only': args.split_only}, indent=2), flush=True)
    if args.split_only:
        return
    for model_name in args.models:
        extract_model(args, model_name, rows, split, origin, model_files[model_name])


if __name__ == '__main__':
    main()
