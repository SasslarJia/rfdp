import argparse, os, sys, json, hashlib, zipfile, re, io, time, importlib.util
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.io import savemat
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
B = None
ASSETS = None
MODEL_SOURCE = None

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def imp(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

class Images(Dataset):

    def __init__(self, names):
        self.names = names
        self.z = None
        self.tf = transforms.Compose([transforms.Resize((256, 128)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        if self.z is None:
            self.z = zipfile.ZipFile(ASSETS / 'DukeMTMC-reID.zip')
        with Image.open(io.BytesIO(self.z.read(self.names[i]))) as im:
            return self.tf(im.convert('RGB'))

def legacy_scalar(*args):
    return np.core.multiarray.scalar(*args)
legacy_scalar.__module__ = 'numpy.core.multiarray'
legacy_scalar.__name__ = 'scalar'

def main():
    global B, ASSETS, MODEL_SOURCE
    p = argparse.ArgumentParser(description='Frozen author Duke ResNet50/OSNet feature extraction')
    p.add_argument('--assets', type=Path, required=True, help='Duke zip, two checkpoints and resnet.py')
    p.add_argument('--osnet-source', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--download-manifest', type=Path, default=Path(__file__).with_name('duke-download-manifest.json'))
    args = p.parse_args()
    B = args.output_dir
    ASSETS = args.assets
    MODEL_SOURCE = args.osnet_source
    B.mkdir(parents=True, exist_ok=True)
    if sha(MODEL_SOURCE) != 'c7c1c29187d6330f859c91da229271531920464c7011aec13842a086b2263cae':
        raise ValueError('OSNet model source differs from archived source')
    torch.manual_seed(0)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available()
    src = json.loads(args.download_manifest.read_text())
    for (n, h) in src['asset_hashes'].items():
        assert sha(ASSETS / n) == h, n
    z = zipfile.ZipFile(ASSETS / 'DukeMTMC-reID.zip')
    names = z.namelist()
    splits = {}
    validation = {}
    pidsets = {}
    filehash = {}
    for (name, count) in [('query', 2228), ('bounding_box_test', 17661), ('bounding_box_train', 16522)]:
        rows = sorted((n for n in names if n.startswith('DukeMTMC-reID/' + name + '/') and n.endswith('.jpg') and (not Path(n).name.startswith('._'))))
        assert len(rows) == count and len(set(rows)) == count, (name, len(rows))
        ids = []
        cams = []
        sizes = {}
        for n in rows:
            m = re.fullmatch('(\\d+)_c([1-8])_f\\d+\\.jpg', Path(n).name)
            assert m, n
            ids.append(int(m[1]))
            cams.append(int(m[2]))
            raw = z.read(n)
            filehash[n] = hashlib.sha256(raw).hexdigest()
            with Image.open(io.BytesIO(raw)) as im:
                im.load()
                sizes[str(im.size)] = sizes.get(str(im.size), 0) + 1
        splits[name] = rows
        pidsets[name] = set(ids)
        validation[name] = dict(count=count, identities=len(set(ids)), cameras=sorted(set(cams)), sizes=sizes)
    assert not pidsets['bounding_box_train'] & (pidsets['query'] | pidsets['bounding_box_test'])
    assert pidsets['query'] <= pidsets['bounding_box_test'] and len(pidsets['query']) == 702
    (B / 'dataset-validation.json').write_text(json.dumps(dict(splits=validation, train_test_identity_disjoint=True, image_sha256=filehash), indent=2))
    print('DATASET_VALIDATED', validation, flush=True)
    (B / 'features').mkdir(exist_ok=True)
    for (modelname, filename, source, ctor, dim) in [('resnet50', 'resnet50-duke.pth.tar', ASSETS / 'resnet.py', 'resnet50', 2048), ('osnet', 'osnet-duke.pth', MODEL_SOURCE, 'osnet_x1_0', 512)]:
        target = B / 'features' / f'{modelname}.mat'
        if target.exists():
            raise FileExistsError(target)
        with torch.serialization.safe_globals([legacy_scalar, np.dtype, type(np.dtype('float64')), type(np.dtype('float32'))]):
            weights = torch.load(ASSETS / filename, map_location='cpu', weights_only=True, encoding='latin1')
        state = weights.get('state_dict', weights)
        state = {k.removeprefix('module.'): v for (k, v) in state.items()}
        print(modelname, 'checkpoint keys', list(weights)[:5], list(state)[-5:], flush=True)
        model = getattr(imp(modelname, source), ctor)(num_classes=702, pretrained=False)
        model.load_state_dict(state, strict=True)
        model.eval().cuda()
        fields = {}
        t = time.monotonic()
        for (subset, folder) in [('query', 'query'), ('gallery', 'bounding_box_test')]:
            rows = splits[folder]
            out = []
            with torch.inference_mode():
                for batch in DataLoader(Images(rows), batch_size=128, num_workers=2, pin_memory=True, shuffle=False):
                    f = model(batch.cuda(non_blocking=True))
                    assert f.shape[1] == dim and torch.isfinite(f).all()
                    assert (f.norm(dim=1) > 0).all()
                    out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
            fields[subset + '_f'] = np.concatenate(out).astype(np.float32)
            fields[subset + '_label'] = np.array([int(Path(n).name.split('_')[0]) for n in rows])
            fields[subset + '_cam'] = np.array([int(re.search('_c([1-8])', n)[1]) for n in rows])
            print(modelname, subset, fields[subset + '_f'].shape, flush=True)
        savemat(target, fields)
        manifest = dict(feature_sha256=sha(target), weights_sha256=sha(ASSETS / filename), model_source_sha256=sha(source), model=modelname, protocol='author DukeMTMC-reID training split checkpoint; full test; frozen weights', preprocessing='RGB resize256x128 bilinear antialias; ImageNet mean/std; no flip; L2 float32', query_paths=splits['query'], gallery_paths=splits['bounding_box_test'], seconds=time.monotonic() - t, torch=torch.__version__, cuda=torch.version.cuda)
        target.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2))
        del model
        torch.cuda.empty_cache()
if __name__ == '__main__':
    main()
