"""Fetch pinned author sources with SHA-256 verification; no model/data downloads."""
import argparse
import json
import urllib.request
from pathlib import Path

from common import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', choices=['kr', 'ecn', 'gcr'], default=['kr', 'ecn'])
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / 'upstream_sources.json').read_text())
    for method in args.methods:
        for record in manifest[method]['files'].values():
            target = here / record['destination']
            if target.exists():
                if sha(target) != record['sha256']:
                    raise ValueError('Existing upstream file has different contents: ' + str(target))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + '.download')
            with urllib.request.urlopen(record['url'], timeout=60) as source, temporary.open('wb') as dest:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    dest.write(block)
            if sha(temporary) != record['sha256']:
                temporary.unlink()
                raise ValueError('Downloaded upstream source hash mismatch')
            temporary.replace(target)
            print('Verified:', target)


if __name__ == '__main__':
    main()
