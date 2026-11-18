#!/usr/bin/env bash
# Included at the root of the MSMT17 deployment bundle.
set -eu
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: bash run-step1.sh DATASET_DIR OUTPUT_DIR [V2|V1]" >&2
    exit 2
fi
bundle_dir="$(cd "$(dirname "$0")" && pwd)"
(cd "$bundle_dir" && sha256sum -c SHA256SUMS)
python3 -m pip install --no-index --find-links "$bundle_dir/wheels" yacs==0.1.8 PyYAML==6.0.2
export PYTHONPATH="$bundle_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
exec python3 -m experiments.msmt17.extract run --data "$1" --out "$2" \
    --version "${3:-V2}" --assets "$bundle_dir/assets"
