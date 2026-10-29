#!/bin/bash
# Run only after lambda2 acceptance/reflection, fresh resource and source checks.
# It never retries, resumes, archives, or queues a population experiment.
set -u
cd -- "$(dirname -- "$0")"
test ! -e launch.log || exit 90
exec >launch.log 2>&1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/root/miniconda3/bin/python -B -u run_fixed.py >compute.log 2>&1
compute_code=$?
printf '%s\n' "$compute_code" >compute-wrapper.exit
if [ "$compute_code" -ne 0 ]; then
    printf '%s\n' "$compute_code" >total.exit
    exit "$compute_code"
fi
/root/miniconda3/bin/python -B -u accept_fixed.py >acceptance.log 2>&1
accept_code=$?
printf '%s\n' "$accept_code" >acceptance-wrapper.exit
printf '%s\n' "$accept_code" >total.exit
exit "$accept_code"
