#!/bin/bash
# Detached, resumable local execution. Never infer completion from a log banner.
set -eo pipefail
source /home/abc/temp/modern/env.sh
cd /home/abc/temp/modern
export PYTHONUNBUFFERED=1
export WORKERS=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_THREAD_LIMIT=4
exec "$PYTHON_CROWDNAV" -u -B /home/abc/temp/modern/supervise.py "$@"
