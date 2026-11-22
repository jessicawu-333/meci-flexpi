#!/usr/bin/env bash
# MECI x Flex-pi tests on CPU.
#   FLEXPI_PRISTINE: src/ tree of geyan21/flex-pi @ 044b8c2080a14a3d0082f81f612474643446faf3 (see fetch_upstream.sh)
#   FLEXPI_PATCHED : the same tree after `patch -p1 < ../patches/flexpi-meci.diff`
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
REF=${MECI_REF:-/tmp/flexpi_meci_ref.pt}
PYTHONPATH="$FLEXPI_PRISTINE/src:." $PY reference_pristine.py "$REF"
MECI_REF="$REF" PYTHONPATH="$FLEXPI_PATCHED/src:." $PY -m pytest -q -p no:cacheprovider -W ignore test_meci_flexpi.py -rA --tb=short -s
