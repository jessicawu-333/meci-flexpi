#!/usr/bin/env bash
# Download src/ of geyan21/flex-pi at the pinned commit (raw files, no clone) into $1/pristine and
# $1/patched, check the patch applies (dry run first), and print the commands to run the tests.
set -euo pipefail
SHA=044b8c2080a14a3d0082f81f612474643446faf3
OUT=${1:?usage: fetch_upstream.sh OUTDIR}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/pristine"
cd "$OUT/pristine"
curl -sSfL "https://api.github.com/repos/geyan21/flex-pi/git/trees/$SHA?recursive=1" \
  | python3 -c "import json,sys; [print(t['path']) for t in json.load(sys.stdin)['tree'] if t['type']=='blob' and t['path'].startswith('src/flexpi/') and t['path'].endswith('.py')]" \
  | xargs -P 16 -I{} curl -sSfL "https://raw.githubusercontent.com/geyan21/flex-pi/$SHA/{}" -o {} --create-dirs
rm -rf "$OUT/patched"; cp -r "$OUT/pristine" "$OUT/patched"; cd "$OUT/patched"
patch -p1 --dry-run < "$HERE/../patches/flexpi-meci.diff"
patch -p1 < "$HERE/../patches/flexpi-meci.diff"
echo "Run: FLEXPI_PRISTINE=$OUT/pristine FLEXPI_PATCHED=$OUT/patched $HERE/run_tests.sh"
