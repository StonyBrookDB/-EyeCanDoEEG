#!/usr/bin/env bash
# One-time setup for src/eyecando/kenlm_lm.py: builds the `kenlm` Python
# bindings against this project's Python (3.14), and downloads the
# character 12-gram model kenlm_lm.py expects at models/lm_dec19_char_large_12gram.kenlm.
#
# Why this isn't just `pip install kenlm`:
#   PyPI's kenlm==0.3.0 (and the same code at kpu/kenlm's git HEAD) ships a
#   pre-generated Cython .cpp file that doesn't compile against Python
#   3.14's C API (removed/changed internals: _PyLong_AsByteArray's
#   signature, _PyGen_SetStopIterationValue). The fix is mechanical --
#   regenerate that .cpp from the checked-in .pyx source with a current
#   Cython, which knows about 3.14 -- but it's not something `pip install`
#   does on its own, and the project's own default KENLM_MAX_ORDER=6
#   would also reject this specific model (it's a 12-gram) without the
#   MAX_ORDER=12 override below.
#
# Re-run this if the .venv is ever recreated -- the built kenlm wheel is
# NOT a normal pyproject.toml dependency (see pyproject.toml's own
# comment) precisely because it needs this non-standard build.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$(pwd)"
MODEL_PATH="models/lm_dec19_char_large_12gram.kenlm"

# Every pip/python3 call below resolves via $PATH -- if .venv isn't
# active, this builds and installs kenlm against whatever Python happens
# to be first on $PATH instead, and the "Verifying" step at the end would
# then pass using that SAME wrong interpreter, silently leaving .venv's
# own Python still unable to import kenlm.
if [ -z "${VIRTUAL_ENV:-}" ] || [ "$VIRTUAL_ENV" != "$REPO_ROOT/.venv" ]; then
  echo "error: activate this project's .venv first (source .venv/bin/activate)," >&2
  echo "       then re-run this script -- see README.md's Setup section." >&2
  exit 1
fi

if [ -f "$MODEL_PATH" ]; then
  echo "Model already present at $MODEL_PATH, skipping download."
else
  echo "Downloading lm_dec19_char_large_12gram.kenlm.gz (~231MB) from OSF..."
  curl -L -o "$MODEL_PATH.gz" "https://osf.io/download/c6mnz/"
  echo "Decompressing..."
  gunzip "$MODEL_PATH.gz"
fi

if python3 -c "import kenlm" 2>/dev/null; then
  echo "kenlm already importable, skipping build."
else
  command -v cmake >/dev/null || { echo "Installing cmake via Homebrew..."; brew install cmake; }
  pip install cython

  BUILD_DIR="$(mktemp -d)"
  echo "Cloning kenlm into $BUILD_DIR..."
  git clone --depth 1 https://github.com/kpu/kenlm.git "$BUILD_DIR/kenlm_src"
  echo "Regenerating python/kenlm.cpp with a current Cython (fixes the Python 3.14 build)..."
  (cd "$BUILD_DIR/kenlm_src/python" && cython --cplus -3 kenlm.pyx)
  echo "Building with MAX_ORDER=12 (this model is a 12-gram; kenlm's own default caps at 6)..."
  (cd "$BUILD_DIR/kenlm_src" && MAX_ORDER=12 pip install --no-build-isolation .)
  rm -rf "$BUILD_DIR"
fi

echo "Verifying..."
python3 -c "
import kenlm
m = kenlm.Model('$MODEL_PATH')
assert m.order == 12, f'expected order 12, got {m.order}'
print('OK: kenlm importable, model loads, order =', m.order)
"
