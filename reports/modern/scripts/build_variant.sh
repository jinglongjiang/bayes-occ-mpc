#!/bin/bash
# Build one prediction-horizon variant of a comparator workspace.
#
# The order requires the development candidates to cover the physical
# prediction horizon, and N enters at code-generation time: the acados solver
# is generated from settings.yaml, so a different N is a different binary and
# must live in its own workspace with its own generator configuration.  This
# replaces the one-off build_*.sh files; the preflight snapshot then hashes the
# generator config and the resulting .so together, which is the pairing that a
# hand-edited yaml over a stale binary would break.
#
#   build_variant.sh <source workspace> <new workspace> <N>
set -euo pipefail
SRC=$1; DST=$2; NEW_N=$3
B=/home/abc/workspace/bayes_occ_mpc/build_modern
LOG=/home/abc/temp/modern/build_${2##*/}.log

. /home/abc/temp/modern/env.sh

if [ -d "$B/$DST" ]; then echo "$DST already exists"; exit 0; fi
mkdir -p "$B/$DST"
cp -a "$B/$SRC/src" "$B/$DST/src"

CFG="$B/$DST/src/mpc_planner/mpc_planner_jackalsimulator/config/settings.yaml"
sed -i -E "s/^N: [0-9]+/N: $NEW_N/" "$CFG"
grep -E "^N: " "$CFG"

export CC=/usr/bin/gcc-9 CXX=/usr/bin/g++-9
# The v0.4.2 tree has no venv of its own (the system python3 lacks ensurepip),
# so codegen runs under the crowdnav interpreter with acados_template shadowed
# to v0.4.2 -- the same resolution the original two workspaces were built with.
export ACADOS_PYTHON=$PYTHON_CROWDNAV
export PYTHONPATH="$MODERN_ACADOS/interfaces/acados_template:${PYTHONPATH:-}"
export ACADOS_SOURCE_DIR="$MODERN_ACADOS"
# guidance_planner resolves GSL through pkg-config and then links the bare
# name `gsl`, so both the .pc directory and a plain libgsl.so in the workspace
# have to be in place; GSL is a local build because installing it system-wide
# needs a password this session does not have.
export PKG_CONFIG_PATH="$MODERN_GSL/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
# The package links the pkg-config libs but never adds its include dir.
export CPLUS_INCLUDE_PATH="$MODERN_GSL/include:${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="$MODERN_GSL/include:${C_INCLUDE_PATH:-}"
# ...and links the bare names `gsl`/`gslcblas`, so the local prefix has to be
# on the linker search path as well.
export LIBRARY_PATH="$MODERN_GSL/lib:${LIBRARY_PATH:-}"
mkdir -p "$B/$DST/devel/lib"
ln -sfn "$MODERN_GSL/lib/libgsl.so" "$B/$DST/devel/lib/libgsl.so"
ln -sfn "$MODERN_GSL/lib/libgslcblas.so" "$B/$DST/devel/lib/libgslcblas.so"

{
  echo "=== generate (N=$NEW_N) ==="
  cd "$B/$DST/src/mpc_planner/mpc_planner_jackalsimulator/scripts"
  "$ACADOS_PYTHON" generate_jackalsimulator_solver.py
  echo "=== catkin build ==="
  cd "$B/$DST"
  catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=/usr/bin/gcc-9 -DCMAKE_CXX_COMPILER=/usr/bin/g++-9 >/dev/null
  # Only the bridge and its dependency chain, the same 8 packages the two
  # registered workspaces were built with.  The three ROS-node packages
  # (jackal, jackalsimulator, rosnavigation) need a simulator stack this
  # evaluation does not use and the order rules out.
  catkin build mpc_planner_bridge -j6 --no-status
  ln -sfn "$MODERN_GSL/lib/libgsl.so" "$B/$DST/devel/lib/libgsl.so" 2>/dev/null || true
  ln -sfn "$MODERN_GSL/lib/libgslcblas.so" "$B/$DST/devel/lib/libgslcblas.so" 2>/dev/null || true
} >"$LOG" 2>&1

test -x "$B/$DST/devel/lib/mpc_planner_bridge/mpc_bridge" \
  && echo "OK  $DST  N=$NEW_N" \
  || { echo "FAILED $DST -- see $LOG"; tail -25 "$LOG"; exit 1; }
