#!/bin/bash
# Put the current bridge source and solver-diagnostics accessors into every
# comparator workspace, then rebuild.  One command, so the six workspaces cannot
# drift apart: a bridge built from a different source in one of them would make
# its numbers incomparable with the rest.
set -euo pipefail
B=/home/abc/workspace/bayes_occ_mpc/build_modern
SRC=/home/abc/temp/modern/bridge/mpc_bridge.cpp
HDR_PATCH=/home/abc/temp/modern/patch_solver_header.py
. /home/abc/temp/modern/env.sh
export CC=/usr/bin/gcc-9 CXX=/usr/bin/g++-9
export PKG_CONFIG_PATH="$MODERN_GSL/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export CPLUS_INCLUDE_PATH="$MODERN_GSL/include:${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="$MODERN_GSL/include:${C_INCLUDE_PATH:-}"
export LIBRARY_PATH="$MODERN_GSL/lib:${LIBRARY_PATH:-}"

for ws in "$@"; do
  echo "=== $ws ==="
  cp "$SRC" "$B/$ws/src/mpc_planner_bridge/src/mpc_bridge.cpp"
  python3 "$HDR_PATCH" "$B/$ws/src/mpc_planner/mpc_planner_solver/include/mpc_planner_solver/acados_solver_interface.h"
  ( cd "$B/$ws" && catkin build mpc_planner_bridge -j6 --no-status ) \
      > "/home/abc/temp/modern/build_deploy_${ws}.log" 2>&1 \
    && echo "OK $ws" \
    || { echo "FAILED $ws"; tail -25 "/home/abc/temp/modern/build_deploy_${ws}.log"; exit 1; }
done
