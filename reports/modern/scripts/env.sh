# The one registered runtime environment for every modern-baseline task.
#
# This preamble was previously copy-pasted into each launcher, which is exactly
# the drift the execution order forbids: a bridge that resolves a different
# `.so` or a different `rosparam` tree is a different experiment.  Source this
# instead, and the preflight snapshot's hashes describe what actually ran.
#
#   . /home/abc/temp/modern/env.sh
set +u
export MODERN_ACADOS=/home/abc/workspace/bayes_occ_mpc/results/acados_v042_mpcplanner
export MODERN_GSL=/home/abc/workspace/bayes_occ_mpc/build_modern/gsl
export MODERN_BUILD=/home/abc/workspace/bayes_occ_mpc/build_modern
export MODERN_ROOT=/home/abc/workspace/bayes_occ_mpc
export MODERN_OUT=/home/abc/temp/modern

source /opt/ros/noetic/setup.bash
source "$MODERN_BUILD/tmpc/devel/setup.bash"

export ROS_MASTER_URI=http://localhost:11311
export MODERN_BRIDGE_LOG_DIR="$MODERN_OUT/bridge_logs"
# guidance_planner exports a bare `gsl`, and SH-MPC's rospack lookup needs the
# scenario_module workspace on the search path as well as T-MPC++'s.
export ROS_PACKAGE_PATH="$MODERN_BUILD/shmpc/devel/share:$MODERN_BUILD/shmpc/src:$ROS_PACKAGE_PATH"
export LD_LIBRARY_PATH="$MODERN_ACADOS/lib:$MODERN_GSL/lib:$MODERN_BUILD/tmpc/devel/lib:$MODERN_BUILD/shmpc/devel/lib:$LD_LIBRARY_PATH"
# Both packages read their parameters from the master, not from the yaml files
# directly, so the tree has to be loaded before any bridge starts.
for v in tmpc shmpc; do
  rosparam load "$MODERN_BUILD/$v/src/mpc_planner/mpc_planner_jackalsimulator/config/guidance_planner.yaml" 2>/dev/null
done
rosparam load "$MODERN_BUILD/shmpc/src/scenario_module/config/params.yaml" 2>/dev/null
export PYTHON_CROWDNAV=/home/abc/miniconda3/envs/crowdnav/bin/python
# Keep the source tree free of bytecode; the order requires caches elsewhere.
export PYTHONDONTWRITEBYTECODE=1
cd "$MODERN_ROOT"
