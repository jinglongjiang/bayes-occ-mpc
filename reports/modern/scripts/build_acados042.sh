set -e
A=/home/abc/workspace/bayes_occ_mpc/results/acados_v042_mpcplanner
# 克隆（低内存），等 JMID 的 cc1 退出后再编译，避免再次 OOM
[ -d "$A/.git" ] || git clone --branch v0.4.2 --depth 1 --recursive https://github.com/acados/acados.git "$A"
cd "$A"; git log -1 --format="clone_ok %H"
while pgrep -f "sicnav_mpc_constr_h_fun_jac_uxt_zt_hess" > /dev/null; do sleep 20; done
echo "=== JMID 编译已结束，开始编 acados v0.4.2 ==="
mkdir -p build && cd build
cmake -DACADOS_WITH_QPOASES=ON -DACADOS_PYTHON=ON -DCMAKE_BUILD_TYPE=Release ..
make -j4 install
cd "$A"; mkdir -p bin
curl -fL --retry 3 -o bin/t_renderer https://github.com/acados/tera_renderer/releases/download/v0.0.34/t_renderer-v0.0.34-linux
chmod +x bin/t_renderer
python3 -m venv "$A/venv"
"$A/venv/bin/pip" -q install --index-url https://pypi.org/simple "numpy<1.24" scipy casadi==3.6.5 pyyaml
"$A/venv/bin/pip" -q install --index-url https://pypi.org/simple -e "$A/interfaces/acados_template"
echo "=== ACADOS042_OK $(git rev-parse HEAD) ==="
