set -e
A=/home/abc/workspace/bayes_occ_mpc/results/acados_v026_sicnav
if [ ! -d "$A/.git" ]; then
  git clone https://github.com/acados/acados.git "$A"
fi
cd "$A"
git checkout -q 285d382
git submodule update --recursive --init --quiet
mkdir -p build && cd build
cmake -DACADOS_WITH_QPOASES=ON -DACADOS_PYTHON=ON -DCMAKE_BUILD_TYPE=Release ..
make -j6 install
cd "$A"
# tera renderer，同 0.0.34
mkdir -p bin
curl -fL --retry 3 -o bin/t_renderer https://github.com/acados/tera_renderer/releases/download/v0.0.34/t_renderer-v0.0.34-linux
chmod +x bin/t_renderer
/home/abc/miniconda3/envs/crowdnav/bin/pip install -e "$A/interfaces/acados_template"
echo "=== BUILD_OK $(git rev-parse HEAD) ==="
