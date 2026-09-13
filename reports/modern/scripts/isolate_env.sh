set -e
export PIP_EXTRA_INDEX_URL= PIP_TRUSTED_HOST= PIP_INDEX_URL=https://pypi.org/simple
CONDA=/home/abc/miniconda3/bin/conda
echo "=== 1) 克隆 crowdnav -> sicnav ==="
$CONDA create -y --clone crowdnav -n sicnav
S=/home/abc/miniconda3/envs/sicnav/bin/python
echo "=== 2) 恢复 crowdnav：卸掉我加的包，protobuf 退回 tensorflow 兼容版 ==="
C=/home/abc/miniconda3/envs/crowdnav/bin/python
$C -m pip uninstall -y tensorboardX orjson ncls dill distinctipy easydict future-fstrings acados-template || true
$C -m pip install -q "protobuf==3.19.6"
echo "--- crowdnav 校验 ---"
$C -c "import tensorflow as tf; print('  tensorflow OK', tf.__version__)" 2>&1 | tail -2
$C -c "import google.protobuf as p; print('  protobuf', p.__version__)"
echo "=== 3) sicnav 环境装 SICNav 依赖 ==="
$S -m pip install -q orjson ncls dill distinctipy easydict "tensorboardX==2.6.2.2"
A=/home/abc/workspace/bayes_occ_mpc/results/acados_v026_sicnav
$S -m pip install -q -e "$A/interfaces/acados_template"
echo "--- sicnav 校验 ---"
$S -c "
import importlib
for m in ('torch','casadi','gym','rvo2','orjson','ncls','dill','distinctipy','easydict','tensorboardX','acados_template'):
    try:
        mod=importlib.import_module(m); print(f'  ok {m:16s} {getattr(mod,\"__version__\",\"\")}')
    except Exception as e: print(f'  缺 {m}: {type(e).__name__}')"
echo "=== ISOLATE_OK ==="
