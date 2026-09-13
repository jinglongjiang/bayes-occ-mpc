# 四组现代方法执行状态

依据 `temp/ORDER_modern_baselines_CC.md`。本文件只记录已经在本机实际验证过的事实。
日期 2026-09-08。

## 一行状态

| 方法 | 状态 | 依据 |
|---|---|---|
| SICNav-np（classic） | `NATIVE_OK` | 5 人 circle，test_case 0，成功，nav_time 4.5 s，0 碰撞，18 s 跑完 |
| SICNav-CVG | `NATIVE_OK` | 5 人 circle，test_case 0，成功，nav_time 3.25 s，0 碰撞，0 too_close |
| SICNav-JMID | `NATIVE_OK` | 5 人 circle，test_case 0，成功，nav_time 2.75 s，0 碰撞，0 too_close，frozen_freq 0 |
| T-MPC++ | `BUILD_OK` | 8 个 catkin 包全部编译通过，`GuidanceConstraints` |
| SH-MPC | `BUILD_OK` | 9 个 catkin 包全部编译通过，`ScenarioConstraints` + `libscenario_module.so` |
| BNBRL+ | `BLOCKED_ASSET` | 导航策略权重在本机和上游均不存在 |

`BUILD_OK` 不等于 `NATIVE_OK`：库编出来了，控制器还没跑过一步。headless bridge 未实现。

## 开工快照

四个外部仓库的 HEAD 与指令记录一致：

| 仓库 | HEAD | 工作区 |
|---|---|---|
| mpc_planner | `3002da6a5e43b3577f584381bbcad569c82a381b` | 干净 |
| scenario_module | `c11addbb7239752d585beb181e8f9197893c6c53` | 干净 |
| BNBRLplus | `1bcdcc13e8609b24c61bdd2b4597b947ae235db9` | 干净 |
| safe-interactive-crowdnav | `c702fb8ac9ba6439ca61da7dde68b8524bbc6a1f` | 1 个改动文件（既有延迟导入补丁） |

本轮新增克隆：

| 仓库 | HEAD |
|---|---|
| ros_tools | `f2ff45830961db8d3d813ee4769f27eea708e989` |
| guidance_planner | `2c4188371e18e2fb3d083e0867b5e4d537a42860` |
| DecompUtil | `b0836c7228d19f0fa97282c584b55adf642279da` |

## 求解器与依赖

### acados v0.4.0（SH / T-MPC 用）

既有构建记录 `results/modern_solver_setup_20260906_2114`，commit `43354449192aa51557abda37d1bc17d907077354`，
tag v0.4.0。该记录标记为 `dependencies_ready`，但只做过 import 冒烟，**代码生成从未跑通**：
`bin/t_renderer` 缺失，任何 `AcadosOcpSolver` 构造都会在模板渲染处停下并等待交互输入。

已补：`t_renderer` v0.0.34 linux，sha256
`390063f34a8e13620564b4a136012270168e1421dd7920a747048749e1d99718`。

补上之后用 acados 自带的 `examples/acados_python/getting_started/minimal_example_ocp.py`
验收：10 次 SQP 迭代收敛，`res_stat` 3.86e-07，退出码 0。工具链（代码生成 → 编译 → 求解）成立。

venv：`results/modern_solver_setup_20260906_2114/venv/bin/python`，Python 3.8.20，
casadi 3.6.7，numpy 1.23.5，scipy 1.10.1，本轮加装 pyyaml 6.0.3（solver_generator 需要）。

### acados v0.2.6（SICNav-Diffusion 用）

README 要求 v0.2.6 commit `285d382`，与上面的 v0.4.0 不同。已在
`results/acados_v026_sicnav` 独立构建，`-DACADOS_PYTHON=ON -DACADOS_WITH_QPOASES=ON`，
HEAD `285d382b6c6d59c0983644caf6ed2924ed2153cb`，
`acados_template` 以 editable 方式装进 `miniconda3/envs/crowdnav`。
两套 acados 通过每进程的 `ACADOS_SOURCE_DIR` / `LD_LIBRARY_PATH` 隔离，未互相覆盖。

### GSL

`guidance_planner` 的 cmake 在 `pkg_check_modules(gsl)` 处失败，是核心构建链上唯一缺的系统包。
本机 sudo 需要密码，因此不装 `libgsl-dev`，改为把 GSL 2.7.1 源码编到
`build_modern/gsl` 本地 prefix，不触碰系统目录。
`gsl-2.7.1.tar.gz` sha256 `dcb0fbd43048832b757ff9942691a8dd70026d5da0ff85601e52687f6deeb34b`。

`rosdep` 另外报缺 `ros-noetic-jackal-gazebo`、`ros-noetic-jackal-navigation`、
`ros-noetic-nav-core`、`ros-noetic-base-local-planner`、`ros-noetic-costmap-2d`、
`ros-noetic-costmap-converter`、`ros-noetic-derived-object-msgs`，以及无法解析的
`vicon_util`、`roadmap`。这些全部只被 `mpc_planner_jackalsimulator` 和
`mpc_planner_rosnavigation` 两个 ROS 节点包需要。按指令 §4.1「不盲装整个 GUI 仿真栈」，
这两个包不构建；headless bridge 直接调用 `mpc_planner` 核心库。

## SH / T-MPC 构建隔离

两个完全独立的 catkin 工作副本，各自一份源码、一份生成产物：

```
build_modern/tmpc/src/{mpc_planner,ros_tools,guidance_planner,DecompUtil}
build_modern/shmpc/src/{mpc_planner,ros_tools,guidance_planner,DecompUtil,scenario_module}
```

两边都已 `switch_to_ros.py 1`（脚本回报 already in ROS1 mode，即仓库本来就是 ROS1 布局）。
SH 副本的生成脚本改为 `configuration_safe_horizon(settings)`，T-MPC 副本保留
`configuration_tmpc(settings)`。

生成结果证明隔离生效：

| 副本 | modules.h 里实际实例化的约束模块 | definitions.h |
|---|---|---|
| tmpc | `MPCBaseModule`, `Contouring`, `GuidanceConstraints` | `GUIDANCE_CONSTRAINTS_TYPE EllipsoidConstraints` |
| shmpc | `MPCBaseModule`, `Contouring`, `ScenarioConstraints` | 权重多一项 `slack` |

**`solver.cmake` 两边都生成了。**此前认为缺该文件、怀疑需要 ForcesPro 授权，是没跑过代码生成，
不是授权问题。`settings.yaml` 里 `solver_settings.solver` 本来就是 `acados`，
`t-mpc.use_t-mpc++` 本来就是 `true`。

生成时的实际参数（**尚未按本项目协议改过**）：`N: 30`，`integrator_step: 0.2`，
`control_frequency: 20`，`max_obstacles: 12`，`reference_velocity: 2.0`。
按 §4.2、§5，正式组要改成 dt 0.25、4 秒时域、障碍容量覆盖 20 人，届时重新生成并重新登记。

### scenario_module 配置核对（§4.3）

- `launch/ros1_jackalsimulator.launch` 第 15 行的 `params.yaml` 加载行确实是注释掉的，bridge 必须显式加载。
- `safe_sampling.compute_automatically: true`，`confidence: 0.01`，`removal_count: 0`。
  在这个开关下 `sample_size: 10` 是「非自动时才用」的回退值，不是实际样本数；
  实际数由风险界算出。运行时必须把真实 scenario count 打进日志核对。
- `sample_distribution.propagate_covariance: false`。我们的 Bayes 是沿时域传播协方差的，
  这一项在公平比较里要如何设置需要在接口冻结前登记决定。
- `use_real_samples: false`、`binomial_distribution: false`，即用模型采样而非数据库。

## SICNav

### 环境

`miniconda3/envs/crowdnav/bin/python`，casadi 3.6.4，gym 0.15.7，torch 2.1.0+cu121，
本机 GPU RTX 3060。本轮加装 `orjson`、`ncls`、`dill`、`distinctipy`、`easydict`、`tensorboardX`。

pip 在该环境下会挂死：`~/.config/pip/pip.conf` 配了 `pypi.ngc.nvidia.com` 作为
extra-index-url，该主机不可达，pip 反复重试直到超时。用
`PIP_EXTRA_INDEX_URL= PIP_INDEX_URL=https://pypi.org/simple` 逐进程覆盖，未改用户全局配置。

### 对指令的两处更正

指令 §7.2 列的两项修复，在本机实际代码上不成立，因此没有改：

1. **`config.set(..., args.num_humans)` 传 int**。`simple_test.py` 用的是
   `configparser.RawConfigParser`，它不做值类型校验，传 int 不报错。
   （换成 `ConfigParser` 才会 `TypeError`。）
2. **campc.py 的 ma57**。第 435 行的 ma57 调用是**探测**本身：代码先用 ma57 解一个
   dummy 问题，`return_status != 'Solve_Succeeded'` 就把 `hsl_available` 置 false，
   正式 opts 里的 ma57 三项全部在 `if hsl_available:` 之内。本机 HSL 不存在，
   探测失败并打出显式警告，自动退到 IPOPT 默认的 MUMPS。这是仓库自带的、有日志的回退，
   不是静默失败，无需改代码。日志实证：
   `HSL solvers are not loading properly. MA57 linear solver not available for IPOPT, using default instead.`

### 实际做的修改

1. `simple_test.py`：新增 `--render`，默认关闭；文件末尾无条件的
   `env.render('video', video_file)` 与 `plt.show()` 收进该开关。
   实证 headless 生效：CVG/np 两次运行的输出目录只有 `.pkl`，没有 `.mp4`。
   （既有的延迟导入补丁保留，未被覆盖。）
2. `sicnav_diffusion/utils/mpc_utils/orca_casadi_new.py`：两处 `cs.nlpsol(..., 'blocksqp', ...)`
   之前加 `opts["linsol"] = "ldl"`。

   **这是登记在案的数值求解器替换。** blocksqp 内部默认用 HSL 的 MA27；本机
   `cs.Linsol.has_plugin('ma27')` 为 False，调用时报
   `libhsl.so: cannot open shared object file`，且这条路径没有自动回退。
   实测 casadi 3.6.4 可用的 linsol 插件里，blocksqp 只有 `ldl` 能跑通
   （`qr`/`ldl`/`csparse`/`mumps`/`lapacklu`/`symbolicqr` 逐个试过，只有 `ldl` 返回正确解）。
   算法仍是 blocksqp，只换内部稀疏线性求解。

### 已验收的 native 结果

均为 `--num_humans 5 --circle --test_case 0`，headless。

| 别名 | 配置 | 结果 |
|---|---|---|
| SICNav-np | `hum_model=orca_casadi_kkt`，`priviledged_info=False`（日志确认） | success 1，nav_time 4.5 s，碰撞 0，frozen_freq 0.056 |
| SICNav-CVG | `human_goal_cvmm=true`，`human_pred_MID=false`，`human_pred_MID_joint=true` | success 1，nav_time 3.25 s，碰撞 0，too_close 0 |
| SICNav-JMID | `human_goal_cvmm=false`，`human_pred_MID=true`，`human_pred_MID_joint=true`；`SICNAV_EXT_FUN_FLAGS=-O1` | success 1，nav_time 2.75 s，碰撞 0，too_close 0，frozen_freq 0 |

CVG 用的是新建的 `sicnav_diffusion/configs/policy_cvg.config`，原
`policy.config`（JMID 配置）未改动。

Diffusion 权重 sha256：

```
f0f6544cb53bcc2fda04cd5762755f1f7806e0141f2ff9b16444cd8c3f5832e3  sim_gen_sicnav_p_mid_cvg_epoch169.pt
4b52475f73380368acf8965fd1bdae17e03c898da077aa5c76a09feb5e8dea3f  sim_gen_sicnav_p_midjp_cvg_epoch121.pt
```

既有的 `results_sicnav-np/hallway_bottleneck_N_3/...tc_0.pkl` 已读取核对：
test_case 0，success 1，num_steps 15，nav_time 3.75，碰撞 0。是一次真实的成功运行，
但场景是 hallway_bottleneck N=3，与本项目六场景无关。

## BNBRL+：`BLOCKED_ASSET`

本机两份副本里唯一的权重是 GST 预测器：

```
100-gumbel_social_transformer-.../sj/checkpoint/epoch_100.pt   828521 字节
```

上游 `JinnnK/BNBRLplus` 在 commit `1bcdcc13` 的完整 tree（209 条目）里
**没有 `trained_models/` 目录，没有任何其它 `.pt`**；GitHub Releases 为空。
README 第 46 行写「`trained_models/` contains some pretrained models provided by us」，
与仓库内容不符。`test.py` 默认请求的 `trained_models/BNDNN/checkpoints/20200.pt` 不存在。

因此需要向作者索取：导航策略 checkpoint、对应的 `arguments`/config、训练人数、预测器版本。
按指令不用 AttnGraph 或 GST 权重顶替，不启动 PPO 训练。§6.2 的工程修复未做——
没有权重时改这些无法验证，等资产到位再一并处理。

## 未做的部分

- matched-unicycle 执行器与 Bayes 的动力学适配（§3）尚未开始。三个现代 MPC 都是
  二阶 unicycle，现有 Bayes 是全向，**不能并表**。
- headless C++ bridge（§4.3、§5.4）尚未实现。
- 观测/动作/人数/重置的接口测试（§10.1）尚未开始。
- 正式 case 集合尚未登记；未指定任何新测试 case。


---

# 追加：本轮后续（2026-09-08 晚）

## acados 版本对齐

`mpc_planner`（HEAD 2025-03-30）的 C++ 接口调用三参数
`ocp_nlp_get(solver, field, value)`；本机 v0.4.0（2024-09-16）是四参数
`ocp_nlp_get(config, solver, field, value)`，因此 `mpc_planner_solver` 编不过。

逐版核对上游头文件后确定 **v0.4.2 是第一个改成三参数的版本**：

| 版本 | 签名 |
|---|---|
| v0.4.1 | `ocp_nlp_get(ocp_nlp_config *config, ocp_nlp_solver *solver, ...)` |
| **v0.4.2 起** | `ocp_nlp_get(ocp_nlp_solver *solver, ...)` |

`mpc_planner` 用到的其余接口（`ocp_nlp_out_get`、`ocp_nlp_in_set`、
`ocp_nlp_solver_opts_set`、`ocp_nlp_eval_cost`）签名与 v0.4.2 逐个吻合。

已在 `results/acados_v042_mpcplanner` 独立构建 v0.4.2，HEAD
`d49d47e0f0c185c0e448671328c43ccda705eceb`，另配 t_renderer v0.0.34。
它的 `python3 -m venv` 步骤失败（系统 python3 缺 `ensurepip`），未新建环境，
改为用 `PYTHONPATH` 让既有 venv 优先解析到 v0.4.2 的 `acados_template`，已验证生效：

```
acados_template -> results/acados_v042_mpcplanner/interfaces/acados_template/acados_template/__init__.py
get_acados_path -> results/acados_v042_mpcplanner
```

现在本机共三套互不干扰的 acados：v0.2.6（SICNav-Diffusion）、v0.4.0（保留，
未删）、v0.4.2（mpc_planner）。靠每进程的 `ACADOS_SOURCE_DIR` / `LD_LIBRARY_PATH` /
`PYTHONPATH` 隔离。

## T-MPC++ / SH-MPC 构建通过

两个副本都已按 v0.4.2 重新生成求解器并完整编译：

| 副本 | 结果 | 约束模块 | 产物 |
|---|---|---|---|
| tmpc | 8 / 8 包通过 | `MPCBaseModule`, `Contouring`, `GuidanceConstraints` | 含 `libguidance_planner.so`、`libguidance_planner_homotopy.so` |
| shmpc | 9 / 9 包通过 | `MPCBaseModule`, `Contouring`, `ScenarioConstraints` | 含 `libscenario_module.so` |

两边的 `libmpc_planner_solver.so` 各自链接到自己副本的
`mpc_planner_solver/acados/Solver/libacados_ocp_solver_Solver.so`，隔离成立。

过程中需要解决的三件事，都不改上游源码：

1. **GSL 找不到**。sudo 需要密码，没装 `libgsl-dev`。GSL 2.7.1 源码编到
   `build_modern/gsl` 本地 prefix；`guidance_planner` 又把 `gsl` 以裸名放进
   `catkin_package(LIBRARIES)`，下游 `mpc_planner_modules` 解析不到，
   于是把 GSL 的 `.so` 软链进各工作空间的 `devel/lib`（已在链接搜索路径上）。
2. **`solver.cmake` 反推 acados 路径失败**。它用
   `find_library(... PATHS $ENV{LD_LIBRARY_PATH})` 定位 `libacados.so` 再推 include 路径；
   catkin 的 `devel/env.sh` 会按记录的基准环境重建 `LD_LIBRARY_PATH`，把我在外层
   export 的 acados 路径抹掉，于是 `acados_include_path` 变成 `/../include`。
   删掉 `devel/`、在设好环境后从零重建，让基准环境把 acados 记进去，即解决。
3. **OpenMP + GCC 8**。`guidance_constraints.cpp:280` 在
   `#pragma omp parallel for` 下用 range-based for，GCC 8 报
   `invalid type for iteration variable`。本机有 GCC 9，改用
   `-DCMAKE_C_COMPILER=gcc-9 -DCMAKE_CXX_COMPILER=g++-9` 后通过。
   **两个副本必须都用 GCC 9，正式比较时 Bayes 侧的编译器也要一并登记。**

## SICNav-JMID 打通

三个问题依次解决：

1. **编译 OOM**。5 人的 ORCA-KKT 精确 Hessian 生成 **25.4 MB 的 C 源码**，
   作者默认 `-O3 -ffast-math`。实测 `-O1` 峰值 **8.7 GB**、耗时 **9 分 42 秒**、
   产出 15 MB 的 `.o`；`-O3` 在 15 GB 内存的本机必被 OOM killer 杀掉
   （`cc: fatal error: Killed signal terminated program cc1`）。
   `campc_acados_opt.py` 的两处 `ext_fun_compile_flags` 改为
   `os.environ.get("SICNAV_EXT_FUN_FLAGS", "-O3 -ffast-math")`，
   **默认仍是作者原值**，只有显式传 `SICNAV_EXT_FUN_FLAGS=-O1` 才降。
   **登记：`-O1` 只改生成函数的执行速度，不改算法；用它测出的耗时不能与 `-O3` 的并列。**
2. **acados 缓存假成功**。上一次被 OOM 杀掉后留下一个没有 `.so`、只有两个 `.o`
   的缓存目录；再跑时 `regen=False`，它跳过编译直接 dlopen 那个不存在的
   `.so`。必须手动删除残缺缓存目录才会重编。
3. **CUDA 张量喂给 numpy**。`JMID/MID/mid.py::_add_mean` 做
   `data + np.expand_dims(node.get_mean_x_and_y(), axis=0)`，而 `data`
   （预测轨迹）是 CUDA 张量，触发 `Tensor.__array__` 报
   `can't convert cuda:0 device type tensor to numpy`。
   调用方 `_get_preds_sicnav_inference` 随后用 torch 张量做索引，所以不能把
   `data` 转成 numpy；改为在 `data` 所在设备上构造均值张量，算术完全等价。
   （`_unnormalize_pixels` 有同类隐患，但 `normalized_px=False` 时提前返回，
   本次未触发，未改。）

## 关于 20 人场景的预警

上面这些不只是构建麻烦，指向一个可能的结构性限制：

- 5 个行人的精确 Hessian 已经是 25.4 MB C 源码、`-O3` 编不动。本项目正式组
  要跑到 **20 人**，该文件规模随人数增长远快于线性。
- 作者自己在非仿真分支里按人数下调求解迭代：1 人 10 次、2 人 8 次、3 人 4 次、
  4 人及以上 3 次（`campc_acados_opt.py` 第 412–424 行附近）。仿真分支是固定 50 次。
  这是作者承认该方法随人数扩展困难的直接证据。

因此 SICNav-JMID 在六场景的 20 人配置上，可能同时撞上代码生成规模和单步耗时两堵墙。
拿到 5 人的实测单步耗时后再判断是做受限人数对比，还是如实记为该配置不可行。
不提前下结论，也不为了出数字而偷偷改人数。

---

# 追加：headless bridge、matched-unicycle、扩展性、BNBRL+ 训练链

## 环境隔离（已修复我造成的污染）

**我先前把 SICNav 依赖装进了共享的 `miniconda3/envs/crowdnav`，而贝叶斯项目本地跑的正是这个环境。**
`tensorboardX` 连带把 protobuf 升到 5.29.6，弄坏了该环境的 tensorflow 2.10.1（要求 `<3.20`）。

已处理：

- 新建专用环境 **`sicnav`**（`/home/abc/miniconda3/envs/sicnav`），由 `crowdnav` 完整克隆而来，
  SICNav 依赖全部装在这里。新环境下重跑 CVG 结果与旧环境一致（成功，nav_time 3.25，0 碰撞）。
- `crowdnav` 已恢复：卸掉 `tensorboardX / orjson / ncls / dill / distinctipy / easydict /
  acados_template`（逐个确认已清除），protobuf 装回 3.19.6。
  `tensorflow 2.10.1`、`torch`、`numpy`、`scipy`、`gym`、`casadi`、`rvo2` 均可导入。

两点必须记明：

1. **protobuf 的原始版本无法证明。**该环境本身自相矛盾：`tensorflow` 要 `<3.20`、
   `onnx` 要 `>=3.20.2`，永远不可能同时满足，所以在我动手之前就有一个是坏的。
   选 3.19.6 是因为它同时满足 tensorflow / wandb / tensorboard，只有 onnx 不满足。
   现在 `import onnx` 失败。
2. `psutil` 缺失导致 `wandb` 无法导入，**这不是我造成的**（未卸过 psutil），未擅自安装。

C++ 侧一直是隔离的：三套 acados 各自独立目录、GSL 本地 prefix、两个 catkin 工作空间互不相干，
**全程未使用 sudo/apt，未写入任何系统目录**。

## headless bridge：T-MPC++ 与 SH-MPC 升到 `NATIVE_OK`

新增 catkin 包 `mpc_planner_bridge`（两个工作空间各一份，源码同一份），
直接调用 `MPCPlanner::Planner` 的 `reset / onDataReceived / solveMPC / getSolution`，
不掺任何自写控制逻辑。要点：

- **协议通道用独立文件描述符。**roscpp 把 INFO 日志写 stdout，会直接冲垮回包；
  启动时 `dup(STDOUT)` 取得私有 fd，再把 fd 1 指向 stderr，此后任何库的输出都污染不了协议。
- 取控制量按上游写法：`getSolution(1, "v")` 与 `getSolution(0, "w")`；求解失败时复现上游的
  制动输入，而不是把失败包装成"命令停车"。
- 需要 `-DMPC_PLANNER_ROS`（否则 `ros_tools/logging.h` 走 ROS2 分支）、
  `ros_tools/profiling.h`（`Planner` 持有 `unique_ptr<RosTools::Timer>`，仅前向声明无法析构）、
  `ros::init` 与一个 headless `roscore`（模块构造时就会建可视化发布器）。
- `guidance_planner.yaml` 与 `scenario_module/config/params.yaml` 必须显式
  `rosparam load`；后者在 launch 里本来就是注释掉的。

闭环冒烟（5 人 circle，起点 (-5,0) 终点 (5,0)，行人匀速、不确定度随前瞻线性增长）：

| | 步数 | 到达 | 碰撞 | 最小净空 | 求解失败 | 单步求解 mean / p50 / p95 / max |
|---|---|---|---|---|---|---|
| T-MPC++ | 105 | 是 | 无 | 0.1149 m | 0 / 105 | 31.71 / 31.60 / 35.99 / 37.91 ms |
| SH-MPC | 38 | 是 | 无 | 0.0358 m | 0 / 38 | 44.57 / 45.72 / 47.86 / 49.24 ms |

身份运行时核对：T-MPC 日志打出 `Using T-MPC++ (Adding the non-guided planner in parallel)`
与 4 个引导求解器；SH 从参数服务器读回 `enable_safe_horizon=true`、
`safe_sampling.compute_automatically=true`、`confidence=0.01`、`removal_count=0`。

这是 native 冒烟，用的是 bridge 自带的简化场景，**不是本项目六场景，不能进主表**。

## matched-unicycle

### 改法

`continuous_mpc_gate.py` 只加一个可覆盖的钩子：

```python
def _rollout(self, samples, obs):
    controls = self._project_controls(samples, obs.robot_velocity)
    positions = obs.robot_xy[None, None, :] + np.cumsum(controls * self.cfg.dt, axis=1)
    return controls, controls, positions      # 全向：参数即速度
```

`plan()` 改为 `params, controls, positions = self._rollout(samples, obs)`，
CEM 拟合与最优解存储用 `params`，代价/净空/危险度全部仍用 `controls`（XY 速度）与 `positions`。
**因此风险函数、可行性分级、精英选择、热启动策略一行未改。**

新文件 `unicycle_mpc_gate.py`：`UnicycleCEMMPC` 只覆盖 `_project_controls`、`_rollout`、
`_initial_mean`、`_seed_trajectories`、`_route_seeds`。采样参数变成 (速度, 角增量)，
展开严格复现 CrowdSim 的 `Agent.step`：

```
theta_{k+1} = theta_k + r_k        # r 是角增量，不再乘 dt
p_{k+1}     = p_k + v_k * [cos theta_{k+1}, sin theta_{k+1}] * dt
```

`run_episode` 增加 `robot_kinematics`，按运动学分派 `ActionXY` / `ActionRot`；
**只把机器人切成 unicycle，行人保持 holonomic 的 ORCA**，并在运行时断言这一点。
初始朝向登记为"指向目标"，使任何一方都不会平白多付一次起步转向。

`crowd_sim.py` 里一处无条件的 `norm([action.vx, action.vy])`（稠密奖励的站立惩罚）
改为按运动学分派，对全向是零变化。

### 登记的动力学界限

`v_max=1.0`、`a_max=2.0`、`dt=0.25`、`horizon=16` 沿用本项目；新增
`omega_max=1.0 rad/s`（每步角增量上限 0.25 rad）、`reverse=False`。
这些必须对所有参与匹配比较的方法相同，尚未与三个现代 MPC 的原生支持范围核定。

### 回归：全向路径逐位不变

18 个回合（circle/square × 5/10/20 人 × 三档工作点 × 三个 case），比对
`event / success / collision / timeout / nav_time / path_length / min_clearance /
actual_min_clearance / actual_overlap_steps / collision_union /
success_without_overlap / solver_steps / observation_hash`：

- 重构 `_rollout` 之后：**逐位一致**
- 接好 unicycle 动作路径并改过 `crowd_sim.py` 之后：**仍逐位一致**

### 接口验收（§10.1，全部通过）

| 项 | 结果 |
|---|---|
| 展开与 CrowdSim `Agent.step` 逐步一致 | 位置/速度最大误差 < 1e-12 |
| 速度落在 `[0, v_max]` | `[0.000000, 1.000000]` |
| 角增量落在 `±omega_max*dt` | max \|r\| = 0.250000，界 0.250000 |
| 加速度落在 `a_max*dt` | max \|dv\| = 0.500000，界 0.500000 |
| 直行等价于全向轨迹 | 最大差 2.45e-16 |
| 0 / 1 / 5 / 20 人均可规划 | 9.4 / 12.9 / 24.2 / 69.1 ms |
| 缺朝向时报错而非猜测 | 抛 `ValueError` |
| 环境确实在跑 unicycle 机器人 | `kinematics=unicycle` |
| 真闭环 | `reach_goal` t=10.25 41 步 净空 0.1740 |

**20 人单步 69.1 ms，仍在 250 ms 周期内。**

## SICNav 扩展性（5 → 10 → 20，进行中）

| 人数 | 退出码 | 峰值内存 | 耗时 | 结局 | SQP 达上限 | QP 失败 |
|---|---|---|---|---|---|---|
| 5 | 0 | 1349 MB | 32 s（求解器已缓存） | 成功 nav_time 3.0，0 碰撞，0 too_close | 12 | 0 |
| 10 | 进行中 | | | | | |
| 20 | 未开始 | | | | | |

已知的 5 人成本：精确 Hessian 生成 **25.4 MB 的 C 源码**，`-O1` 编译峰值 **8.7 GB**、
耗时 **9 分 42 秒**（`-O3` 在 15 GB 内存上必被 OOM 杀死）。

先前一次 10 人尝试跑满 50 分钟仍未调用过一次 `cc`，全部时间花在 CasADi 符号构造上；
该进程后来在我卸载 `crowdnav` 依赖时死亡，结果作废，已在干净的 `sicnav` 环境重跑。

## BNBRL+：状态更正与训练链验证

**更正先前的说法。**缺的是作者的导航策略权重，不是"只有用户能解"。索取与自训并行推进。

已核实的资产状况：

- 导航策略 checkpoint：本机与上游均不存在；上游该提交 209 个文件里无 `trained_models/`，
  Releases 为空；`test.py` 默认请求的 `trained_models/BNDNN/checkpoints/20200.pt` 不存在。
- GST 预测器：**仓库里两个 `gst_updated/results/*/sj` 目录都只有 TensorBoard 事件文件，
  没有任何 `.pt`。**唯一真实权重是仓库根目录
  `100-gumbel_.../sj/checkpoint/epoch_100.pt`（828521 字节），而配置指向的是
  `gst_updated/results/..._rand/sj`。
- 该权重是否为 `_rand`（随机化行人属性）变体，无法从其 `args.pickle` 证明——
  里面只记了 dataset/层数/头数/种子。而 BNBRL+ 自己的配置是 `env.randomize_attributes = True`。

已做的修改（登记）：`crowd_nav/configs/config.py` 的 `pred.model_dir` 指向本机实际存在的
那份权重，并在注释中写明这是偏离作者默认、且 `_rand` 归属无法证明。
原文件已备份至 `temp/modern/bnbrl_config_original.py`。

模块身份核对（§2.4，不靠 cwd 猜）：

```
crowd_nav -> src/BNBRLplus/./crowd_nav/__init__.py
crowd_sim -> src/BNBRLplus/./crowd_sim/__init__.py
rl        -> src/BNBRLplus/./rl/__init__.py
```

**训练链本身可以跑通。**`train.py` 加载 GST 模型成功（`LOADED MODEL`，`device: cuda:0`），
PPO 正常更新，2 进程下 42 FPS，奖励从 -13.0 起步。
未在 `crowdnav` 环境里安装任何东西（依赖本来就齐），因此没有二次污染。

作者默认预算：`num_env_steps = 10e6`、`num_processes = 16`、`num_steps = 30`、`ppo_epoch = 5`。

---

# 追加：统一评测接入、扩展性定论、公平性问题

## SICNav-JMID 的规模限制：定论

| 人数 | 退出码 | 峰值内存 | 耗时 | 结果 |
|---|---|---|---|---|
| 5 | 0 | 1349 MB | 32 s（求解器已缓存） | 成功，nav_time 3.0，0 碰撞，SQP 达上限 12 次，QP 失败 0 |
| 10 | 124（超时） | **481 MB** | **4:00:00** | **一次 `cc` 都未调用** |
| 20 | 124（超时） | **481 MB** | **4:00:00** | **一次 `cc` 都未调用** |

两次超时都停在 CasADi 符号构造阶段，**单线程满载、内存不到 0.5 GB**。
即加机器、加核、加内存都不改变量级；卡的是符号表达式构造本身。

5 人时同一步骤约 1 分钟即可完成，其后编译 25.4 MB 的 Hessian C 文件在 `-O1` 下
耗时 9 分 42 秒、峰值 8.7 GB。

**结论：SICNav-JMID 在本项目的 10 人与 20 人配置上不可行**，不是资源不足，
也不是接口问题（5 人同一条链路完整跑通），而是精确 Hessian 的符号构造随人数的增长。
后续如需该对照，只能限定在 5 人量级，并明确标注这一适用范围。

## 编译优化等级：`-O1` 不改变数值结局

同一 5 人回合，只改 `ext_fun_compile_flags`，其余（配置、权重、种子、场景、case）全同：

| | 结局 | nav_time | 碰撞 | too_close | SQP 达上限 | QP 失败 |
|---|---|---|---|---|---|---|
| `-O3 -ffast-math`（作者默认） | 成功 | 3.25 | 0 | 0 | 13 | 0 |
| `-O1` | 成功 | 3.25 | 0 | 0 | 13 | 0 |

结局、导航时间、求解器行为逐项一致。**`-O1` 可用于推进**；
但这是单回合证据，不是逐位等价的证明，正式耗时报告仍须注明实际编译工具链。

## headless bridge 接入本项目六场景

`modern_worker.py` 新增，把 T-MPC++ / SH-MPC 包装成 `run_episode` 能直接用的
`planner_type`，使用本项目自己的观测、布局与审计计分，机器人跑 matched unicycle。
两者都是 **shared-belief 臂**：接收本项目的后验，而非它们自己的推断。

三个必须记明的实现点：

1. **SH-MPC 结构上不接受点预测。**`scenario_constraints.cpp:119` 断言
   `prediction.type != DETERMINISTIC`，给零不确定度直接 abort。
   因此它**不可能做端到端臂**，除非另有不确定度来源。
   现做法是把本项目后验的 2×2 协方差按特征分解成椭圆长短半轴与倾角传入。
2. **参考路径必须每回合固定一次。**Contouring 模块用 spline 状态记录沿路径的进度；
   我最初每步都按机器人当前位置重建路径，等于每步把进度清零，控制器永远走不出去。
   改为 reset 后第一次构建、之后复用。
3. **两个工作空间的包都要在 `ROS_PACKAGE_PATH` 里。**否则 SH 的 `rospack`
   找不到 `scenario_module`，随后尝试创建 `//samples` 被拒并 abort。

## 统一执行器界限（§3.6，已登记）

发现两边原生界限并不相同：

| | 速度 | 加速度 | 角速度 |
|---|---|---|---|
| mpc_planner 模型原生 | `[-0.01, 3.0]` m/s | `±2.0` m/s² | `±0.8` rad/s |
| 本项目 unicycle 初版 | `[0, 1.0]` m/s | `±2.0` m/s² | `±1.0` rad/s |

若不统一，比较里会混入"一方能跑 3 m/s、另一方限速 1 m/s"。
已统一为各方原生都支持的交集并写入两个求解器的模型界限与 `UnicycleConfig`：

**`v ∈ [0, 1.0]`、`a ∈ ±2.0`、`ω ∈ ±0.8`、不倒车。**

`ω` 取 0.8 而非 1.0，是取现代 MPC 原生支持的较紧值，不让 Bayes 使用对方做不到的转向。
SH 的模型多一个 slack 状态，界限向量为 8 维而非 7 维，已分别处理。

## 全向 vs matched-unicycle（统一界限后，240 回合）

6 场景 × 20 case × 2 臂，同布局、同人群、同跟踪器、同风险函数，只差执行器：

| 臂 | n | SR% | CR% | TR% | 失败惩罚时间 s | 路径 m | p95 ms |
|---|---|---|---|---|---|---|---|
| holonomic | 120 | **94.17** | **0.83** | 5.00 | 12.465 | 10.634 | 155.3 |
| unicycle | 120 | 87.50 | **10.83** | 1.67 | 13.785 | 9.649 | 149.4 |

配对 100 个布局：全向成功而 unicycle 失败 9 个，反向 1 个，
**配对精确 McNemar p = 0.02148**。

即：在统一转向界限下，把机器人变成非全向使成功率下降 6.7 个点、碰撞率上升约 13 倍，
且配对检验显著。**这就是必须建 matched-unicycle 组的直接证据**——
拿全向的成绩去对三个 unicycle 方法，赢的一部分只是转向自由度。

（先前一版在 `ω=1.0` 下测得 SR 90.83 / CR 7.50 / p=0.1094，因界限不统一已作废，
存档为 `kinematics_compare_omega1.0.json`。）

## 尚不成立的比较：风险水平未对齐

首轮接入后的六场景结果为 T-MPC++ SR 3.33%、SH-MPC SR 26.67%，
**这些数字不作数，也不会被引用**。原因是风险参数根本不在同一量级：

| | 允许的碰撞风险 |
|---|---|
| T-MPC++ / SH-MPC | `probabilistic.risk = 0.05` |
| 本项目 Bayes（工作点 3） | `chance_limit = 0.75` |

相差 15 倍。实测证据：把本项目后验按接口约定以 1σ 传入后，
**4 秒前瞻处的 σ 达到 2.19 m**（行人半径仅 0.3 m），
经他们各自的风险放大后障碍等效半径接近 2.5 m，
在半径 4 m 的圆形场景里等于封死通路——表现为净空高达 +1.6 m、
25 秒只走 1.2 m 的"冻结"，而非碰撞。

这正是 §10.2 所警告的：两边的风险事件定义不同，不能因为都叫"风险"就直接比。
**若照搬这组数字，会得出"我们大幅优于两个现代 MPC"的虚假结论。**

正在进行的补救（§10.2 阶段 V）：对每个现代方法扫其自身的 `probabilistic.risk`
（0.05 / 0.15 / 0.30 / 0.50 / 0.75），在开发集上各自求安全—效率前沿，
再用 `freeze_points.py` 里已登记的同一条规则冻结各自工作点，然后才比较。
仅用开发 case，不触碰正式测试集。

## BNBRL+ 训练预算

16 进程实测 194 FPS（与另两个任务竞争，属下界）。
作者默认 `num_env_steps = 10e6` → 约 **14.3 小时**。
2 进程时 42 FPS。训练链本身已验证可跑通（GST 模型加载成功，PPO 正常更新）。

---

# 追加：对 Codex 复核的响应与自我纠正（2026-09-09）

Codex 复核后指出四个问题。我逐条在源码与数据上核实，**四条全部成立**，
并额外发现两处它没有点明的。以下是纠正与处理。BNBRL+ 按用户指示暂缓。

## 纠正一：配对统计报错了，撤回"成功率显著下降"

我先前的配对键是 `(scenario, humans, case_id)`。但登记的六场景里有两个
`square_crossing` 20 人，只差 `square_width`（10.0 与 12.0），
**互相覆盖，120 个回合塌成 100 个键，20 个回合被静默丢弃**。

重算（按循环顺序恢复 120 对）：

| | 我先前报的（错） | 正确值 |
|---|---|---|
| 配对数 | 100 | **120** |
| 成功：全向胜 / 负 | 9 / 1 | **13 / 5** |
| 成功的 McNemar p | 0.0215（称显著） | **0.0963，不显著** |
| 碰撞：unicycle 多碰 / 少碰 | 未单独检验 | **13 / 1** |
| 碰撞的 McNemar p | — | **0.00183，显著** |

**撤回**"非全向使成功率显著下降"。正确结论：非全向对**碰撞率**的影响显著，
对**成功率**的影响在这个样本量下不显著。总体百分比（94.17/0.83 对 87.50/10.83）不变。

已修 `compare_kinematics.py`：配对键含场景尺寸，并加唯一性断言（塌陷即报错而非静默丢弃），
碰撞单独做 McNemar。

## 纠正二：SH-MPC 的风险被翻倍，我的两个高档位无效

`scenario_module/src/config.cpp:27`

```cpp
risk_ = CONFIG["probabilistic"]["risk"].as<double>() * 2.;
```

所以我扫的 0.50、0.75 实际是 1.0、1.5，越界；日志出现
`Safety Certifier: Sample size maximum in bisection was reached!`，随后去算
五十万规模的风险查表，卡死。已终止并作废，存档为
`risk_sweep/sweep_INVALID_risk_doubling.log`。

**Codex 没点明但同样严重的一点**：T-MPC++ 的 `EllipsoidConstraintModule`
**不翻倍**。所以同一个 yaml 数值在两个方法上含义不同，
我先前那轮扫描里两者从来就不在同一有效风险上（config 0.05 → T-MPC 0.05、SH 0.10）。

已重建档位，按**有效风险**对齐：

| 有效风险 | T-MPC++ config | SH-MPC config |
|---|---|---|
| 0.05 | 0.0500 | 0.0250 |
| 0.10 | 0.1000 | 0.0500 |
| 0.20 | 0.2000 | 0.1000 |
| 0.35 | 0.3500 | 0.1750 |
| 0.50 | 0.5000 | 0.2500 |

上限定在有效 0.50：再高 SH 的二分搜索就撞样本数上限，是退化而不是"更宽松"。

## 纠正三：SH-MPC 收到的不是"同一份后验"

`scenario_module/src/sampler.cpp:452`

```cpp
bivariate_gaussian += A_[v][k][mode] * xi_k;              // 逐步累加
samples_[k][v][0](s) = cur_pose(0) + bivariate_gaussian(0) * dt;
```

其中 `A_ = chol(Sigma_)`，`Sigma_` 由我送的半轴与倾角构造
（源码注释：`the covariance ... is assumed constant over the horizon`）。

即 **SH-MPC 把该字段当成每步的过程噪声增量，积分成随机游走，再乘 dt**；
而 T-MPC++ 的椭圆约束把同一字段当作第 k 步的**边缘半轴**。
**两者对同一协议字段的语义相反**，送同一个数组必然有一方拿到错误分布。

已修：`modern_worker.py` 按消费者分别给出正确形式。增量由边缘反推：

    Cov(d_k) = dt² · Σ_{j≤k} S_j  ⇒  S_k = (P_k − P_{k−1}) / dt²

`verify_uncertainty.py` 用解析已知的边缘序列做蒙特卡洛验证：
20 万样本下随机游走复原边缘协方差，**相对误差 0.38%**。
接口验收里再验一次，12 万样本下 **0.52%**。
若某一步边缘协方差下降（无法表示成随机游走），代码**报错而不是截断**，
因为截断等于悄悄换掉规划器拿到的分布。

两种语义的差距有多大：同一份后验下，k=0 处增量半轴 0.8973 对边缘 0.2243，差约 4 倍。

## 纠正四：撤回"确定性、忽略种子"的声明；两个方法都是随机的

`sampler.cpp:377` 用 `std::mt19937{std::random_device{}()}`，SH-MPC 本就不可复现。

**Codex 没点明的第二处**：T-MPC++ 同样不可复现，而且不是 PRM 种子的问题
（`guidance_planner.yaml` 里 `seed: 1` 是固定的）。它的四个引导求解器跑在
`#pragma omp parallel for num_threads(8)` 下，胜出的分支随运行而变。实测同一布局三次重复：

| 设置 | 路径长度 |
|---|---|
| 默认（并行） | 4.461 / 3.944 / 5.494 m |
| `OMP_THREAD_LIMIT=1`（串行） | 4.460405 / 4.460568 / 4.460595 m |

即并行是主导的随机来源，串行化后离散度从 1.55 m 降到 2e-4 m，
但串行化改变了该方法的运行时行为，耗时数据也就不再是它的设计配置。

**结论：两个现代方法的单次结果都不可复现，任何比较都必须重复多次并报离散度。**
接口验收改为如实记录离散度，不再断言确定性。

## 纠正五：撤回"SICNav 规模不可行"的判词

先前写的是"10 人与 20 人配置上不可行"。**这话说过头了。**
实测只支持："**当前构建、当前配置下，10 人与 20 人的代码生成在 4 小时预算内未完成**"，
两次都停在 CasADi 符号构造阶段（单线程满载、内存不到 0.5 GB，`cc` 一次未调用）。

不能据此断言所有大规模配置不可行：未定位符号构造的耗时热点，未尝试降低
`orca_kkt_horiz`、改用 Gauss-Newton 近似 Hessian 等作者提供的选项。
下一步应先定位热点，而不是再盲等 4 小时。

## 新增：bridge 接口验收（不看成功率，看量本身）

`bridge/interface_audit.py`，全部通过：

| 检查 | 结果 |
|---|---|
| 机器人状态逐字段转录 | PASS |
| 预测第 k 步 = 未来 k+1 个控制周期 | 最大误差 8.88e-16 |
| 全部检测都送达（不被 12 障碍上限截断） | 3/3 |
| T-MPC 收到边缘半轴 | 最大误差 4.99e-10 |
| SH 收到的与边缘形式不同 | 通过（相同才是 bug） |
| SH 的随机游走复原后验边缘 | 相对误差 0.52% |
| 角速度→角增量恰好乘一次 dt | PASS |
| 重复运行完成并记录离散度 | T-MPC 0.218 m，SH 2.589 m（三次） |

## 新增：受限观测 facade

`visible_history_env.py`，用于关闭 `sicnav_acados.py:1177` 直接读 `self.env.states`
（仿真器全量真值，含被遮挡行人）这条路径。facade 只转发时钟、配置与场景几何，
`states` 返回适配器按检测维护的历史；未审查的属性一律抛
`AttributeError` 而不是转发给真环境；只读。已用假环境验证真值被屏蔽、越权属性被拦截。

**尚未接入 SICNav 的正式评测**——目前 SICNav 仍是完全可观测的原生冒烟，
遮挡评测前必须先接上它。

## 扫描调度的修正

- 档位文件按有效风险重建（见纠正二）。
- **失败回合计入分母。**先前的汇总把 `event == "error"` 的回合从分母剔除，
  等于用"能跑通的子集"评分，会让在难布局上崩掉的配置显得最安全，
  再被冻结规则选中。现在失败按非成功计入，并带时限惩罚时间。
- **任何有失败回合的档位不得被冻结。**
- 每个失败回合打印方法、档位、场景、case 与异常类型，不再只有一个计数。

---

# 追加：求解失败的部分归因（2026-09-09，结论未闭合）

## 先撤回两个我下过的错误推断

1. **"QP error status 3 = 问题不可行"不成立。**导出底层量后：失败步的
   `qp_status = 0`（QP 正常求解），`sqp_iter` 恒为 1。真正的判据在
   `mpc_planner_solver/src/acados_solver_interface.cpp`：

   ```cpp
   ocp_nlp_get(_nlp_solver, "res_eq", &res_eq);
   if (res_eq > 1e-2 && _exit_code_one_iter == ACADOS_SUCCESS)
       _exit_code_one_iter = ACADOS_QP_FAILURE;   // 名字误导：这是动力学残差超阈值
   ```

   即"失败"= 动力学等式残差超过 1e-2，不是无解。常量名 `ACADOS_QP_FAILURE`
   与日志里的 `QP solver returned error status 3` 一起把我带偏了。

2. **"终端权重把机器人按在直线上导致 4 秒内无解"不成立。**那是目标函数惩罚，
   不是硬约束，不能由它推出不可行。

## 又发现一层：主求解器从未被求解

我给 bridge 加了残差导出后，四个残差全是 0.0。原因是 T-MPC++ 的
`GuidanceConstraints` 用自己的一组并行分支求解器，`FindBestPlanner()` 选出胜者后
把它的 `_info` 拷回主求解器（`guidance_constraints.cpp` 约 383 行），
但主求解器的 NLP 对象从未求解，所以 `ocp_nlp_get(主求解器, "res_eq")` 恒为 0，
`_info.nlp_res` 是未初始化内存（`6.93e-310`）。

整体失败等于 `FindBestPlanner()` 返回 −1，即所有并行分支都不可行；
结合分支的 `qp_status = 0`，失败只能来自分支求解器的 `res_eq` 超阈值。

## 决定性对照：原因是不确定度量级，不是我的配置

建了一个**完全未改动的作者原生工作副本** `build_modern/tmpc_native`
（N=30、dt=0.2、20 Hz、原生半径、连我加的统一执行器界限都撤掉），
与我们的配置在同一驱动、同一场景下对比，只改不确定度：

| 配置 | 末端 sigma | 求解失败 | 到达目标 |
|---|---|---|---|
| 作者原生 N=30 dt=0.2 | 0.4 m | **0 / 107** | 是 |
| 作者原生 N=30 dt=0.2 | **2.2 m** | **41 / 152 = 27%** | 是 |
| 我们的 N=16 dt=0.25 | 0.4 m | **0 / 61** | 是 |
| 我们的 N=16 dt=0.25 | **2.2 m** | **30 / 250 = 12%** | 否 |

**收窄后的结论（先前写过头，此处更正）：**

*成立*：在这个驱动场景里，**增大 sigma 会显著增加求解失败**——两种配置在
sigma≈0.4 m 下都是零失败，在 sigma≈2.2 m 下都出现大量失败。

*不成立，我先前误写为成立*：
"与配置无关"。同一张表里就有反证——**高 sigma 下原生配置失败 27% 却到达了目标，
我们的配置失败 12% 却没到达**。失败率更低而结果更差，说明配置仍在影响结果，
只是影响的不是失败率本身。

*仍未排除*：接入问题。零失败不等于接入正确；这个驱动用的是我自己写的简化场景，
与六场景的观测、参考路径、行人行为都不同。

*仅对 T-MPC++ 取证*：SH-MPC 尚未做原生配置对照，**不能用 T-MPC++ 的结论代替它**。

对应的事实：本项目遮挡感知后验在 4 秒前瞻处 sigma 达 **2.19 m**（行人半径 0.3 m）。

## 这意味着什么，以及不意味着什么

**目前只能说**：sigma 是一个已确认的影响因素——在这个驱动场景下，
把末端 sigma 从 0.4 m 提到 2.2 m 会让 T-MPC++ 出现 12%~27% 的求解失败。

**不能说**：这是唯一原因，或已排除接入问题，或与配置无关。三者都还没有证据。

**更不能说**："我们的方法更好"。这不是同一任务上的性能比较。

**正式比较不应强制统一预测步数**，但"允许各自选时域"不等于允许拿到更多观测或
更多控制机会。必须把三个频率分开登记并统一其中两个：

| 量 | 处理 |
|---|---|
| 预测步长与步数（N、integrator_step） | **允许各方法自选**，登记并报告 |
| 内部求解频率 | **允许各方法自选**，计入耗时预算 |
| 实际感知与动作执行频率 | **必须统一**（本项目 dt = 0.25 s），任何方法都不得每 0.05 s 拿一次新观测或下一次新指令 |

上游的 `control_frequency: 20` 同时兼作这三者，接入时必须拆开：
预测/求解可以按它自己的节奏，但送进来的观测和取走的指令仍是每 0.25 s 一次。
**也不得为了改善成绩人为压小后验 sigma。**

先前把 N 从 30 砍到 16 的做法作废；但 20 Hz → 4 Hz 这一项要保留在"执行频率"上，
撤销的是它对"预测/求解频率"的连带影响。

## 为诊断新增的东西

| 位置 | 改动 |
|---|---|
| `mpc_planner/include/mpc_planner/planner.h` | 加只读 `solverForDiagnostics()`（两个副本） |
| `mpc_planner_solver/include/.../acados_solver_interface.h` | 加只读 `nlpSolverForDiagnostics()` |
| `bridge/mpc_bridge.cpp` | 每步回报 `sqp_iter/qp_status/nlp_res/kkt/cost/res_*`；`MPC_BRIDGE_FAILURE_DUMP` 指定时把失败步原样存盘以便重放 |
| `build_modern/tmpc_native/` | 作者原生配置的独立工作副本，作对照基准 |
| `bridge/interface_audit.py` | 10 项接口验收，全部通过 |
| `bridge/verify_uncertainty.py` | 边缘/增量换算的蒙特卡洛验证 |

---

# 追加：T-MPC++ 求解失败的证据链（2026-09-09）

## 先说一个会让此前诊断失效的陷阱

`guidance_constraints.cpp` 的 `optimize()` 里，当 `FindBestPlanner()` 返回 −1
（没有可用分支）时，**直接 return，不拷贝任何 solver info**。
因此在失败步上读到的 `_info.acados_status`、`res_eq` 等全部是**上一个成功步残留的陈旧值**。

我此前几轮基于 `_info` 的失败步诊断因此无效，已作废。
现在在 `optimize()` 里**每一步都写**分支统计字段，包括没有可用分支的那一步。

## 逐层取证的结果（T-MPC++，5 人 circle，工作点 3）

| 层 | 观测 |
|---|---|
| 回合层 | case 3650 成功 10/101，case 3651 成功 34/101 |
| 分支层 | 分支总数 5，**实际启用中位 2**（引导规划器找到的同伦类少），**成功分支 0** |
| 分支退出码 | 失败步上 100% 是 `exit_code = 4` |
| 分支 acados status | **`QP_FAILURE` × 91 / × 67**，来自 `Solver_acados_solve` 本身 |
| `res_eq > 1e-2` 规则 | **0%**——失败与这条规则无关，`res_eq` 恒为 0 |
| QP 层打印 | `SQP_RTI: QP solver returned error status 3` |

## 关键：status 3 排除了不可行，但不能断定是 NaN

`acados/ocp_qp/ocp_qp_hpipm.c:334` 只重映射 0/1/2：

```c
int acados_status = mem->status;
if (mem->status == 0) acados_status = ACADOS_SUCCESS;
if (mem->status == 1) acados_status = ACADOS_MAXITER;
if (mem->status == 2) acados_status = ACADOS_MINSTEP;
return acados_status;
```

HPIPM 自身的状态码（`external/hpipm/include/hpipm_common.h`）：

| 值 | 含义 |
|---|---|
| 0 | SUCCESS |
| 1 | MAX_ITER |
| 2 | MIN_STEP |
| **3** | **NAN_SOL —— 解里检测到 NaN** |
| 4 | INCONS_EQ |

**这里我又推断过头了，现更正。**该映射把 HPIPM 的 `MIN_STEP=2` 也映射成 3，
而 `NAN_SOL=3` 未被映射、原样透传也是 3：

| HPIPM 原始 | 返回值 |
|---|---|
| 0 SUCCESS | 0 |
| 1 MAX_ITER | 2 |
| **2 MIN_STEP** | **3** |
| **3 NAN_SOL** | **3** |
| 4 INCONS_EQ | 4 |

**返回值 3 同时来自 MIN_STEP 和 NAN_SOL，二者不可区分。**
所以「是 NaN」这个断言没有依据，撤回。只能说：**排除了不可行**
（INCONS_EQ 会返回 4，而我们看到的是 3），是「最小步长终止」或「解含 NaN」两者之一。
要区分必须读 HPIPM 的原始状态，不能从映射后的数字反推。

## 目前能说与不能说

**能说（有证据）**：T-MPC++ 在这些步上失败，是因为所有启用分支的 QP 求解返回
非成功状态，`FindBestPlanner()` 无可选项，规划器退化为制动。
`res_eq > 1e-2` 那条规则没有参与。QP 层返回值 3 **排除了 INCONS_EQ（不可行）**，
但在 MIN_STEP 与 NAN_SOL 之间不可区分。

**能说（有证据，但只是相关性）**：NaN 的发生率随不确定度量级上升——
在同一驱动场景下，末端 sigma 0.4 m 时零失败，2.2 m 时作者原生配置失败 27%。

**不能说**：NaN 的直接成因已确定。相关不等于因果；还没有定位是哪个约束项、
哪个数据条目产生了非有限值。下一步应在失败步上导出送入 QP 的约束数据并检查有限性与条件数。

**不能说**：SH-MPC 同因。它走 `ScenarioConstraints`，不经过分支求解器，
上面的分支字段对它是 N/A（已在代码里标为 0 而不是留下误导性的哨兵值）。
**必须单独取证。**

## 为这条证据链新增的导出

`AcadosInfo` 增加（三个工作副本同步）：`acados_status`、`res_eq_at_exit`、
`res_stat/ineq/comp_at_exit`、`branches_total/enabled/succeeded`、
`branch_worst_exit`、`branch_acados_status`、`branch_res_eq`。
`GuidanceConstraints::optimize()` 每步写入分支统计；
`ScenarioConstraints::optimize()` 把分支字段置 0 表示不适用。
bridge 每步随回复一并上报。

## 另一个独立缺陷：一个 bridge 进程只能跑 5 个回合

`experiment_util.cpp:109` 在 `Planner::reset()` 里数实验次数，
达到 `recording.num_experiments`（默认 5）就 `ROSTOOLS_ASSERT` 直接 abort；
同时数据记录写死到不存在的 `/workspace/src/mpc_planner/data`，
创建失败的异常顺协议通道冒出来打乱协议。

重放 25 步时触发了 **14 次崩溃**。已在三个工作副本与 14 个档位文件里关闭记录、
把上限提到 10^6，并核验 YAML 仍可解析；复测 75 次交互零崩溃。

**这个缺陷此前被"每回合新建 bridge 进程"的写法掩盖**。若为省开销改成进程复用，
第 6 个回合会静默死亡。

## 失败步重放（25 步 × 3 次）

- **18 步确定性复现**（同一请求同样失败），4 步稳定成功，3 步时好时坏。
  → 失败主要由输入决定，不是并行随机性。
- 预测—执行一致性：在成功步上执行一个周期后与最近行人预测的最小净空 **+1.31 m**，
  未发现预测与执行错位。


---

# 追加：撤回「直线参考是根因」，转向参考路径链路排查（2026-09-09）

## 撤回

先前写的「根因是我喂了一条直穿人群的直线，而上游用绕开障碍的 roadmap 参考」——**撤回**。

场景里没有静态障碍，起点到终点的直线本来就是**合法参考**；
躲开动态行人本来就是规划器该做的事，不是参考路线该替它做的。
而且当时只有两个回合、随机性未控制，两个方法本身又都是随机的
（同一 case 上 T-MPC 前后两轮是 34/101、0.51 m 与 66/101、8.93 m）。

**只能说：Contouring 相关的配置与接入值得排查。不能由此推出这两个方法只适合道路场景。**

## 已被数据否定的假设（累计）

| 假设 | 否定依据 |
|---|---|
| QP 不可行 | HPIPM 原始状态是 MIN_STEP，不是 INCONS_EQ |
| 解里出现 NaN | 原始状态是 MIN_STEP，不是 NAN_SOL；先前由映射后的 3 反推是错的 |
| `res_eq > 1e-2` 规则触发 | 失败步 0% 触发 |
| 脏数据进入求解器 | 参数与热启动非有限个数均为 0 |
| 哑元障碍造成量级跨度 | max_obstacles 降到 5 无哑元，失败数几乎不变 |
| 引导层是元凶 | 无引导的 `tmpc_contour` 表现与完整 T-MPC++ 一致 |
| 分支赋值重置 QP 内存 | 在独立版打开同一重置，结果逐项相同 |
| 直线参考是根因 | 无静态障碍时直线是合法参考；样本太少且随机性未控 |

## 当前唯一稳定的观测

`Contouring`（跟踪路径）与 `GoalModule`（跟踪目标点）之间存在稳定差异：
两个 case 上 Contouring 版的 `MIN_STEP` 失败数是 Goal 版的约两倍（40/41 对 17/21）。
这是**待排查的线索**，不是结论。

## 方法身份的约束（已登记）

- **T-MPC++ 正式对照必须保留 Guidance + Contouring + Ellipsoid。**
- **SH-MPC 正式对照必须保留其场景约束机制。**
- `Goal + Ellipsoid`（`configuration_lmpcc`）只作诊断工具，或作为**单列的 LMPCC 对照**，
  **不得替代**这两个现代方法后仍沿用它们的名字。

---

# 追加：又一处我自己制造的混淆，以及 Contouring 的取舍机制（2026-09-09）

## 更正：先前的三方对比被执行器界限污染

`configuration_original_branch` 与 `configuration_lmpcc` 构造模型的路径没有经过我加的
统一界限，因此它们跑的是**作者原生的执行器箱**（`v ∈ [-0.01, 3.0]`、`w ∈ ±0.8`），
而正式的 T-MPC++ / SH-MPC 跑的是 `v ∈ [0, 1.0]`。症状是 `tmpc_contour` 成功步的
速度中位数为 **−0.0100**，正好是原生下界。

**因此先前「`tmpc_contour` 表现与完整 T-MPC++ 一致，故引导层被排除」这一结论作废**，
它比较的是执行能力而不是被测模块。已给两个诊断副本补上同一界限并重建。

## 统一界限、固定随机性后的对比

`OMP_THREAD_LIMIT=1` 使 T-MPC 的并行分支选择确定化；每档重复 3 次，结果逐次一致。

| 变体 | 模块 | case 3650 | case 3651 | 成功步的速度中位 |
|---|---|---|---|---|
| tmpc | Guidance + Contouring + Ellipsoid | 85/101，4.33 m，超时 | 34/101，0.50 m，超时 | 0.009 / 0.000 |
| tmpc_contour | Contouring + Ellipsoid（无引导） | 62/101，3.93 m，超时 | 34/101，0.50 m，超时 | 0.308 / 0.000 |
| tmpc_lmpcc | Goal + Ellipsoid | 84/101，5.27 m，超时 | **39/62，7.88 m，到达** | 0.000 / **1.000** |

## 观察到的机制（待权重扫描确认）

**即使求解成功，带 Contouring 的配置下发的速度也接近 0；Goal 版下发 1.0 并到达。**
所以主要症结不在求解失败，而在目标函数的取舍：

Contouring 代价惩罚偏离参考线（`contour` 与 `terminal_contouring: 10.0`），
而在穿越场景里绕开行人必然要偏离。在上游那套为道路跟随调的权重下，
**「停在线上等」比「绕过去」便宜**。LMPCC 没有这个惩罚项，所以敢横向绕行。

这不是「该方法只适合道路」，而是**权重取舍在无地图穿越场景下的表现**，
且我们从未给过它任何调参预算，而本项目自己的方法是调过的。

## 参考路径窗口的量纲修正（已做，但不是主因）

`Contouring` 只跟踪 `num_segments`（=5）段样条；越过窗口末端时
`Spline2D::getParameters` 用最后一段做常数外推，即路径在窗口末端「停住」。
我原先的 20 点路径段长 0.53 m，5 段仅 2.6 m，而时域可走 4 m。

已让 bridge 在 `INFO` 里报出 `num_segments`，并按
`horizon × dt × v_max × 1.5 / num_segments` 反算点距：现在是 8 点、段长 1.14 m、
窗口 5.7 m > 4 m。**这一项确实生效（轨迹改变），但没有改变结论**，因此不是主因。

## 正在进行：同等调参预算的权重扫描

保持方法身份不变（Guidance + Contouring + Ellipsoid），只动决定「偏离参考线有多贵」
的权重，在开发集上扫五档：

| 档 | 改动 |
|---|---|
| W0_upstream | 上游原值 |
| W1_less_contour | `contour` 0.05→0.01，`terminal_contouring` 10→1 |
| W2_free_lateral | 再加 `terminal_angle` 100→10，`contour`→0.005，`terminal_contouring`→0.5 |
| W3_push_speed | `velocity` 0.55→2.0 |
| W4_combined | W2 + W3 |

2 个场景 × 8 个开发 case × 2 次重复 = 160 回合。不动风险参数、不动约束、不动动力学。

---

# 追加：求解失败的一条影响链（2026-09-09，导航失败尚未解释完）

## 空场景对照：链路本身没问题

无行人、只有参考路径时，T-MPC++（上游权重、Guidance + Contouring + Ellipsoid）：

**49 步到达目标，求解失败 0/49，速度中位 0.675、最大 0.995 ≈ v_max。**

所以下列环节在无障碍时全部正常：参考路径构造与样条窗口、Contouring 代价与上游权重、
引导规划器时域、执行器界限、求解器本身。

## sigma 缩放对照：其余全不动，只缩放共享后验

| sigma 倍数 | case 3650 | case 3651 | case 3652 |
|---|---|---|---|
| ×1.00 | 超时 62/101 | 超时 59/101 | 超时 75/101 |
| ×0.50 | 超时 79/101 | **到达 80/88** | 超时 99/101 |
| ×0.25 | **到达 85/85** | 超时 101/101 | 超时 101/101 |
| ×0.00 | 超时 98/101 | 超时 100/101 | 碰撞 19/20 |

（分数为求解成功步/总步）

**求解成功率随 sigma 单调上升：60–75% → 78–98% → 100%。**
sigma×0.25 时求解器不再失败。sigma×0 时开始出现碰撞，符合预期。

## 找到的是一条影响链，不是全部原因

**我自己的表里就有直接反例：sigma×0.25 时 case 3651 与 3652 都是 101/101 步
求解成功，却仍然超时。**所以至少存在两个独立的问题：

1. 求解失败导致制动；
2. **求解成功，但控制器没有有效向目标推进。**

第 2 条尚未解释。因此下面这条链只是**求解失败**的成因，不是导航失败的完整解释：

**共享后验 sigma 的量级 → HPIPM 以 MIN_STEP 终止 → 分支返回 QP_FAILURE
→ 无可用分支 → 退化为制动**。

另外两处措辞也要收窄：

- `MIN_STEP` 只说明**数值求解提前终止**，既不能证明也不能排除问题不可行。
  先前写「排除了不可行」过头了，正确说法是「返回值不是 INCONS_EQ」。
- 空场景通过只验证了**基本链路**可用，不能排除有障碍时的路径或初始化问题。

已逐项排除的其它解释：

| 曾经的假设 | 排除依据 |
|---|---|
| 问题不可行 | HPIPM 原始状态是 MIN_STEP，不是 INCONS_EQ |
| 解里出现 NaN | 原始状态是 MIN_STEP，不是 NAN_SOL |
| `res_eq > 1e-2` 规则 | 失败步 0% 触发 |
| 脏数据进入求解器 | 参数与热启动非有限个数均为 0 |
| 哑元障碍造成量级跨度 | max_obstacles=5 无哑元，失败数几乎不变 |
| 分支赋值重置 QP 内存 | 独立版打开同一重置，结果逐项相同 |
| 引导层 | 统一界限后 `tmpc_contour` 与完整 T-MPC++ 表现一致 |
| 参考路径不合适 | 空场景下同一条直线参考可跑满 v_max 并到达 |
| Contouring 权重 | 上游权重在空场景正常；放开权重反而让机器人跑离目标 10.8 m |
| 引导规划器时域失配 | 对齐 T=4.0/N=16/v≤1.0 后失败率不变 |

## 这意味着什么

**成立**：在本项目遮挡感知后验的量级下（4 秒前瞻处 sigma 达 2–4 m），
T-MPC++ 的椭圆约束使相当比例的控制步无法在其求解器容差内收敛，退化为制动。
把同一后验按比例缩小即可让**求解**恢复正常。

**但求解恢复正常并不等于导航恢复正常**：sigma×0.25 下 3651、3652 零求解失败仍然超时。
「解出来却走不到」是一个独立的、尚未诊断的问题。

**不成立**：不能说这两个方法「不会导航」——空场景与小 sigma 下它们都能到达目标。

**不成立**：不能说「我们的方法更好」。两边对同一份 sigma 的**风险语义不同**：
本项目把它放进带 0.75 概率预算的 chance constraint；T-MPC++ 把它当作按
`chi(risk=0.05)` 放大的硬椭圆。比较的正确做法是**各方各自调自己的风险参数、后验保持原样**，
比较实测的安全—效率前沿，而**不是把定义不同的风险数值机械对齐**。
求解失败、超时与碰撞全部计入评分，不得从分母剔除。

**也不得**通过人为压小 sigma 来「修好」成绩——上面的缩放只是诊断手段。

## 本轮修好的接入缺陷（都不是靠改我们的方法）

1. 参考路径每回合固定一次（此前每步重建，Contouring 的进度被清零）。
2. 参考路径点距按 `horizon × dt × v_max × 1.5 / num_segments` 反算，
   使跟踪窗口（5.7 m）覆盖时域可走距离（4 m）；此前窗口只有 2.6 m。
3. 行人半径加上本项目的 `human_margin`，与我们自己的碰撞预算一致。
4. 两个诊断副本补上统一执行器界限（此前跑作者原生 `v ∈ [-0.01, 3.0]`，对比被污染）。
5. 引导规划器时域与 MPC 对齐（T=4.0、N=16、v≤1.0、a≤2.0）。
6. 关闭上游的实验计数与数据记录（`num_experiments` 默认 5 会让一个 bridge 进程
   跑满 5 回合后 abort；数据目录写死到不存在的 `/workspace`）。
7. SH-MPC 与 T-MPC++ 对同一协议字段的语义相反（增量 vs 边缘），已分别给出正确形式。
8. SH-MPC 的风险配置会被源码乘 2，扫描档位按有效风险重建。

## 第二个失败模式：解出来却走不到（已诊断，未修复）

在 sigma×0.25、**零求解失败**的设置下追踪逐步状态：

| case | 最近到目标 | 之后 | 样条进度 s(0) / 全长 |
|---|---|---|---|
| 3651 | **0.84 m**（第 84 步） | 速度衰减到 0.001，停住 | 7.62 / 8.06 |
| 3652 | 1.02 m（第 67 步） | 又走开，末了 2.54 m | 7.18 / 8.06 |

**机器人开到离目标 0.8–1.0 m 处停住或飘走，样条进度同时在末端附近停滞。**
这与求解失败无关：这两个回合每一步都求解成功。

一个可查证的定义差异：`Contouring::isObjectiveReached` 用
`distance(pos, spline_end) < 1.0`，即**上游自己认为 1.0 m 即算到达**；
而本项目 `env.config` 的 `success_radius = 0.25`。**两边「到达」的定义差 4 倍。**
contouring 在样条末端退化为点镇定，收敛精度与上游阈值相称，但达不到 0.25 m。

尝试的修法：把参考路径沿同一直线延伸过目标 2 m，使机器人「穿过」目标区而非停在样条终点
（不含任何行人信息）。**结果不稳定**：sigma×0.25 下 3650 由超时转为到达（44 步），
但 3651 仍超时、3652 变差。**因此不作为定论，仅记录。**

## 当前对两个方法的判断（收窄）

已识别**两个独立的失败模式**：

1. **求解失败 → 制动**，随共享后验 sigma 的量级单调变化（60–75% → 100% 求解成功）。
2. **求解成功但走不到目标**，表现为逼近到 0.8–1.0 m 后停滞或飘离，样条进度停在末端附近。

第 1 条有完整的逐层证据链；第 2 条只做到现象定位，尚未找到可靠修法。
**两者都不足以支持任何关于方法优劣的结论。**

## 自查发现：我的埋点改坏了 SH-MPC（已还原）

为给 SH-MPC 单独取证，我在 `mpc_planner_modules/src/scenario_constraints.cpp` 里
加了场景求解器的统计代码。它看起来只读，但实测**显著改变了 SH-MPC 的行为**：

| case | 埋点后 | 还原干净源码后 |
|---|---|---|
| 3650 | 4/101，路径 **0.00 m** | 14/101，路径 1.38 m |
| 3651 | 6/101，路径 **0.00 m** | **reach_goal**，43/67，路径 **7.85 m** |

**SH-MPC 实际上能到达目标。**先前那条「SH 单独取证：全部 0.00 m、分支全 QP_FAILURE」
的结论**作废**，那是我的埋点造成的。

影响范围：

- **风险扫描是在埋点之前跑的，那组 SH 数据有效。**
- 埋点之后产生的所有 SH 数字无效，已作废。
- 埋点代码存档于 `diag/scenario_constraints_instrumented.cpp` 备查，不再使用。

教训记在这里：SH 的场景求解器路径不适合侵入式埋点；对它的诊断改用不修改源码的方式
（bridge 侧统计、日志解析），并且**每次给外部方法加诊断代码后，必须先与埋点前的
同一 case 对拍，确认行为未变，再采信任何数字**。

## 埋点无害性对拍（新规程，两个方法都做了）

**规程**：给外部方法加任何诊断代码后，先建一个只把被埋点文件还原为干净源码、
其余（配置、执行器界限、参考路径、构建选项）完全一致的副本，在同一批 case 上对拍，
并**用同一构建重复多次先量出噪声带**，再判断差异是否来自埋点。

| 方法 | 结论 | 依据 |
|---|---|---|
| T-MPC++ | **埋点无害** | `tmpc` 与 `tmpc_pristine` 在 case 3652 上重复 3 次均为 96/101、5.28 m，完全一致；3653 上两者各自都有 ±4 步的残余波动。先前观察到的 66 vs 96 是残余随机性，不是埋点。 |
| SH-MPC | **埋点有害，已还原** | 埋点后路径 0.00 m，还原后 case 3651 达 7.85 m 并 `reach_goal`，差异远超噪声带。 |

因此：**T-MPC++ 的埋点诊断数据可用；SH-MPC 的一律改用非侵入方式**（bridge 侧统计、
日志解析），埋点版本存档于 `diag/scenario_constraints_instrumented.cpp` 不再使用。

## SH-MPC 的干净结果（8 个开发 case，circle 5 人）

| risk（自身单位） | SR% | **CR%** | TR% | 路径均值 | 求解成功% |
|---|---|---|---|---|---|
| 0.02 | 0.00 | **0.00** | 100.00 | 2.24 m | 23.5 |
| 0.05 | 0.00 | **0.00** | 100.00 | 5.08 m | 52.5 |
| 0.10 | 0.00 | **0.00** | 100.00 | 6.91 m | 80.0 |
| 0.20 | 0.00 | **0.00** | 100.00 | 7.43 m | 78.6 |
| 0.35 | **25.00** | **0.00** | 75.00 | 8.89 m | 89.7 |

**每一档碰撞率都是 0**，risk 越大走得越远、求解成功率越高、成功率上升，单调且符合
场景法安全时域 MPC 的预期行为：**安全但保守**。前沿尚未出现拐点，最优点在 0.35 之外，
因此把扫描向上延伸（SH 的 config 必须 < 0.5，因为源码会乘 2）。

这与埋点期间那组「全部 0.00 m」完全相反，再次说明那组数据必须作废。

## SICNav-np 的 10 人：与 JMID 同一瓶颈

按指令对 np 与 CVG 分别取证，不把 JMID 的超时直接推广过去。结果：

| 变体 | 10 人 | 退出码 | 峰值内存 | 编译的 C 文件数 |
|---|---|---|---|---|
| SICNav-np（IPOPT/CasADi，**不经 acados**） | **1:30:00 超时** | 124 | 432 MB | **0** |
| SICNav-CVG（acados） | 进行中 | | | |

**np 走的是完全不同的求解链路（IPOPT，不用 acados），10 人同样卡死在 CasADi 符号构造，
一次编译都没进入。**因此该瓶颈不是 acados 或 Diffusion 特有，而是
ORCA-KKT 双层问题的符号规模本身。

仍不能推广成「所有大规模配置不可行」：未定位符号构造的耗时热点，
也未尝试作者提供的降规模选项（如缩短 `orca_kkt_horiz`）。

---

# 第 13 节执行令：验收阶段记录（2026-09-09）

## 13.1 预检快照

`snapshot/preflight.json`：六仓库 HEAD/脏状态/未跟踪、八个工作副本各 11 项产物 SHA256、
`ldd` 实际解析出的相关 `.so` 及哈希、四个项目模块的绝对解析路径、两个 conda 环境与编译器版本。
`mpc_planner`/`scenario_module`/`ros_tools`/`guidance_planner` 工作区均干净。

## 外部阻塞：4090 不能承载现代 MPC

磁盘 1.1 GB 可用（98% 满，`/root/poult` 独占 14 GB，非本项目）、**未装 ROS**、**未装 acados**、
同事 ECG 任务在跑。装齐需十余 GB。第 13.7 节「两机都使用已有隔离环境」的前提不成立。

第 13.7.4 节禁止一机跑我方、另一机跑对手（硬件与方法混杂），而 4090 只能跑 Bayes，
**故本轮全部在本地执行**（Ryzen 7 5700G，8 物理核 / 16 线程，15 GB），2 个 worker。

## 13.4-A 目标与路径链路验收

**`isObjectiveReached` 在 bridge 调用链上出现 0 次**，已用 grep 确认。
因此本回路里没有任何方法能提前宣布到达；未进入 0.25 m 即为未收敛。
**先前把「上游 1.0 m 判据」当作解释的说法作废**——那个函数根本没被执行。

无障碍探针（两个方法结果几乎相同，符合预期：无障碍时都退化为同一 contouring 问题）：

| 探针 | 首次进入 0.25 m | 终点距离 |
|---|---|---|
| 长距离，朝向对准目标 | 8.75 s | 0.17 m |
| 短距离 1.5 m | 2.00 s | 0.12 m |
| 最后一米 0.9 m | 1.25 s | 0.18 m |
| 障碍移开后继续 | 10.75 s | 0.14 m |
| 长距离，朝向垂直目标 | **从未** | 2.52 m |
| 长距离，朝向背对目标 | **从未** | **8.00 m（原地未动）** |

**结论一**：目标链路本身是通的。朝向正确时两个方法都能进入共同的 0.25 m 目标区，
包括「障碍移开后继续前进」。**先前「它们够不到 0.25 m」的判断，在朝向正确时不成立。**

**结论二（新的适用边界）**：朝向背对目标时原地不动。共同界限 `omega_max = 0.8 rad/s`
与 N=16 的 4 s 时域最多转 3.2 rad ≈ 183°，刚好卡在 180° 掉头边缘，
于是「不动」比「先朝反方向开」代价更低。

对正式实验不构成不公平：`run_episode` 对**所有**方法都把初始朝向设为指向目标（已登记）。
但这是需要记录的边界，并提示回合中途需要大幅转向时可能同样受限。

## 13.4 其它已通过项（与当前二进制哈希一致）

`bridge/interface_audit.log` 十项全通：状态转录、预测时间索引（误差 8.9e-16）、
全部检测送达、两种不确定度语义、角速度→角增量恰好一次 dt、重复运行离散度已登记。

## 13.4-C 求解与埋点验收：失败步完整迭代已抓到，两个旧结论被推翻

bridge 现在在失败步转储完整迭代：每个 stage 的 `x`、`u`、`z`、`pi`（等式乘子）、
`lam`（不等式乘子）、`sl`/`su`（松弛），长度全部向 acados 现场查询而非假定。
本版 acados 的 `ocp_nlp_out_get` 没有 `t` 字段，已在转储里显式标注缺失，不静默省略。
仅在 `MPC_BRIDGE_FAILURE_DUMP` 指定文件时启用；正式运行不设该变量。

**无害性已按自定规则验证**（上次我的埋点悄悄把 SH-MPC 打死过，路径 7.85 m → 0.00 m）：

| 臂 | 埋点关闭（3 次） | 埋点开启（2 次） | 判定 |
|---|---|---|---|
| tmpc c01_author | 路径 8.31 / 6.74 / 7.25 m | 7.76 / 6.85 m | 落在带内 |
| shmpc c01_author | 4.19 / 4.58 / 4.27 m | 4.52 / 4.47 m | 落在带内 |

SH-MPC 仍在导航（4.2–4.6 m），未重复上次的错误。

### 抓到 3640 个失败步后的两个结论

**一、失败不是 NaN。** T-MPC++ 1491 条、SH-MPC 2149 条失败步，
**迭代中含非有限值的为 0 条**。此前"NaN 来源"这条线可以正式关闭。

**二、（已撤回，见下）关于失败步求解器状态分布的说法不成立。**

| 臂 | qp_status | hpipm_raw | acados_status | 次数 | 占比 |
|---|---:|---:|---:|---:|---:|
| tmpc | 0 | 0 | 0 | 1335 | 89.5% |
| tmpc | 1 | 1 | 0 | 126 | 8.5% |
| tmpc | 0 | −99 | −1 | 30 | 2.0% |
| shmpc | 0 | 0 | 0 | 1814 | 84.4% |
| shmpc | 1 | 1 | 0 | 174 | 8.1% |
| shmpc | 0 | −99 | −1 | 125 | 5.8% |
| shmpc | 1 | −99 | −1 | 36 | 1.7% |

**撤回理由（同一批数据自证）**：上表读的是主求解器的 `_info`。
统计 `nlp_res` 时发现它的值全部在 6.9e-310 量级——这是未初始化内存的非规格化数，
且 `kkt_norm_inf` 恒为 0。也就是说**失败步上这个结构体根本没被写入**。

原因在 `guidance_constraints.cpp:425`：`FindBestPlanner()` 返回 −1 时函数提前返回，
`_solver->_info = best_solver->_info` 这一句不执行。而 `selected_branch` 在全部
1491 条失败步上都是 −1，所以 `qp_status`、`hpipm_raw`、`acados_status`、`nlp_res`、
`kkt` 这些字段读到的是默认值或残留值，**不是本步的求解器状态**。
上表里占 89.5% 的「0/0/0」因此只说明「没写过」，不说明「QP 求解成功」。

这正是先前登记的陈旧 `_info` 陷阱，只是这一次它同时污染了我自己的归因。
可信的替代来源是 `guidance_constraints.cpp:381-420` 在 `FindBestPlanner()`
**之前**就写入的分支级字段（`br_total/br_enabled/br_ok/br_worst_exit/br_status/
br_res_eq/orig_*`），它们已经在每一条 STEP 回复里，只是 Python 侧此前丢弃了。
D1 跑完后改 Python 解析（不动二进制），重新取一次可信分布。

T-MPC++ 的 `selected_branch` 在全部 1491 条失败步上都是 −1，即
`FindBestPlanner()` 认定没有任何分支可用——这同时说明此时读到的 `_info` 是上一步的陈旧值，
先前登记的陈旧字段陷阱在真实数据上得到确认。

**机制推断（待 D1 数据佐证，不作为结论）**：`warmstart_with_mpc_solution: false`
意味着永远用引导轨迹热启动，而引导轨迹对 unicycle 模型并非动力学一致；
再加上我们把 `integrator_step` 从作者的 0.2 s 改成本项目的 0.25 s，
单次 RTI 迭代后的等式残差更大。D1 的 N=16/24/32 候选会给出时域方向的证据。

## 13.3 原生模型与环境推进的一周期偏差（已量化）

`snapshot/model_gap.json`，`model_gap.py`。同一条命令下比较两种离散化：

- **原生**：`x'=v cos psi, y'=v sin psi, psi'=w, v'=a`，求解器在时域上积分；
  bridge 下发的是上游自己下发的那一对——`getSolution(1,"v")` 与 `getSolution(0,"w")`。
  已核对上游三个节点（`ros1_jackalsimulator.cpp:185-186`、`ros2_jackalsimulator.cpp:130-131`、
  `ros1_rosnavigation.cpp:313-314`）完全一致，**不是我们发明的命令**。
- **环境**：CrowdSim `ActionRot` 先转后走，`theta_1 = theta_0 + r`，
  再以恒定 v 沿新朝向平移。

125 个网格点（v0 × a × w 取注册值）：

| 量 | 中位 | p95 | 最大 |
|---|---:|---:|---:|
| 位置偏差 | 32.06 mm | 63.97 mm | 67.24 mm |
| 朝向偏差 | — | — | 1.4e-16 rad |

朝向完全一致；偏差全部来自「边转边走 vs 先转后走」的位置积分，
以及速度在这一周期内是斜坡还是阶跃。最大 67 mm 相当于**一步最大位移的 26.9%**，
也是 0.25 m 成功半径的 26.9%。

这是两个对照方法为进入本环境而付出的、可量化的单周期预测误差；
贝叶斯规划器的 rollout 与环境逐位一致，不带这项误差。环境对所有臂是同一个仲裁者，
不因此改判任何一方，但主表必须披露这一条。

## 13.5 数据划分、候选冻结与执行器（2026-09-09 晚）

### case 登记表：按实际布局哈希，不按 case 整数

`snapshot/case_registry.json`。每个 case 都实际 reset 环境一次，
哈希机器人起点/终点/v_pref/半径与每个行人的同样五项，再检查重复与交集。

| 划分 | case 区间 | 布局数 | 重复布局 |
|---|---|---:|---|
| debug（永久禁止进 T） | 3650–3749 | 100 + 100 | 无 |
| D1 | 4000–4039 | 20 circle + 20 square | 无 |
| D2 | 4100–4199 | 50 + 50 | 无 |
| T（六场景各 100） | 5000–5099 | 6 × 100 | 无 |
| L（六场景各 5，取自 debug） | 3650–3654 | 6 × 5 | 无 |

六项交集检查（D1×D2、D1×debug、D2×debug、D1×T、D2×T、debug×T）全部为 0。
未发现生成器周期回绕。T 的布局只生成并校验，未跑策略、未看结局。

### 候选清单：第一条 D1 结果之前锁定

`snapshot/candidates.json`，三家族各 12 个。两个对照家族的四个轴：

- **风险**：作者 0.05 / 0.20 / 各自当前工作点（tmpc 0.35、shmpc 0.42），
  按各方法自身 yaml 单位，同时登记 SH 内部翻倍后的有效值；
- **物理预测时域**：N=16/24/32，即 4/6/8 s。N 在代码生成期固定，
  因此是三个分别生成、分别哈希的工作副本，不是改个 yaml；
- **原生跟踪与终端权重**：原生 `contour 0.05 / terminal 10`、
  收紧 `0.5 / 50`、极紧 `2.0 / 100`；
- **参考路径是否穿过目标**：0 m 与 2 m。

第四个轴是被追踪逼出来的：此前 `PATH_OVERSHOOT=2.0` 是我硬写的常数。
逐步追踪显示它让机器人**开过目标继续走到样条末端**（case 3700，最近点 0.47 m，
从未进 0.25 m，随后漂到 y=5.6）；而 0 m 又让它停在目标前。
根因是作者原生权重 `contour: 0.05` 对 `lag: 0.75`——按 6 m 宽车道调的，
横向几十厘米无所谓，对 0.25 m 目标球致命。这个量是接入参数，不该由我手调，
所以进扫描交给规则选。

Bayes 家族同样 12 个：五个注册风险点 × 时域 × 求解预算（256/512/1024 样本、
4/6 次迭代），不改风险模型、不改滤波算法、不加代价项。

### 选参规则（写在代码里，不是事后判断）

`risk_sweep/front.py`。D1 每家族保留三点后去重：碰撞率最低的安全端、
惩罚时间最短的效率端、以及注册折中规则（自身最低 CR +2 个百分点内取惩罚时间最短）。
并列一律按 CR 低 → SR 高 → 清单序号。D2 用注册规则定最终工作点。
全部候选前沿与未入选原因一并落盘。

### 正式执行器

`bridge/run_bridge_cohort.py` 重写为 `plan / precheck / run / aggregate`。
25 case 一块；块内 bridge 常驻，reset 开销计入回合；块先写 `.partial`，
通过 case 集合 / 条数 / block_id 完整性检查后才原子改名。
基础设施失败原配置重试一次并保留失败尝试，第二次仍失败记 error、罚时 25 s、
CR 记未知。**shell 里猴子补丁 `BridgeController.act` 的做法已彻底移除。**

发现并记录的一个健壮性问题：杀掉 pool 父进程后，`spawn` 出来的 worker 会被
重新挂到 init 下继续拉起 bridge，必须显式清理。块级原子性保证了不会留下损坏数据。

### 统计模块自检

`test_analysis.py` 用已知答案校验 `analysis.py`：Holm 校正（含单调性）、
配对 cluster bootstrap（相同两臂差值与 CI 恰为 0；已知常数差恰好复原）、
场景等权（1 布局的场景与 99 布局的场景同权）、
四个审计结局互斥且合计 100%。全部通过。

## D1 逐步记录给出的失败画像（与求解器诊断相反的结论）

取 `tmpc:c03_current` 的 dev_circle 块（20 回合，14 个超时），从逐步记录算：

| 量 | 中位 |
|---|---|
| 每步实际推进 | 121.2 mm（上限 250 mm，即约 0.48 m/s） |
| 逐步求解成功率 | 96.5% |
| 回合结束时距目标 | 2.83 m |
| 最后 20 步几乎不动（真停车） | 3 / 14 |
| **一直在动但没到达** | **11 / 14** |

**这修正了失败画像的重点。** 在当前工作点上求解成功率是 96.5%，
所以求解失败不是主导因素；主导的是**机器人全程在动、速度也不低，却收敛不到目标**。
25 s × 0.48 m/s ≈ 12 m 的行程，覆盖 8 m 的任务绰绰有余，因此也不是速度受限。

这与更早那次逐步追踪一致（开过目标、绕开、再漂走），指向**跟踪与终止**，
而不是求解。根因仍是原生权重 `contour 0.05` 对 `lag 0.75`：
沿路径推进的权重是横向贴合的 15 倍。候选 c07/c08/c09（contour 收紧到 0.5 / 2.0，
terminal_contouring 50 / 100）正是为这一条设的，D1 会直接给出答案。

先前从失败步转储得到的求解器画像因此也要放到正确位置：那些失败步是少数派，
不能用来解释多数超时。

## D1 tmpc 家族完整结果：跟踪权重假设被否定

12 个候选 × 40 个五人布局（circle/square 各 20），两场景等权 macro：

| 候选 | contour | 穿目标 | N | risk | SR% | CR% | 罚时s | 越界步 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c01_author | 0.05 | 0.0 | 16 | 0.05 | 5.0 | 15.0 | 24.56 | 2 |
| c02_author_reach | 0.05 | 2.0 | 16 | 0.05 | 15.0 | 15.0 | 23.48 | 1 |
| c03_current | 0.05 | 2.0 | 16 | 0.35 | 22.5 | 15.0 | 22.03 | 2 |
| c04_current_stop | 0.05 | 0.0 | 16 | 0.35 | 7.5 | 15.0 | 24.58 | 8 |
| c05_risk20 | 0.05 | 0.0 | 16 | 0.20 | 12.5 | 12.5 | 24.22 | 9 |
| c06_risk20_reach | 0.05 | 2.0 | 16 | 0.20 | 10.0 | 15.0 | 23.74 | 19 |
| c07_tight | 0.50 | 0.0 | 16 | 0.20 | 12.5 | 12.5 | 24.25 | 8 |
| c08_tight_reach | 0.50 | 2.0 | 16 | 0.20 | 12.5 | 15.0 | 23.44 | 16 |
| c09_verytight | 2.00 | 0.0 | 16 | 0.20 | 12.5 | 12.5 | 24.24 | 10 |
| **c10_n24** | 0.05 | 2.0 | **24** | 0.20 | **25.0** | **10.0** | 22.39 | 6 |
| **c11_n24_tight** | 0.50 | 2.0 | **24** | 0.20 | **25.0** | **10.0** | 22.33 | 12 |
| c12_n32_tight | 0.50 | 2.0 | 32 | 0.20 | 10.0 | 15.0 | 24.32 | 11 |

**我先前的跟踪权重根因假设被数据否定。** `contour` 从 0.05 → 0.50 → 2.00
（c05 / c07 / c09，其余条件相同）得到的 SR 是 12.5 / 12.5 / 12.5，CR 全是 12.5，
**完全没有差别**。所以「原生 contour 0.05 对 lag 0.75 导致横向误差、错过目标球」
这条归因不能成立，至少在这个权重范围内不成立。

实际有效的是另外两个轴：

1. **参考路径是否穿过目标**：c01→c02（5.0→15.0）、c04→c03（7.5→22.5），
   同风险同权重下一致地大幅改善。这是 N=16 下最强的单一因素。
2. **预测时域**：N=24 的两个候选同时拿到最高 SR（25.0）和最低 CR（10.0）；
   N=32 反而退回 10.0。6 s 是个拐点，不是越长越好。

风险方向也有效（同 overshoot 同权重下 0.35 的 c03 为 22.5，0.20 的 c06 为 10.0），
但样本量不足以分离它与噪声。

**这正是「不许手调、必须扫描」的价值**：我基于一次逐步追踪得出的机制推断听起来
自洽，扫了才知道它是错的。当前最好的 tmpc 工作点是 25% SR / 10% CR——仍然远低于
贝叶斯在五人场景的表现，但这是在契约正确、接入已验收前提下的真实数字。

注意 40 布局单次 repeat 的噪声不小（两方法都是非确定性的），12.5 与 25.0 之差
接近但未必超过噪声。D2 用 100 个新布局 × 2 repeat 复查。

## 2026-09-09 Codex接管：本地串行守护队列

- 保留CC正在运行的新D1（PID 3762653），不清除已完成记录。实际新划分是D1 11000–11039、D2 11100–11199、L 11200–11204、T 12000–12099；上文旧划分仅为历史记录。
- 守护进程PID 3766871，入口 `pipeline.sh`，自动接D1、D2、冻结、T-OCC、T-FULL、L及统计。后续1 worker、4内部线程；D1旧任务2 worker收尾。全部在本地CPU，不重写GPU算法。
- 修复：env.sh切换cwd后相对脚本失效；L误调用旧实验耗时脚本；失败块仍退出0；旧partial文件被继续追加；缺块仍可选参汇总；error被剔除分母；L的1线程标志未真正限制OpenMP；IPC无截止时间；多个bridge覆盖同名日志。
- 冻结后校验源码、候选、布局、求解器及实际加载库哈希，复制可恢复输入。D1旧源码保留于 `takeover_20260909/original/`。保留已完成旧D1作为粗选开发，不将其争用耗时当硬件基准。
- 9项断点/统计/IPC回归测试通过；两个真实求解器接口复测通过，包括20人字段容量；三个家族的新执行器各1回合SMOKE通过，保存在单独SMOKE队列，不混入正式数据。
- 布局历史扫描和实际初始状态哈希复验通过；每个新回合step0再次核对布局哈希。
- 动力学诊断已完成：40布局、四变体均100% SR/0% CR，时间变化小。这只是Bayes敏感性证据，不能据此声称对手完全不受模型失配影响。其他适配限制见 `snapshot/acceptance_takeover.json`。
- 每90分钟自动追加 `answer.md`。自动程序只能记录进度，不能假冒人工回复文档中新问题；新增讨论标记待交互复核。
- 实时状态 `snapshot/pipeline_status.json`；阶段日志 `logs/`；最终导出 `final/`。只有全部计数、原始记录、哈希及独立重算通过才记TEST_DONE；异常或对比不显著如实保留。

---

# 交接与只读复核（2026-09-10 00:2x，CC）

执行链已由 `supervise.py`（PID 3766871）接管，串行跑 T-OCC → T-FULL → L → 统计。
我转为监控与独立复核，不重启实验、不重新调参、不改论文文件。
我自己此前挂的后台任务已全部停止；动力学交叉验收在被停之前（20:11）已完整跑完。

## 复核结论

### 1. 布局身份：通过

正式测试用 **12000–12099**，D1 用 11000–11039，D2 用 11100–11199。
全部位于历史占用（最高 10099）之上。登记表里的历史扫描门禁记录
`historical_case_count = 4554`、`historical_clashes = {}`。

六个场景的配置哈希两两不同（`baseline_square` 与 `dense_square`/`large_square`
各自独立），旧的「两个方形场景互相覆盖」不再存在。

### 2. 冻结配置：与 D2 选点一致，规则应用无误

| arm | 家族 | 候选 | N | 关键参数 |
|---|---|---|---:|---|
| bayes_full | bayes | c05_p4_n16 | 16 | point=4, pop=512, iters=4 |
| bayes_r1 | bayes | c05_p4_n16 | 16 | 同上 + existence_override=1.0 |
| tmpc_shared | tmpc | c03_current | 16 | risk 0.35, contour 0.05, overshoot 2.0 |
| shmpc_shared | shmpc | c10_n24 | 24 | risk 0.20（有效 0.40），overshoot 2.0 |

逐条验算注册规则（自身最低 CR +2pp 内取惩罚时间最短）：

- bayes：c05(CR 3.0) 与 c06(CR 1.0) 都在合格线 3.0 内，罚时 9.93 < 10.11 → c05。**规则正确**。
- tmpc：仅 c03 入选 D2，直接选中。
- shmpc：c08 的 CR 28.0 超出合格线 26.5，出局；c10 入选。**规则正确**。

**必须披露的一点**：bayes 被规则选中的 c05_p4_n16，在 SR（97.0 vs 99.0）与
CR（3.0 vs 1.0）两项上都**劣于**落选的 c06_p2_n24，仅因罚时快 0.18 s 而胜出。
这是事前注册规则的机械结果，不得事后改选——但读者会看出 c06 在两个安全/成功
指标上占优，必须在正文写明，并把完整前沿放补充材料标 DEV。
另外 c05 是**五个注册风险点里最激进的一个**，与本项目历史工作点（point 2）不同；
主表的贝叶斯臂因此不等同于前几章的配置，这一条也要写明。

### 3. 资源预算：已被正确改写，非漂移

`frozen.json` 记录 workers=1 / threads=4，并显式登记 D1 是 2-worker 的遗留开发数据、
`OMP_THREAD_LIMIT` 未强制、**不作硬件基准**。D2 与全部正式队列强制四线程，
L 另含单线程剖面。逐回合记录里的 `threads` 字段实测全为 4。前后一致。

### 4. 配对与重复聚类：通过

已落盘的 T-OCC 回合含 repeat 0/1/2；配对键为
`scene_id | scene_config_hash | case_id | condition | repeat`，另存 `layout_sha256_16`。
`analysis.py` 先按 layout 对 repeat 求均值再做场景内 layout 级 cluster bootstrap，
这一行为已由 `test_analysis.py` 用已知答案校验（相同两臂差值与 CI 恰为 0；
已知常数差恰好复原；1 布局的场景与 99 布局的场景同权）。

### 5. SH-MPC 跨时协方差：已量化披露，口径正确

`snapshot/joint_disclosure.json`：边缘按构造相同，但随机游走在**所有**测试滞后上
系统性低估时间相关，最大低 0.63。`acceptance_takeover.json` 的 scope 写作
"Shared-detector/tracker marginal predictions, adapted planners... not original
end-to-end systems or isolated Bayesian necessity"，与我的披露一致。
**主表不得写成原生端到端系统比较。**

## 动力学不对称交叉验收：结论为「非因果」

单周期偏差分解（中位，mm）：朝向项 6.24，速度斜坡项 31.25，合计 31.41（最大 66.0）。
主项是速度斜坡而非朝向。N=16 时域末端累积漂移中位 62 mm、p95 118 mm、最大 136 mm，
相对 4.0 m 可达距离与 0.25 m 目标半径都不大。

经验对照（D1 开发布局 40 个，贝叶斯规划器背上对手的失配，其余一切不变）：

| 配置 | SR% | CR% | 罚时s |
|---|---:|---:|---:|
| 匹配环境（正式） | 100.0 | 0.0 | 9.95 |
| 仅朝向失配 | 100.0 | 0.0 | 9.92 |
| 仅速度斜坡失配 | 100.0 | 0.0 | 10.09 |
| 两项都失配 | 100.0 | 0.0 | 9.90 |

**把对手的全部模型失配加到我方规划器上，SR 与 CR 纹丝不动。**
所以这个不对称是已披露但**非因果**的次要项，不能用来解释对手的低分。
反过来也说明：对手的低分需要别的解释，不能归到「环境对它们不公平」。

## 验收链路的独立性核查（监控期只读）

`test_takeover.py --verify-final` 的独立性是真的，不是重读汇总文件：
它调用 `run_bridge_cohort.validated_rows(queue)`，该函数**重新展开任务清单**、
逐块校验完整性（缺块即报错），再从逐回合 JSONL 重读，然后独立重算 SR 与罚时，
与 `final/main_summaries.json` 比对；另检查每臂 1800 回合、
`(scene_id, layout_hash, repeat)` 三元组无重复、L 队列 240 回合齐全。

核了两处可能误报的地方，都没问题：

- **容差**：`verify_final` 用 1800 个回合的平坦均值对比 `analysis.py` 的六场景嵌套
  macro，容差 1e-12。设计是平衡的（6 场景 × 100 布局 × 3 repeat），两者数学相等，
  但浮点路径不同。4000 次模拟实测最大差 **2.2e-16**，远在容差内；罚时项差 0。不会误报。
- **产物链**：`analysis.py` 已在 20:36 被扩展，除我原有的六份导出外还写
  `final/main_summaries.json`（`supervise.py` 的 `final_report` 与 `verify_final` 都读它），
  并把独占窗口耗时与评测负载耗时分成 `latency.csv` / `latency_evaluation_load.csv`
  两份。链路一致，不会在最后一步断掉。

## T-OCC 完成（7200/7200）与密度崩溃的性质判定

独立重读通过：`validated_rows("T-OCC")` 重新展开任务清单、逐块校验、从逐回合
JSONL 重读，得到 7200 回合；四臂各 1800；每臂六场景各 300、repeat 0/1/2 各 600、
布局 600；唯一配对键 7200 = 记录数；`code_sha256` 全程单一值；`threads` 全为 4；
error 0；缺 `steps` 0。越界步合计 27（约 43 万步中），最大超出 0.0210 rad。

### 密度崩溃：不是接线故障

| 场景 | 人数 | tmpc_shared | shmpc_shared |
|---|---:|---|---|
| baseline_circle | 5 | 10.53 m / 93.2% | 3.74 m / 33.0% |
| dense_circle | 10 | 7.94 m / 80.8% | 0.36 m / 5.8% |
| large_circle | 12 | 10.40 m / 87.1% | 2.05 m / 34.7% |
| baseline_square | 10 | 3.10 m / 52.0% | 0.04 m / 3.4% |
| large_square | 20 | 0.71 m / 27.7% | 0.02 m / 1.5% |
| dense_square | 20 | 0.16 m / 10.3% | 0.00 m / 0.2% |

（平均路径长度 / 逐步求解成功率）

`baseline_circle` 里 T-MPC++ 走完 10.53 m、求解成功 93.2%，证明协议、参考路径、
障碍传递、动作回传全部正常。崩溃随密度单调恶化，且**方形场景显著劣于同人数圆形场景**。
因此这不是接线故障，不构成停止实验的理由。

### 但主表解读前必须回答的一个问题

两个对照的崩溃**共同**表现为求解成功率塌陷，而它们共用**我方后验**。
若我方后验在密集场景的椭球大到让它们的约束集按构造不可满足，
则主表测到的是「我方后验对它们的约束形式太宽」，而不是「我方信念处理更好」。
这两件事在论文里的含义完全不同。

关键的方法学差异在于：**我方用机会约束加风险预算，对手用卡方缩放后的硬椭球约束**。
同一后验下，硬约束在 risk=0.05 时远比机会约束保守。若确认如此，
主表必须写成「**共享后验下两种约束形式的比较**」，而不是信念处理优劣，
也不能据此宣称贝叶斯不可替代。

逐回合记录里存的是净空，没存协方差，所以现有数据无法回答。
`corridor_analysis.py` 已写好待命，跑完正式矩阵后执行（见下）。


## 2026-09-10T10:27:27+09:00 Codex 正式队列交付
状态：TEST_DONE。实际回合数：15280。
统计单位为场景内独立布局，三个规划重复不当作三个独立样本。
|条件|方法|SR%|CR%|TR%|ERROR%|罚时秒|
|---|---|---:|---:|---:|---:|---:|
|occluded|bayes_full|89.06|10.72|0.22|0.00|13.158|
|occluded|bayes_r1|89.39|10.33|0.28|0.00|13.128|
|occluded|shmpc_shared|2.50|42.28|55.22|0.00|24.792|
|occluded|tmpc_shared|11.33|45.83|42.83|0.00|24.030|
|full|bayes_full|91.00|8.61|0.39|0.00|12.825|
|full|shmpc_shared|3.11|41.44|55.44|0.00|24.740|
|full|tmpc_shared|11.50|46.39|42.11|0.00|24.068|
- bayes_full_vs_shmpc_shared|cr: 差 -0.3156, 95%CI [-0.3461, -0.2850], Holm p=0.0005999。
- bayes_full_vs_shmpc_shared|sr: 差 +0.8656, 95%CI [+0.8422, +0.8878], Holm p=0.0005999。
- bayes_full_vs_shmpc_shared|time: 差 -11.6336, 95%CI [-11.9400, -11.3065], Holm p=0.0005999。
- bayes_full_vs_tmpc_shared|cr: 差 -0.3511, 95%CI [-0.3833, -0.3183], Holm p=0.0005999。
- bayes_full_vs_tmpc_shared|sr: 差 +0.7772, 95%CI [+0.7472, +0.8072], Holm p=0.0005999。
- bayes_full_vs_tmpc_shared|time: 差 -10.8713, 95%CI [-11.2390, -10.4837], Holm p=0.0005999。
文件：`temp/modern/final/`；续跑/复现入口：`temp/modern/pipeline.sh`。
结果仅适用于共享跟踪预测的适配版本；执行动力学误差、SH跨时协方差近似与随机并行限制见snapshot/frozen.json。
不能用不显著证明安全等价，也不能单凭本表证明novelty或贝叶斯不可替代。
