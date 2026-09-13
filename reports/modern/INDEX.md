# 结果索引

一切都在 `/home/abc/temp/modern/` 下，除非另注。
本文件只列路径和它是什么，结论在 `STATUS.md`。

生成时间 2026-09-09。

---

## 先看这个

| 文件 | 是什么 |
|---|---|
| `STATUS.md` | **全部结论与登记事项**，592 行，按时间分段追加 |
| `INDEX.md` | 本文件 |
| `/home/abc/temp/ORDER_modern_baselines_CC.md` | Codex 给的执行指令（我依据的原文） |

---

## 一、SICNav 扩展性（已完成，定论）

| 文件 | 内容 |
|---|---|
| `scaling/jmid_scaling.log` | **汇总表**：5 / 10 / 20 人的退出码、峰值内存、耗时、结局 |
| `scaling/jmid_N5.log` / `.time` | 5 人的完整日志与资源统计（成功） |
| `scaling/jmid_N10.log` / `.time` | 10 人：4 小时超时，`cc` 未调用 |
| `scaling/jmid_N20.log` / `.time` | 20 人：同样 4 小时超时 |
| `scaling/run_scaling.sh` | 跑这三档的脚本 |

`.time` 文件里看 `Maximum resident set size` 与 `Elapsed (wall clock)`。

## 二、编译优化等级对照（已完成）

| 文件 | 内容 |
|---|---|
| `scaling/optflags.log` | **结论行**：`-O3` 与 `-O1` 的结局逐项一致 |
| `scaling/optflags_O3.log` / `optflags_O1.log` | 两次运行的完整日志 |
| `scaling/optflags_compare_N5.json` | 机器可读的对照结果 |
| `scaling/compare_opt_flags.py` | 对照脚本 |

## 三、全向 vs matched-unicycle（已完成，显著）

| 文件 | 内容 |
|---|---|
| `unicycle/kinematics_compare.log` | **汇总表 + 配对 McNemar**（统一界限 ω=0.8） |
| `unicycle/kinematics_compare.json` | 240 回合逐条明细 |
| `unicycle/kinematics_compare_omega1.0.log` / `.json` | 作废的一版（ω=1.0，界限未统一），存档备查 |
| `unicycle/compare_kinematics.py` | 对照脚本 |

## 四、全向路径回归（已完成，逐位一致）

| 文件 | 内容 |
|---|---|
| `unicycle/baseline_holonomic.json` | 改代码**之前**的 18 回合基线 |
| `unicycle/after_refactor.json` | 加 `_rollout` 钩子之后 |
| `unicycle/after_unicycle.json` | 接完 unicycle 动作路径、改过 `crowd_sim.py` 之后 |
| `unicycle/regression.py` | 回归脚本（三份 json 应当完全相同） |

比对方法：`diff <(python -m json.tool a.json) <(python -m json.tool b.json)`

## 五、unicycle 接口验收（已完成，11/11 通过）

| 文件 | 内容 |
|---|---|
| `unicycle/interface_test.py` | 展开与 CrowdSim 逐位一致、限幅、0/1/5/20 人、缺朝向报错、真闭环 |

重跑：`cd /home/abc/workspace/bayes_occ_mpc && ~/miniconda3/envs/crowdnav/bin/python /home/abc/temp/modern/unicycle/interface_test.py`

## 六、headless bridge（已完成）

| 文件 | 内容 |
|---|---|
| `bridge/mpc_bridge.cpp` | bridge 源码（两个工作空间各有一份副本） |
| `bridge/drive_bridge.py` | 独立闭环驱动，用于 native 冒烟 |
| `bridge/CMakeLists.txt` / `package.xml` | 新增 catkin 包的构建文件 |
| `bridge/apply_protocol.py` | 把 settings.yaml 改成本项目协议的脚本 |
| `bridge_logs/tmpc_bridge.log` / `shmpc_bridge.log` | bridge 自己的 stderr（崩溃诊断看这里） |
| `build_bridge_tmpc.log` / `build_bridge_shmpc.log` | 构建日志 |

## 七、现代 MPC 接入六场景（**数字不可用，见 STATUS.md**）

| 文件 | 内容 |
|---|---|
| `bridge/cohort.log` | 首轮 120 回合的汇总（风险水平未对齐，作废） |
| `bridge/bridge_cohort.json` | 同上，逐条明细 |
| `bridge/run_bridge_cohort.py` | 跑这一组的脚本 |
| `/home/abc/workspace/bayes_occ_mpc/modern_worker.py` | **适配器源码**（协方差→椭圆、参考路径、IPC） |

## 八、风险参数扫描（**正在跑**）

| 文件 | 内容 |
|---|---|
| `risk_sweep/sweep.log` | **实时进度**，每档一行；结束时打 `RISK_SWEEP_DONE` |
| `risk_sweep/risk_sweep.json` | 逐档、逐回合明细（边跑边写） |
| `risk_sweep/sweep.py` | 扫描脚本，含冻结规则 |
| `risk_sweep/tmpc_risk*.yaml`、`shmpc_risk*.yaml` | 五个风险档位的配置副本 |
| `risk_sweep/launch.sh` | 启动脚本（含 ROS 环境） |

看进度：`tail -f /home/abc/temp/modern/risk_sweep/sweep.log`

## 九、σ 归零诊断（**排队中**，扫描结束后自动开跑）

| 文件 | 内容 |
|---|---|
| `risk_sweep/sigma_diag.log` | 结果；结束时打 `SIGMA_DIAG_DONE` |
| `risk_sweep/diagnose_sigma.sh` | 诊断脚本 |

判定：若 sigma 归零后 T-MPC 立刻能导航，则冻结源于后验 sigma 的量级
（4 秒处 2.19 m），即两方风险语义不同；若仍不能，则是我的接入还有别的缺陷。

## 十、BNBRL+（资产阻塞，训练链已验证）

| 文件 | 内容 |
|---|---|
| `bnbrl_train_probe.log` | 2 进程训练探针（42 FPS） |
| `bnbrl_train_probe16.log` | 16 进程（194 FPS，与其它任务竞争，属下界） |
| `bnbrl_config_original.py` | 我改动前的 `crowd_nav/configs/config.py` 备份 |
| `bnbrl_trainprobe/`、`bnbrl_trainprobe16/` | 探针产生的 checkpoint 目录 |

## 十一、SICNav native 冒烟（已完成）

| 文件 | 内容 |
|---|---|
| `sicnav_np_smoke.log` | classic SICNav-np，5 人 circle，成功 |
| `sicnav_cvg_smoke.log` | SICNav-CVG，成功 |
| `sicnav_jmid_smoke.log` | SICNav-JMID，成功 |
| `sicnav_cvg_newenv.log` | 在新建的 `sicnav` 环境里复验，结果一致 |

## 十二、构建与依赖

| 文件 | 内容 |
|---|---|
| `build_acados026.log` | acados v0.2.6（SICNav-Diffusion 用） |
| `build_acados042.log` | acados v0.4.2（mpc_planner 用） |
| `build_gsl.log` | GSL 2.7.1 本地 prefix |
| `gen_tmpc*.log` / `gen_shmpc*.log` | 求解器生成日志（多轮，`_bounds` 是统一界限那版） |
| `build_tmpc*.log` / `build_shmpc*.log` | catkin 构建日志 |
| `isolate_env.log` | **环境隔离**：新建 `sicnav`、恢复 `crowdnav` |

---

## 我改过的仓库文件

| 路径 | 改了什么 |
|---|---|
| `bayes_occ_mpc/continuous_mpc_gate.py` | 加 `_rollout` 钩子、`robot_kinematics` 分派、`robot_heading` |
| `bayes_occ_mpc/unicycle_mpc_gate.py` | **新增**，matched-unicycle 规划器 |
| `bayes_occ_mpc/modern_worker.py` | **新增**，现代 MPC 适配器 |
| `nav_data/mamba/camrl/CrowdNav/crowd_sim/envs/crowd_sim.py` | 一处 `action.vx` 改为按运动学分派 |
| `bayes_occ_mpc/src/safe-interactive-crowdnav/simple_test.py` | 加 `--render`（默认关） |
| `.../sicnav_diffusion/utils/mpc_utils/orca_casadi_new.py` | blocksqp 的 `linsol` 改 `ldl`（无 HSL） |
| `.../sicnav/utils/mpc_utils/sicnav_acados/campc_acados_opt.py` | 编译 flag 改为环境变量可覆盖，默认不变 |
| `.../JMID/MID/mid.py` | `_add_mean` 的 CUDA 张量 bug |
| `.../BNBRLplus/crowd_nav/configs/config.py` | `pred.model_dir` 指向本机实际存在的权重 |
| `build_modern/{tmpc,shmpc}/.../settings.yaml` | 本项目协议 + 统一执行器界限；`.upstream_default` 是原件 |
| `build_modern/{tmpc,shmpc}/.../generate_jackalsimulator_solver.py` | 加 `model.set_bounds`；SH 副本切 `configuration_safe_horizon` |

用 `git diff` 看具体改动：
```
cd /home/abc/workspace/bayes_occ_mpc && git diff
cd /home/abc/workspace/nav_data/mamba/camrl/CrowdNav && git diff
cd /home/abc/workspace/bayes_occ_mpc/src/safe-interactive-crowdnav && git diff
cd /home/abc/workspace/bayes_occ_mpc/src/BNBRLplus && git diff
```

## 环境

| | 路径 |
|---|---|
| SICNav 专用（新建） | `/home/abc/miniconda3/envs/sicnav` |
| 贝叶斯项目（已恢复） | `/home/abc/miniconda3/envs/crowdnav` |
| acados v0.2.6 | `bayes_occ_mpc/results/acados_v026_sicnav` |
| acados v0.4.0（保留未删） | `bayes_occ_mpc/results/modern_solver_setup_20260906_2114/acados` |
| acados v0.4.2 | `bayes_occ_mpc/results/acados_v042_mpcplanner` |
| GSL 本地 prefix | `bayes_occ_mpc/build_modern/gsl` |
| T-MPC++ 工作空间 | `bayes_occ_mpc/build_modern/tmpc` |
| SH-MPC 工作空间 | `bayes_occ_mpc/build_modern/shmpc` |

---

# 追加索引（2026-09-09 下午）

## 最终交付

| 文件 | 内容 |
|---|---|
| **`FINAL.md`** | **最终结论**：可用的、不可用的、我更正过的错误 |
| `STATUS.md` | 全过程记录，含每一个被推翻的假设 |

## 安全—效率前沿（核心结果）

| 文件 | 内容 |
|---|---|
| `risk_sweep/front.log` | **两个方法的完整前沿 + 冻结点**，120 回合零异常 |
| `risk_sweep/front.py` / `front.sh` | 扫描脚本（各自单位，不做跨方法对齐） |
| `risk_sweep/tmpc_own*.yaml`、`shmpc_own*.yaml` | 各档位配置 |
| `risk_sweep/sweep_INVALID_risk_doubling.log` | **作废**：SH 风险越界那一轮 |
| `risk_sweep/risk_sweep_INVALID.json` | 同上 |

## 诊断（全过程可复现）

| 文件 | 内容 |
|---|---|
| `bridge/interface_audit.py` / `.log` | 十项接口验收 |
| `bridge/verify_uncertainty.py` | 边缘/增量换算的蒙特卡洛验证 |
| `diag/replay.py` | 失败步重放（可容忍并统计 bridge 崩溃） |
| `diag/tmpc_failures.txt` | 113 条原样保存的失败步 |
| `diag/scenario_constraints_instrumented.cpp` | **有害埋点存档**，不再使用 |
| `weights/sweep.log` | contouring 权重扫描（放开权重反而跑离目标） |

## 诊断用工作副本

| 目录 | 用途 |
|---|---|
| `build_modern/tmpc` | **正式** T-MPC++（Guidance + Contouring + Ellipsoid） |
| `build_modern/shmpc` | **正式** SH-MPC（场景约束） |
| `build_modern/tmpc_native` | 作者原生配置对照（N=30、dt=0.2、原生界限） |
| `build_modern/tmpc_pristine` | 干净源码对照，用于验证埋点无害 |
| `build_modern/tmpc_lmpcc` | Goal + Ellipsoid，**仅作排查工具**，不得冒充 T-MPC++ |
| `build_modern/tmpc_contour` | Contouring + Ellipsoid（无引导），排查工具 |
| `build_modern/tmpc_nodummy` | max_obstacles=5，排查哑元障碍假设 |

## 新增源码

| 文件 | 内容 |
|---|---|
| `bayes_occ_mpc/unicycle_mpc_gate.py` | matched-unicycle 规划器 |
| `bayes_occ_mpc/modern_worker.py` | 现代 MPC 适配器（含两种不确定度语义） |
| `bayes_occ_mpc/visible_history_env.py` | 受限观测 facade（供 SICNav 遮挡评测，未接入） |

## Codex接管后的运行入口（2026-09-09）

| 文件 | 用途 |
|---|---|
| `pipeline.sh` | 启动或续跑整条本地串行链；已有守护进程时拒绝重复启动 |
| `supervise.py` | 依赖串联、进程退出码、每90分钟进度更新与最终交付 |
| `snapshot/pipeline_status.json` | 当前PID、阶段、各队列进度与动态ETA |
| `logs/` | 每个阶段的真实命令与输出；不是只有完成横幅 |
| `test_takeover.py` | 断点、IPC、统计单测及最终独立计数复核 |
| `snapshot/acceptance_takeover.json` | 本轮允许比较的范围与保留限制 |
| `snapshot/frozen.json`、`snapshot/frozen_files/` | D2后生成的不可悄悄覆盖的配置与可恢复输入 |
| `takeover_20260909/original/` | 接管前源文件与登记表备份，不删除旧证据 |
| `final/` | 正式CSV/LaTeX、配对统计、耗时、失败分析与完成凭据 |

---

## 第 13 节执行令新增文件（2026-09-09）

### 单一入口

| 文件 | 职责 |
|---|---|
| `temp/modern/env.sh` | **唯一注册运行环境**。此前该段落被复制进每个启动脚本，是漂移来源 |
| `temp/modern/build_variant.sh` | 生成并编译一个预测时域变体工作副本（改 N 必须重新生成求解器） |
| `temp/modern/deploy_bridge.sh` | 把当前 bridge 源码与诊断访问器铺到全部六个工作副本并重编 |
| `temp/modern/bridge/run_bridge_cohort.py` | **唯一正式批量入口**：`plan / precheck / run / aggregate`，任务清单驱动，25 case 一块，原子落盘，可续跑 |
| `temp/latency/run_latency_audit.py` | 独占窗口计时：以单 worker 驱动上面的执行器跑 L 队列；负载不空闲时拒绝计时 |
| `temp/latency/run_latency_audit_holonomic.py` | 原 holonomic 八臂计时脚本，原样保留 |

### 冻结与登记

| 文件 | 内容 |
|---|---|
| `temp/modern/candidates.py` → `snapshot/candidates.json` | 三家族各 12 个开发候选，**第一条 D1 结果之前锁定** |
| `temp/modern/case_registry.py` → `snapshot/case_registry.json` | 按实际初始布局哈希校验的 D1/D2/T/L/debug 划分，六项交集全 0 |
| `temp/modern/freeze.py` → `snapshot/frozen.json` | T 评测前的完整冻结：臂、契约、资源、RNG 策略、统计方法、case 清单、源码与产物哈希 |
| `temp/modern/risk_sweep/front.py` | 可审计选参：D1 三点规则、D2 注册规则，落盘全部候选前沿与未入选原因 |
| `temp/modern/analysis.py` | 统计与论文表：layout 级 cluster bootstrap、Holm 校正、六份 CSV/LaTeX 导出 |

### 验收证据

| 文件 | 结论 |
|---|---|
| `snapshot/goal_audit.json` | 13.4-A 目标链路；朝向正确时两方法都能进 0.25 m；背对目标时原地不动（适用边界） |
| `snapshot/joint_disclosure.json` | 13.4-B SH 联合构造：随机游走系统性低估时间相关，最大差 0.63 |
| `snapshot/model_gap.json` | 13.3 原生模型与环境推进的一周期偏差：中位 32 mm，最大 67 mm |
| `snapshot/preflight.json` | 13.1 仓库 HEAD、产物哈希、实际加载 `.so` |
| `/tmp/claude-1000/dump_*.txt.iterate` | 13.4-C 失败步完整 primal/dual/slack；3640 条里非有限值 0 条 |


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
