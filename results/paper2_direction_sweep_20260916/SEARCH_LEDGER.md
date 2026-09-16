# 第二篇候选研究与验证台账

## 验收原则

保留贝叶斯、IL和RL总要求；允许重设计贝叶斯内部表示。已运行系统、局部预测改善、人工例子的数学差异，均不等于可投入主线。
可投入至少需要真实问题依据、合法观测可估计性、独立决策收益、强简单对照、明确近邻差异。最终投稿仍不能保证。
不以反复试到显著为停止标准。每个新实验先冻结范围，开发数据与确认数据隔离，失败与不确定结果完整记录。

## 历史核查

已重读Reviewer3、Reviewer5、编辑汇总；其要求是与既有方法区分、动作模型一致和公平消融，不是证明世界上一切非贝叶斯方法不可能。
已重读33方案总结、20260916证据库存、似然失配及相关残差实验报告、正式三臂PPO报告。
现有FULL三臂PPO完成9次训练，各20480步；不再把失败解释为只有4次更新。
历史不能支持“全部意图不可辨识”“预测更准必然没用”“Bayes信息只能是标量”等全称断言。

## 候选清单与进度

本节为开始检索时的候选状态；后续真实数据和实验结果以各追加实验为准。目前记录到实验37及各补测，尚未批准主线。

1. 旧GDBN/随机切换：历史有效负结果，暂不重跑。
2. reciprocity二值后验：现有Oracle局部决策价值低，暂不重构。
3. 意图/未知邻居混淆：缺陷有证据，简单相关修复失败；保留条件化误差模型问题，不承诺成功。
4. 关联/人数：现有接口决策机会低，真实匿名观测尚未建立，不直接重跑历史探针。
5. 各向异性传感器：现有零噪声数据不能否决新噪声，但缺真实误差数据，不把合成噪声当实证。
6. 未知执行时延：旧Gazebo故障存在，但日志缺最终命令与同步反馈；固定补偿为首要强对照。
7. 共同sensor-health：数据缺口仍在，不人为制造故障后宣布通过。
8. 群组/社会偏好：期望加性代价可压缩，且真实社会评价成本高。
9. Bayesian active DAgger：直接近邻BAgger、EnsembleDAgger、ConformalDAgger，通用查询机制不作为创新。
10. Bayesian sim-to-real：BayesSim、NPDR已有完整先例，简单后验域随机化不作为创新。
11. 有限数据下的新场景在线适配：进入真实PeRoI数据筛查；重点是后验预测校准和在线数据效率，不是宣称历史外新增信息。
12. 稀有风险/长任务风险预算：待查历史首碰诊断与现有近邻，避免重造CVaR/RFS。
13. 任务相关后验压缩/决策保持：历史Hermite工程收益不足，须有新的决策误差或样本复杂度贡献才重开。
14. belief训练接口/观测训练分布：保留可变置信度训练假说，但不把调参包装成新算法。

## 新实验1：真实轨迹的新场景在线适配

范围：PeRoI完整本地包的全部recording，合法位置历史，不使用Robot_Influence、行人最终目标或未来作为输入。
按recording名称SHA256的模5值划分：0为筛查测试，1为验证，其他训练。历史已查看过该数据，此处不是最终投稿确认。
每轨迹最多6个查询；历史1秒，预测1/2秒，查询至少间隔1秒；原始采样间隙>0.3秒排除；非有限或估计速度>4m/s排除。
固定全局岭回归预测器，训练域确定归一化。新场景只在已观测到对应未来之后更新：target_time <= query_time；不能把未成熟的标签加入适配。
对照：CV、固定全局模型、EWMA偏差、递归贝叶斯线性适配；参数仅验证集选择。记录不适配、点估计、后验预测之间的区别。
主要筛查指标：按recording等权的2秒位移误差、概率NLL、覆盖率，另列测试recording数。
该实验只决定是否值得继续决策验证。至少相对固定和EWMA有稳定预测/概率收益才继续；未通过就记录，不扩大网格硬救。

## 已核查文献入口

- BAgger (2018): https://personalrobotics.cs.washington.edu/workshops/mlmp2018/assets/docs/24_CameraReadySubmission_180928_BAgger.pdf
- EnsembleDAgger (2018): https://arxiv.org/abs/1807.08364
- Conformalized Interactive Imitation Learning: https://cmu-intentlab.github.io/conformalized-interactive-il/
- BayesSim (RSS 2019): https://www.roboticsproceedings.org/rss15/p29.pdf
- Neural Posterior Domain Randomization (CoRL 2021/PMLR 2022): https://proceedings.mlr.press/v164/muratore22a.html
- DistNav (RSS 2021): https://www.roboticsproceedings.org/rss17/p053.html
- DiPCAN (RSS 2022): https://www.roboticsproceedings.org/rss18/p045.html
- Risk-Averse Bayes-Adaptive RL (2021): https://arxiv.org/abs/2102.05762
- Online multisource pedestrian prediction (2024): https://ieeexplore.ieee.org/document/10433100/

这些是已发现的近邻，不是声称已完成所有全文审查。搜索摘要不构成全球首次证明。

## 新实验1结果：不批准投入

实际获得94773个查询，按recording隔离为训练23、验证2、测试8。查询并非独立样本；统计按recording聚合。验证recording只有2是固定哈希划分的限制，没有为结果重新划分。
测试2秒误差：CV 0.455915，固定岭模型0.421606，EWMA 0.425670，贝叶斯适配0.418682米。
相对固定模型改善仅0.002923米；NLL差-0.01511，recording bootstrap区间[-0.03819,0.00219]。
完整后验与点估计均值完全相同；后验NLL比点估计改善0.00752，但缺真实导航收益，更不能称独立新算法。
结果保存在paper2_scene_adaptation_20260916/summary.json；生产代码未改。既定筛查条件未满足，不接PPO。
额外近邻：TrajICL (NeurIPS 2025)，https://arxiv.org/abs/2506.00871；普通在线场景适配本身不新。

## 新实验2协议：旧意图似然的合法邻居记忆

先做条件动力学诊断，不训练策略、不宣称完整新Bayes成立。
现有goal_model_probe30布局；每布局最多32个等距抽取的连续可见转移，排除已知出生反点目标。
三个预测器只在目标已知的诊断条件下比较：旧D、原生ORCA+当前可见邻居、原生ORCA+4秒内合法最后观测CV记忆。
每个预测器使用相同目标，只问“合法记忆是否缓解遗漏交互导致的模型误差”；真实目标严格属于Oracle诊断，不进入任何可部署方法。
固定TTL=4秒，不调阈值。ORCA前向初始化仍是当前观测速度，不使用私有响应记忆。
若记忆比仅可见邻居没有稳定降低预测残差，不开展该版本后验重构。
若有改善，仍需再验证未知目标后验和决策收益，不能把Oracle残差改进当成成功。

## 新实验2结果与未知目标扩展

条件诊断共910个连续转移：开发12布局366条，历史留出17布局544条。
留出单步位置残差：旧D 0.007579米、原生ORCA仅可见0.007916米、加合法邻居记忆0.005550米。
记忆减仅可见=-0.002366米，布局bootstrap95%区间[-0.003498,-0.001285]。合法记忆能部分缓解模型遗漏，但绝对残差仍非零。

随后实际运行未知目标递归后验，不再给予真目标；原生ORCA+记忆未来模型对所有后验臂保持一致。只比较似然用可见邻居还是合法记忆。
每布局最多4个合法未知目标冲突查询，64/128嵌套目标候选；4档噪声尺度由开发Energy Score选定。
历史留出15布局50查询：

| 候选数 | 可见FULL点误差 | 记忆FULL点误差 | 记忆MAP点误差 | CV |
| --- | --- | --- | --- | --- |
| 64 | .452730 | .429026 | .444048 | .523865 |
| 128 | .461962 | .403089 | .414751 | .523865 |

128候选时记忆FULL较可见FULL的点误差改善约12.7%，区间排除零；Energy Score改善区间仍跨零。
记忆FULL减记忆MAP=-.011662米，区间[-.054705,+.028315]，未证明超过MAP。
64候选时Energy Score反而变差，显示候选近似仍影响结论。没有追加粒子直到显著。
判决：通过“遗漏邻居确有部分影响”的诊断；未通过完整后验的研究投入门。不启动PPO，不把12.7%报成主线成功。
记录：paper2_neighbor_memory_20260916/cases.json及posterior/summary.json。

## 新发现的直接近邻

GOFI已经把目标与遮挡因素联合推断并接入MCTS，不能再把“未知邻居造成目标归因错误”当未被提出的问题：
Hanna et al., Interpretable Goal Recognition in the Presence of Occluded Factors for Autonomous Vehicles, IROS 2021, https://arxiv.org/abs/2108.02530。

Bordallo et al.的交互反事实意图推断已发表于IROS 2015，arXiv上传2016不是首次发表日期。其使用HRVO并讨论遮挡跟踪：
https://www.research.ed.ac.uk/en/publications/counterfactual-reasoning-about-intent-for-interactive-navigation-/
https://arxiv.org/html/1610.08424v1

这两篇不证明当前问题已完全解决，但要求任何改进说明连续不确定邻居、推断精度、实时性或决策学习的具体技术差异；不能仅替换成ORCA+PPO。

值导向belief压缩同样有长期先例，不能把“只保留决策相关后验信息”本身作为原创：
https://arxiv.org/abs/1508.00986

## 当前状态

搜索任务仍在进行，尚无满足最终标准的方向。没有远程长训、没有删除任何历史成果。
已运行三个数值阶段：真实场景适配、邻居记忆条件动力学、未知目标后验。
下一步优先检查真实预测残差的结构与任务风险，而不是再增加无语义模式或重新长训同一GDBN。

## 新实验2最后一个预定扩展：邻居不确定性边际化

在点记忆条件误差改善的基础上，固定8个反对称高斯样本，过程加速度标准差沿用旧RFS的0.55m/s²，按遮挡时长传播位置/速度联合协方差。
每步对未知邻居状态求似然平均后再更新目标后验；不以真实隐藏位置生成样本。保持相同目标候选、查询和未来预测函数，仍用开发集选4档尺度。
这是逐步边际化近似，不是精确联合滤波；未知邻居状态还没有被目标行人的运动证据反向更新，跨帧相关也未完整传播。
冻结比较：边际FULL对点记忆FULL的点误差及Energy Score至少改善2%且布局区间上界<0，同时FULL对自身MAP有正证据。64/128近似敏感性仍须报告。
若不满足，停止这组邻居修复扩展，不接PPO，不额外增大采样数直到过门。
脚本marginal/protocol.json继承前阶段通用memory gate文字；本段在边际化结果产生前明确新增臂判定，不以结果修改。

边际化结果：128候选点误差.409443，对照点记忆.403089；Energy Score .353288，对照.347132。64候选改善但置信区间跨零；两个候选规模均没有FULL对MAP的可靠优势。按门停止此扩展。
自检通过：重跑的visible/memory全部汇总均值与前阶段逐位相同；没有隐藏邻居时采样上下文恒等；反对称样本均值保持点记忆。

## 新实验3：真实轨迹误差的方向结构与有限数据协方差

复用新实验1完全相同的PeRoI查询、分割、固定岭预测均值；仅检验2秒预测误差分布，不把它称为传感器测量噪声。
比较固定圆、固定完整协方差、固定Student-t、在线EWMA完整协方差、逆Wishart贝叶斯协方差后验预测、相同贝叶斯协方差的Gaussian点近似。
误差在合法过去速度方向的坐标系表示。贝叶斯只学习协方差，均值固定为原全局预测器；先验等效样本数5/20/100，由既有验证recording选择。
更新继续只在一个人的首个2秒未来已观测完后进行，每人最多一条证据；所有方法共享证据，禁止提前使用未来。主指标recording等权NLL、90%椭圆覆盖与面积；另预先报告证据数<=20的新场景冷启动。
只有相对EWMA和固定Student-t均有稳定概率收益，才做导航风险验证；仅比圆好不能批准Bayes。该共轭模型是标准工具，不把公式本身当创新。

## 扩展检索与资源限制

延迟belief与RL已有直接近邻：https://arxiv.org/abs/2509.20869 、https://arxiv.org/abs/2506.00131 。结合缺同步命令日志，暂不启动合成延迟长训。
Bayesian expert干预已有HiRIL：https://doi.org/10.1016/j.eswa.2026.131118 ；简单不确定性触发DAgger不能作为新主张。
JRDB群组数据官方要求登录：https://jrdb.erc.monash.edu.au/ 。本机未找到群组原始标注，不绕过访问控制、不假装已取得数据。
现有resid2_clean/severe只有残差和协方差、没有当时人速度/机器人朝向，无法严谨恢复定向margin所需坐标。不能据此判定方向信息不可用。

新实验3结果：IW后验NLL .903450，EWMA .917100，固定完整Gaussian .917429，固定Student-t .725518；IW对EWMA有小幅改善，但未超过固定重尾分布。固定圆/椭圆也相近。冷启动子集结论不变，停止此标准协方差适配版本，不进入PPO。

## 新实验4：真实激光检测误差的联合性

来源DROWv2公开MIT数据与官方T=1 DROW3预训练模型：https://github.com/VisualComputingInstitute/DROW 。下载原始激光扫描、人员位置标注、里程计；不是给CrowdNav添加人造dropout。
先做最便宜的实测诊断：在官方val/test的每第5个已标注帧运行冻结检测器，置信度阈值仅val选择，匹配半径固定0.5m。记录漏检、位置误差、同帧成对漏检及激光统计；不训练新检测器。
先核查在距离/方位条件化后是否仍存在跨人的误差相关。如果没有明显剩余联合结构，不开发shared-health滤波器。
真实人位置只用于离线匹配、诊断条件化和评价，不能当部署时的输入。即使发现相关，也不能宣称它一定来自共同传感器故障；空间遮挡、标注方式、距离混合均可能解释它。
官方test仅5个独立序列，统计报告这一限制；重复帧不能冒充独立场景。该阶段不是闭环导航、不是已通过研究投入门。

DROW结果：val选择检测阈值0.4；test486帧、388个人次，209检中、179漏检。
距离/方位/局部扫描条件化后有部分同帧残差相关，但5序列的成对样本数为52、31、0、2、81；只有2对的序列不能支撑推断。
修正后的对称成对相关系数写在audit/geometry.json。summary.json中原normalized covariance是协方差除预测方差的尺度比，不是有界相关系数，不采用它作相关结论。
已匹配位置误差径向/切向RMSE为0.1034/0.0776米，但受0.5米匹配与检测阈值截断，不是完整测量噪声。
判决：只支持继续独立数据复核，尚未证明共同sensor-health、更未证明导航价值。

## 新实验4独立复核：FROG

使用官方逐帧激光人员标注及官方FROG预训练DROW3；不训练检测器。
官方DataLoader明确720束、180度、角度endpoint=False；使用此坐标而非旧DROW坐标。
11-36/12-43两整段录像用于阈值与条件模型校准，10-31/14-57/15-53/16-41四整段录像测试；每段按时间至多1Hz抽帧，不按相邻帧随机分割。
沿用0.5m匹配、阈值0.1~0.9仅校准选取、线性和非线性条件模型诊断同帧剩余误差相关。模型仅诊断，GT几何不允许进入部署状态。
若联合结构复现，下一关仍须合法原始扫描/检测历史能估计，并超过独立模型和简单经验校准；只出现相关不批准主线。
数据：https://robotics.upo.es/datasets/frog/laser2d_people/
源码：https://github.com/robotics-upo/2DLaserPeopleBenchmark
已存在近邻：LRFS综述明确遮挡会使检测概率依赖整个多目标状态；不能把此结构本身声称为新发现。https://www.ba-ngu.vo-au.com/vo/VVNS_TSP24.pdf

FROG复核结果：测试6021帧、13426人次、4418漏检；阈值0.6由校准选择。
四段每段5023~5997成对样本。非线性条件化后对称成对相关为-.0147、-.0290、-.0301、-.0092；线性模型为-.0082、.0064、-.0013、.0300。
没有复现足够强的正相关共同故障证据，不启动shared-health Bayes/PPO。检出位置误差径向/切向RMSE .0412/.0532米，有轻微方向性但尺度小且匹配截断，不能据此宣布导航需求。
这一否决限定当前检测器、数据及筛查；不声称真实传感器永远不会共同失效。
全部产物：paper2_drow_20260916/frog_audit/。已完成所有下载和推理进程。

## 新实验5：真实持续群组关系的Oracle筛查

获得DIAMOR两天真实位置/速度/身体朝向与人工群组标注（research-only，不重新分发）：https://dil.atr.jp/sets/groups/ 。
不使用ATC未标注区域作负类，不把未标注组误当独行。DIAMOR仅一位标注者、同场所两天是明确限制。
按原始时间1Hz选完整帧，近距离2.5m内行人对，每对最多一个查询；合法历史3秒，目标为2秒后的两人相对位移。
三臂预测器容量/超参相同：Current物理状态、History四帧物理状态、Oracle当前状态+真实同组标签。真实组标签只进Oracle。
同时比较Current/History能否从合法特征识别关系。第一天训练、第二天整天测试，不能随机混相邻帧。统计以十分钟块聚合，不能将它当跨场所泛化。
若Oracle对History相对未来误差没有至少5%改进且块区间排除零，不继续以“群组关系改善碰撞预测”为主张；若关系识别有效而动力学增量弱，只记录社会代价问题尚未验证，不用人为reward直接制造成功。
近邻已包括Group Estimation for Social Robot Navigation (2022)、Group-based Motion Prediction (2021)和Multi-Agent Dynamic Relational Reasoning (2024)，并非发明group-aware RL。

群组结果：第一天17608对、第二天53226对；真实同组分别1113、1098对。Current/History的2秒相对未来误差.71164/.57663米，Current+Oracle .71201米。
自检发现Oracle仅加在Current上不足以评价History之外的增量，因此补充History+Oracle：.57673米；减History的十分钟块均值+.000175，区间[-.000187,+.000540]。
关系能识别（History AUC .982），但本探针中真实群组对几何历史之外没有预测增量。不批准“群组Bayes改善碰撞预测”；社会违规代价仍未实测，不能直接用人为组奖励宣布通过。

## 新实验2补充自检：ORCA内部平滑状态

重新核对原生ORCA发现其_last_pref_vel是控制器内部低通状态，不等于实际避碰速度；前面的native单步调用每帧用实际速度初始化，仍是近似似然。
下一步只检查把每个候选目标的内部preferred-velocity按合法观测递归推进，是否改变似然和未知目标后验结果。没有读取真实私有控制状态；失观后重置为观测速度并明确近似。
保留原64/128候选、原查询、开发选尺度、相同未来预测器对照MAP/FULL。此为实现/模型一致性检查，不将模拟器特有TTC平滑本身作为新论文问题。

内部状态递推结果：128候选FULL点误差.42988、MAP .45118；差-.02130，区间[-.06776,+.02559]。64候选FULL反而比MAP差.01061米。没有稳定Full优势，也未优于前面的点记忆近似，停止这组修复扩展。

## 其他候选的近邻复核

共享场景流场：Bayesian Floor Field已经用几何深先验+人流观测贝叶斯更新提高数据效率，Fast Online CLiFF-map已做在线变化检测、概率流场和导航。因此“把单人意图改成共享人流地图”本身不足以作为新贡献。
https://github.com/aalto-intelligent-robotics/bayesianfloorfield
https://arxiv.org/abs/2410.12237

感知/计算预算调度：属于成熟active sensing/RL范式，不能单凭调整感知频率称新Bayes。已有自适应前向时域导航RL：https://arxiv.org/abs/2108.06161 ；未找到本项目独立实测预算瓶颈，不先制造昂贵模块再证明调度省算力。

延迟方向补充正式发表记录：Model-Based RL under Random Observation Delays已发表于L4DC2026，非仅检索到未发表预印本。https://proceedings.mlr.press/v331/karamzade26a.html
Bayes/预测必要性诊断同样不能声称全新：Perfect Prediction or Plenty of Proposals?在自动驾驶IPP中已测完美预测没有自动改善规划。https://arxiv.org/abs/2510.15505

## 新实验6：朝向线索与修复后后验的决策补查

DIAMOR朝向消融已完成。相对运动History不含/含朝向误差.576660/.576629米，区间跨零。
自检认为相对运动会抵消共同转弯，于是追加同批查询的个体位移目标，不改数据划分：四帧History不含/含朝向误差.379158/.382669米；未测出增量。
这只否决本数据的测量朝向和当前预测器，不把它解释为所有视觉gaze无效。

修复后128目标后验补做2秒局部动作诊断，共15历史留出布局50查询。原项目holonomic投影动作集合对所有方法相同；真实人不响应机器人，按实际记录的人轨迹做逐段扫掠碰撞。
代价固定为终点距离+10倍碰撞概率；该诊断不是原RL return、不是unicycle闭环成功率。
Moment保留FULL整个16维时序位置协方差，256反向配对Gaussian样本；Scalar仅开发布局选择余量。
FULL/Moment局部代价6.3858/6.3730、布局等权碰撞率均.2278；ScalarCV(0.2m)代价5.8303、碰撞率.1556。
FULL未赢Moment或标量；不进入RL。OracleAll代价4.2268，说明信息/模型缺口仍存在，但合法后验未兑现。
产物：paper2_neighbor_memory_20260916/decision/；paper2_groups_20260916/facing/和individual_facing/。

## 新实验7：真实停留的剩余时长

不预测随机事件的开始，改问已观察到停留后的剩余时间。使用PeRoI同一recording隔离划分。
仅合法观测确认的停留：0.25秒重采样，过去0.5秒速度<=0.15m/s，先前确实运动；速度>=0.25连续两帧作为离开事件。轨迹结束/观测间断为右删失，不能当离开。
每轨迹只取第一段已观察到起点的停留；查询停留已持续1/2/4/8秒。
固定比较memoryless exponential、两/三指数混合的Bayes类型后验、MAP、Weibull、Kaplan-Meier型经验生存对照。
分布参数只训练录制拟合，K只验证选择；先测可评价未来的概率误差并明确删失限制。完整混合不能可靠超过Weibull和经验生存模型，就不开发wait/detour导航。
此为新候选的筛查，不声称semi-Markov或生存分析本身有新颖性。

结果：PeRoI仅24个符合条件的停留，测试仅2个，信息不足，不能作方向否决。随后在DIAMOR复核：训练377、验证68、测试624个停留；测试3444个可评价查询。第一天训练/验证时间隔离，第二天测试。
验证选择三分量混合。测试NLL：Exponential .60040，FULL .56592，Weibull .57112，经验生存 .57024，MAP .59347。
FULL减Weibull=-.00520，十分钟块bootstrap区间[-.01188,+.00182]，没有可靠优势；不进入导航开发。
自检限制：summary中的recordings实际为十分钟时间块，不是独立录制；经验生存实际用Nelson-Aalen指数变换而非严格KM；不可判定的删失查询被排除，概率指标可能受选择影响。没有将轨迹截断错误标为离开。
近邻OSCAR已研究临时障碍的等待/绕行，使用在线删失生存估计与真实机器人。不能把该组合声称新颖：https://arxiv.org/abs/2606.00990

## 已有PeRoI风险结果复核

重新直接读取switching_validation/nonlinear/summary.json，原6011查询6录制：FULL/Tree的接近事件Brier为.00310555/.00308255，FULL没有更好；FULL/IID预测误差差异仅.000362米。
移动机器人跨录制测试只有一个录制71条轨迹，FULL误差.57814、CV .54581；不可用总体17%改善推广到该子集。
NLL为连续位置密度分数，不是事件NLL；Brier是若干离散时刻到记录机器人CV位置的距离事件，不是完整扫掠碰撞/导航成功率。
这支持不直接将已有非线性模式库投入导航，但没有替代新的真实轨迹闭环反事实实验；后者本轮尚未完成。

## 真实检测数据关联的可验证性检查

DROW逐帧.wp标注没有连续身份；FROG circles六列为中心/半径及极坐标表示，没有身份ID。不能将最近邻重连得到的伪身份当真实Oracle。现有1Hz检测审计也不足以替代连续跟踪序列。
因此真实检测条件下的关联决策价值仍为未验证，而非已经失败。不得把原完美位置速度接口的负结果推广为现实跟踪不需要联合假设。

受限追加测试已完成：只用密集位置标注中双向唯一的0.3m连接生成片段，dt>0.1秒或歧义即截断；不称人工身份真值。三次连续缓存检测与未来片段都可对应的两人子集，816校准、704测试查询。似然尺度仅两校准录制选择，测试MAP与片段对应100%一致，模糊查询0，FULL/MAP预测相同。
该筛选主动排除了难连接、漏检、身份中断等情况，因此只能说明连续清晰可匹配子集无增量，不能否决真正遮挡后关联。产物paper2_drow_20260916/association_screen/。

## 新补充近邻

未知时变动作延迟/丢包也已有Adaptive Reinforcement Learning for Unobservable Random Delays (ACDA)，不只是已知常延迟被解决：https://arxiv.org/abs/2506.14411
动作相关的belief压缩与决策质量保证也并非新问题：Value-Directed Compression of POMDPs, NIPS2002：https://papers.neurips.cc/paper_files/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d-Abstract.html
机器人施加给人的干扰/社会成本不能简单当新的Bayes问题：Planning-based Prediction for Pedestrians已有预测与hindrance折衷：https://www.cs.cmu.edu/~bziebart/publications/planning-based-prediction-pedestrians.pdf ；Bayesian Inference for Human-Robot Coordination已有未知人的代价参数推断：https://shraybansal.com/assets/papers/bansal21bayesian.pdf 。后两项仅初步检索，未在本轮实现对应新方法，不把检索摘要当全面复现。

## 新实验10：用Bayesian Q后验限制IL策略的错误改进

不再预测行人类型，转问动作价值的不确定性能否筛掉TD3的错误动作改进。仅离线探针，没有修改生产Actor，也不是新的RL结果。
使用历史3000条frozen transitions中62条完整回合2956步；其余截断尾段不用于Monte-Carlo拟合。复现存档C4 Critic，min(Q1,Q2)与存档输出误差通过2e-6校验。
对固定Q1末层256维特征做Gaussian线性后验修正。10整布局校准，40布局80状态测试，使用已存在的真实反事实return。动作优势方差包含候选动作与baseline的协方差。
比较原Q1、后验均值、固定误差界、校准posterior误差界、冻结IL。训练数据为带探索行为的MC return，而反事实继续策略是确定Actor，二者有策略差异；历史测试数据已被多次分析，不能当新的最终确认。
原Q1相对IL平均收益-.01678；后验均值-.00362；固定界-.000362；校准posterior+.000000242，只改1/80个状态，没有一个达到.01实际收益。
posterior与固定界差+.000363，布局bootstrap区间[-.0000207,+.000967]，不支持优势。校准倍率7.93，说明原后验严重过窄；报告的覆盖率包含baseline零差项和重复动作，不能解释为严格安全认证。
结论：后验通过近乎不改动作保住IL，没有得到可投入的RL收益。产物paper2_q_posterior_20260916/{protocol.json,summary.json,per_state.csv}。
相关近邻：SPIBB https://proceedings.mlr.press/v97/laroche19a.html ，Bayesian Policy Gradient/Actor Critic https://jmlr.csail.mit.edu/papers/v17/10-245.html 。泛称“用Bayes保证策略改进”不是新贡献。

## 新实验11：NavWareSet真实交互数据筛查

新取得作者公开的541个手选轨迹CSV，仓库HEAD 3c10aa8b43dfc8588fafc465b62ee21a45c03d6a。项目：https://anr-navware.github.io/navwareset/ ；数据：https://github.com/anr-navware/NavWareSet-Quant 。
两类CSV有440个单人和101个多人文件。时间戳纳秒，位置米；原始数据未修改。只使用两种格式都有的焦点行人和机器人位置，不能假称观察到完整人群。
固定1秒历史、2秒未来、1秒查询间隔、最大插值间隔.25秒。参与者组1训练，其中四个blind-corner场景留验证；参与者组2测试。4309训练、544验证、4815测试查询，16/4/19场景。没有重复timestamp冲突。组2仍只有一个参与者组，场景bootstrap不能当独立人群总体置信区间。
同18维输入槽、相同HistGradientBoosting容量：Current、History、OracleRobotFuture，不可用特征置零。三随机种子固定模型，没有调参；实际此树模型在该设置可能接近确定，不能称三个独立训练复现。
场景等权2秒误差：CV .56470，Current .43635，History .43720，Oracle机器人未来 .42811米。
History-Current +.000849，场景bootstrap[-.003295,+.005431]；Oracle-History -.009086，区间[-.015497,-.002801]。
结论：真实数据可用，但尚无Bayes信息或导航收益证据。Oracle小幅预测收益不能当动作因果效应：机器人未来是内生的、操作者也响应人，社会/非社会驾驶顺序不是随机实验。不得直接把记录机器人未来作为在线输入，不进入PPO。
产物：paper2_navware_probe_20260916/{protocol.json,summary.json,queries.csv,test_errors.csv,features.npz}；脚本同名.py。

### NavWareSet同步五人联合风险子检验

只取8个有同步五人标注的circular场景。3个组1场景训练共享行人均值预测器，另1个组1场景校准残差；组2四场景测试。380/162/562查询，最多2秒未来。
在相同40维Gaussian残差下，只删除跨人协方差块，保留每个人整个时间序列的协方差、所有均值和几何。LedoitWolf校准缩减系数.12413；跨人块并非零，最大相关系数.227。
对记录机器人未来四个时刻的近距离事件评分，1024反向配对样本。不是实际换动作后的反事实，不是连续扫掠碰撞，不是RL。
0.6m事件47/562，Joint/Independent Brier=.102383/.102438；1.0m事件241/562，.055247/.055233。差异极小，可能在有限采样误差内。未见值得继续开发的联合风险增量。
该结果仅限制这一Gaussian近似和四个真实录制，不否决任意依赖模型。只下载轨迹数据，未改生产代码。产物paper2_navware_probe_20260916/joint/。
近邻全文确认：de Groot等2025 IJRR已处理跨障碍/跨时刻联合碰撞风险，规划隔离实验假定行人预测分布已知。不能将“联合风险”四字作为新颖性：https://autonomousrobots.nl/assets/files/publications/25-degroot-ijrr.pdf 。

## 新实验13：机器人执行端真实数据

取得作者公开的9段ROS bag，实测总123.35分钟，包含/robot/cmd_vel、/robot/pose、battery、bumper等。来源：https://www.robot.t.u-tokyo.ac.jp/~miyagusuku/datasets/ ，原始包wifi_rosbags.zip，443MiB，解压约1.26GB，全部写/home而非系统盘。
首段确认pose为odom/base_link、ROSARIA，约10Hz；指令约33Hz；里程计不是独立外部真值。需要核对真实响应与发布/记录时间，不能直接声称观察到随机执行延迟。
近邻ACDA全文确认：支持未知随机延迟和丢包，观察包包含历史执行buffer及延迟，使用状态分布embedding；不是仅处理固定延迟。仍留有未观测执行队列/无回执等细分差别，但本轮没有据此声明可投入：https://arxiv.org/html/2506.14411v1 。

三段building3训练，三段building14校准，三段较晚日期building24测试。预测下一次约0.1秒后的报告速度，指令全部取当前里程计收到时已经收到的内容。所有bumper记录均为false，没有由此得到碰撞/导航标签。
CurrentVelocity、CurrentCommand、ARX当前指令、ARX六个指令历史tap的测试角速度RMSE=.04448/.03004/.02272/.01884 rad/s，说明简单命令历史已有作用。
再用六个有效延迟假设(0..0.5s)、相同线性模型和Gaussian likelihood进行Bayesian递推，严格先预测再用下一次观测更新。identity transition下等似然保留70/30赔率的单测通过；切换/噪声系数仅验证选择。
FULL/MAP/ARX History角速度RMSE=.02439/.02459/.01884；线速度.006168/.006142/.005515 m/s。FULL没有赢强简单预测器。
FULL NLL=-6.207，优于未校准Gaussian ARX -5.830，但补充固定Student-t强对照后为-6.351(更好)。该尾分布对照是在初测之后追加，仅用验证选择df=3和scale=1，属于探索性复核，不假称原预注册。
因此当前延迟模型后验不批准进入导航。限制：有效延迟混合了控制器惯性；轮式里程计非外部真值；没有实际执行指令回执；相邻观察之间仍可能出现新的操作者指令，不能把全部预测误差当隐藏动力学变化。未证明物理随机延迟本身是否显著存在，也不否决所有执行端Bayes。
产物：paper2_robot_dynamics_20260916/{tables/manifest.json,audit/summary.json,posterior/summary.json,posterior_tail_control/summary.json}，源文件逐包SHA256保留。

## 新实验14：补齐冻结PPO的密度泛化（已完成，未通过）

复核正式报告第143行明确未测heldout、10/20人；不应把5人负结果直接推广到密度泛化。保持生产HEAD16577ff，train_smoke SHA与正式protocol完全相同，全部GDBN参数SHA匹配。
只加载最终20480步的9个PPO模型，不重训、不选择checkpoint。tensor-only推理避免加载跨Python版本的pickle。生产SetEncoder/ActionHistory原AST，另外函数式实现核对；复现10个历史nominal/nonstationary回合，布局/步数/结果/回报全部一致，回报差0。
冻结六原场景、heldout_nonstationary、每格100新布局、3RL seed×3arm，共5400回合。primary为五个高人数场景等权；FULL须同时赢两个对照至少3pp、碰撞不增、至少2/3seed回报改善且配对区间下界>0。不根据结果改人群参数。
三个RL seed共用一个IL初始化；即使本轮阳性，也需独立IL种子/确认数据和技术差异审计，不能直接宣布论文成立。输出paper2_ppo_density_20260916/。

完成54格、5400回合、600个不同布局；跨臂和种子layout SHA完全一致，零越界、无未终止。高人数五场景SR：No-Belief31.533%、MAP31.400%、FULL31.000%；CR：68.2%、68.2%、68.8%。FULL-No SR差-0.533pp，95%配对bootstrap[-2.867,+1.800]pp；FULL-MAP差-0.400pp，[-2.667,+1.667]pp。回报差分别-0.006691（CI[-0.040656,+0.027390]）、-0.004825（CI[-0.037665,+0.026055]）。10000次重采样训练种子与每场景布局，布局索引在臂/种子间共用。FULL-No分种子回报差[+0.009357,-0.000685,-0.028746]，FULL-MAP[+0.006090,-0.021434,+0.000869]。

5人heldout SR为87/89/89.333%；两个20人场景各臂SR只有5.333%-7%。故既没有FULL优势，也不能继续假设旧96.4%的5人底座已经具备密度泛化。未动主训练代码，未启动新RL。完整六场景表加入中文报告；逐回合JSON、预复现、冻结protocol和完成收据保留。

## 专家查询方向的近邻筛查

Bayesian不确定性指导DAgger/query预算已有EnsembleDAgger(2018)、BAgger(2018)、ConformalDAgger(ICLR2025)。因此不把“省教师调用”本身包装成新方向；本轮未另起相同算法训练。
https://arxiv.org/abs/1807.08364
https://personalrobotics.cs.washington.edu/workshops/mlmp2018/assets/docs/24_CameraReadySubmission_180928_BAgger.pdf
https://proceedings.iclr.cc/paper_files/paper/2025/hash/19deda616aebc9bee6c1990baa5f1f0e-Abstract-Conference.html

## 新实验15：初始重叠错误及隔离修复后的密度复测

对第14项600布局做初始几何审计，机器人重叠0；行人间重叠各场景28/41/78/88/83/69，共387。源crowd_sim.py242/244行列表推导式赋值使每个生成器看到空self.humans，跳过与同批已生成行人的净空检查。上游vita-epfl/CrowdNav第89-95行逐个append；本地根提交2546bae已含错误。不能把此错误描述为Bayesian算法错误或推翻真实数据负结果。

在既有诊断脚本追加--spawn-audit，只在每个评估进程绑定逐个append生成器。生产文件SHA保持0b92fc89ac5900a605a9af9f48de8a0976bb2a0f765b5d58adb33182cc216327，git tracked worktree clean。诊断script SHA6080dfa97ae5a8ec44d84ecef209626c5c0597447ce0f60d15fd92ff75415340。此前旧协议的script SHA保留，未覆盖旧结果；新增结果独立sequential_spawn子目录。

再次5400回合、600唯一布局，跨三臂/三seed初始SHA完全匹配，全部正常终止、零越界。最小机器人/人间净空0.201983/0.201296m，全部超过配置0.2m。相同case随机种子不代表修复前后同物理布局，不能作那种因果配对。本轮仍为冻结旧训练checkpoint的外部评估，未重新训练修复后环境。

高人数等权SR No33.333/MAP30.600/FULL32.200%；CR66.467/69.133/67.467%。FULL-No SR差-1.133pp CI[-3.400,+1.067]；FULL-MAP+1.600pp CI[-0.402,+3.733]。FULL-No回报差-0.016682 CI[-0.047788,+0.014170]，三个seed均负；FULL-MAP+0.022124 CI[-0.007390,+0.052166]，三个seed均正。10000次配对seed/场景内layout bootstrap，固定场景等权，复用跨seed布局索引。没有满足同时优于两个对照的预设门槛，不批准投入。

这不是第二篇方向层面的好消息。第14项作为旧实现结果保留并加此限制；20人泛化崩溃在排除初始重叠后仍存在。完整场景表在中文报告，aggregate.json含均值/区间/分seed结果。
上游来源：https://raw.githubusercontent.com/vita-epfl/CrowdNav/master/crowd_sim/envs/crowd_sim.py

## 新实验16：连续参数后验，而非离散延迟模式库

复用9包ROS数据及原b3/b14/b24划分，单一14维指令历史ARX为基模型，在线15维(含截距)线性残差系数；两个输出按训练残差标准差缩放。先预测后更新，gap>0.2s重置。验证网格prior variance[.001,.01,.1]、forgetting[1,.999,.99]，选.1/.999；EWMA验证选.01。共轭递推对200样本批量闭式解断言通过。方差/df仅验证选择，训练/测试记录不参与校准。

真实记录等权测试RMSE(v,w)：fixed(.00551475,.01884283)，EWMA(.00553947,.01892762)，parameter(.00518561,.01855168)。均值由RLS完全可复现，不宣称Bayes专有。点估计Gaussian NLL-6.014698；含参数方差Gaussian-6.154646。但点估计固定Student-t(df3,varscale1)-6.567337更好。

为避免似然族混淆，另加同族方差控制，输出独立continuous_parameters_family_control，不覆盖第一轮。Student-t+同一参数方差NLL-6.579628，较点估计同族改善0.012291。此项只说明小幅概率分数增量，不能判定零价值，也不能批准导航主线；Student-t输出是方差控制，不是精确Student-t系数后验。无闭环反事实/外部位姿真值/实际执行回执，三个测试记录同日且已被历史审查。没有扩PPO或改生产架构。

新增明确近邻：McKinnon & Schoellig, Learn Fast, Forget Slow, RA-L4(2):2180–2187,2019, DOI10.1109/LRA.2019.2901638。已经wBLR未知执行器动力学+快速适应+长期学习+Tube MPC，900kg机器人、物理和人工动力学变化、立体视觉定位。故通用在线参数后验接导航不是新颖点；本轮不批准投入。
https://www.dynsyslab.org/wp-content/papercite-data/pdf/mckinnon-ral19.pdf
https://www.dynsyslab.org/icra-2019/
产物：paper2_robot_dynamics_20260916/continuous_parameters{,_family_control}/{protocol.json,summary.json}。
最新诊断SHA e763ce639edc56a9d53c77833e2c24f3f2ce1aa606056c3f350b54dfd75d1472。历史结果保留其当时源SHA；仅追加诊断函数，没有篡改旧模型结果。

## 新实验17：真实原始tracklet及自然失检后的关联结构复核

下载官方frog-raw-circles.zip及六份odom.npz；zip SHA256 9a0c1de82537362205693255a0f0ff846fe96a5f4327601d9b4d0e1e877a7b33。34份CSV含1,020,483人标注。原始idp只在标注块内有效，不是跨遮挡永久身份。旧9片段关联筛查不覆盖此总体，不能沿用其“无歧义”作为真实数据总判决。

先发现CSV scan_seq与H5 row不能直接对应，CSV timestamp也不能直接替代H5 timestamp。上游export_dataset.py明确strip_empty过滤11-36/12-43/16-41；仅恢复过滤索引仍不能对齐全部记录。因此不用手调时间偏移，改为整帧(x,y,r) float32精确且双向唯一匹配，获取H5时间。重复整帧剔除并统计，不猜身份。1014123标注成功对应，最大位置数值差5.29e-7m；六记录无非重复几何缺失，重复帧1497/0/4/1088/433/1825。重新按校正时间分片，共4950片、3237片>=2s。初版raw_tracklets/中的world_xy和分片时长不可信，不用于实验；有效版本是aligned_tracklets/。odom只是轮式里程计，不是外部真值，插值只用于离线审计；时间源为bag记录时间。

随后固定官方DROW检测阈值.6、GT匹配<.5m、当前检测人对距离<=2.5m、距机器人<=6m。两条历史检测相距<=1.25s；比较两个全局分配假设、MAP、独立边缘混合、保留完整跨人协方差的Gaussian Moment。先做连续检测，再补自然失检最长5s的控制，不人为dropout。历史身份连接使用审计真值，因此只叫乐观机制筛查，不是合法在线tracker，更不称整个问题的数学上界。

11-36/12-43校准，四条官方测试记录评估。两秒未来插值必须gap<=.1s；按全部合法当前条件先选，再记录未来缺失，不隐藏删样。连续版本947测试query，关联歧义0。自然失检版1064测试query，其中117涉及失检间隔>1.25s、8个posterior在(.1,.9)。cal选择关联过程sigma=.25、预测sigma=1m；非零实测歧义存在，但不足以产生有量级的联合风险差异。

四测试记录等权joint future NLL：MAP4.124748，FULL4.124947，IID4.124660，Moment4.124374。FULL未赢。记录机器人2s后位置的1m近接事件共29个，Brier MAP.03101202/FULL.03101051/IID.03100997。FULL与IID预测概率平均绝对差1.80e-6。Moment近接概率用512固定Gaussian样本估计，Brier.031619不能拿来宣称Full赢Moment：其差异可受积分误差影响，解析NLL反而Moment更好。rank-one Gaussian NLL公式对scipy多元正态100次自检，最大误差1.34e-12。

这是固定记录轨迹近接评分，不是真实候选动作闭环，不是碰撞率；没有真实未来机器人控制反事实。尚有标签块边界、可观测未来删失和oracle历史关联等限制，不可扩大为“所有真实关联Bayes无用”。本轮不批准投入，不接PPO。
结果：paper2_drow_20260916/frog_audit/{aligned_tracklets,annotated_association,annotated_association_gaps}/summary.json；原始query/protocol/frame_map及数据SHA均保留。当前probe.py SHA42f4a0857d0f8440a96d437d005e8710b05f896d24f8fbba39ac700ae79790b4。
来源：https://robotics.upo.es/datasets/frog/laser2d_people/
https://raw.githubusercontent.com/robotics-upo/2DLaserPeopleBenchmark/master/export_dataset.py
https://raw.githubusercontent.com/robotics-upo/2DLaserPeopleBenchmark/master/convert_frog_circles.py
https://raw.githubusercontent.com/robotics-upo/2DLaserPeopleBenchmark/master/convert_frog_bags.py

17项追加时间自检：H5约20.2%的相邻记录间隔<1ms，中位约38ms，属于bag记录时间的明显非均匀采样；原生逐帧差分会给出不可信的瞬时速度。本试验使用约1s检测间隔和2s端点，没有用原生逐帧差分当速度真值，但仍有时间戳抖动与短轨迹删失限制。后续若做高频跟踪/控制，必须进一步取得sensor header时间，不可把这些原生差分值当作真实加速度或人类异常行为。

## 新候选18：Mini Wheelbot真实硬件上下文的预测Oracle（已完成，未通过）

当前没有批准新主线。为补执行侧缺乏外部真值的限制，核查2026 Mini Wheelbot数据：1kHz机器人日志、指令力矩、动捕、硬件/地面/失败元数据。固定Zenodo v2 18260659，不使用来源不明的加工副本。原始zip3.88GB；整包下载速度低，主动停止后改用fsspec+zipfile HTTP Range，仅获取元数据和预选CSV。383个元数据已获取，CRC由zipfile校验，下载落盘原子替换。没有运行机器人、没有发控制指令。

先冻结全部30条velocity记录；三台硬件各按uuid排序6训练/2验证/2测试，不按成败挑选。全为black_pvc，不能说覆盖地面泛化。比较同容量Current/History/History+Oracle hardware，目标0.1s相对Vicon旋转，所有臂获得同一预定力矩序列作为动作条件输入。不把未来传感器状态给模型。三seed、最多3000更新、验证选点；需要Oracle相对History至少5%误差改善且整轨迹配对区间排除零，才考虑继续合法belief推断。仍非导航价值门。

初读CSV发现Vicon位置数值单位需要另核，故当前只用单位明确的四元数，不将位置字段直接当米。公开示例的滤波filtfilt与绘图全段yaw对齐不进入本实验在线特征。来源及脚本版本：wheelbot/dataset abca22d44c9074cbd16fa4c52e3a5742dbc4c9c9。

最近邻限制已查：ARCADE(2512.14331)已做变化点感知Bayesian动力学+闭环；RSS2025(2504.16923)已做Kalman式在线越野动力学适配；EVORA(2311.06234)已做牵引分布与风险规划；Mini Wheelbot作者已有AMPC Bayesian optimization调参(2512.14350)。因此不能把“Bayes+新机器人数据”当作新颖性。本轮只是寻找问题证据，不复制这些方案后改名。
https://zenodo.org/records/18260659
https://arxiv.org/html/2601.11394v1
https://github.com/wheelbot/dataset
https://arxiv.org/abs/2512.14331
https://arxiv.org/abs/2504.16923
https://arxiv.org/abs/2311.06234
https://arxiv.org/abs/2512.14350

18项结果：30条记录完整CRC/SHA检查，36159个查询，18/6/6整轨迹划分全部保留；3seed模型各最多3000更新，验证选点。测试记录等权0.1s相对旋转向量RMSE(rad)：Current .03086200，History .03142058，Oracle hardware .03218519。Oracle相对History为-2.43%改善（即更差），配对seed/整轨迹bootstrap的History-Oracle区间[-.00290229,+.00084995]，未通过预设5%且区间排零门槛。保存权重独立CPU重算九组预测，最大差1.79e-7；输入/目标全部finite。没有接Bayes/PPO。

结论只限这一小样本、已知三硬件、同地面、0.1s、给定记录力矩的预测筛查；不是硬件变化普遍无价值，更不是动作因果控制实验。硬件编号可混入动捕外参/传感器偏差，未来力矩是给定控制条件，不代表在线能预知闭环控制。此项不批准第二篇投入。

## 新实验19：修正出生几何后的高密度未来信息价值

复用现有density诊断脚本，不修改生产；80新布局、四密度场景各20，No-Belief2407正式PPO冻结；初始生成逐个append。第一次t=3..131当前Actor动作保持2s的CV扫掠净空<.5m时触发，不看未来/结局，78合格、53在第3步触发。每状态10开环候选(已有加速度投影)+原Actor反馈，共858完整分支，重置前缀状态hash断言与前8步人位置oracle逐步相等均通过，80布局hash唯一。robot.visible=False，不含动作诱发行人反应。

Current/History/Oracle使用完全相同UnicycleCEM代价、margin.5，仅未来输入不同；执行候选2s后交回冻结Actor直到终止。成功15/11/19，碰撞63/67/59；Actor21成功57碰撞；事后有限候选Best49成功28碰撞1超时。Oracle-Current场景等权return+.048274，整布局分层配对bootstrap CI[-.013464,+.114921]，少4碰撞；History-.046926 CI[-.102349,+.001780]、多4碰撞。均未过预注册return>=.02且CI排零且无碰撞增加门槛。

此为固定决策代价下信息替换，不是全局Bayes价值上界。它保留局部正例而没有聚合确认，不能说高密度完全没有未来信息价值。不给这份负/未确定结果添新的PPO。已使用heldout作为研究诊断，未来论文必须另留新测试。
产物paper2_ppo_density_20260916/dense_forecast/{protocol,summary,各case}.json。

## 追加近邻核查

BRVO已有EnKF+速度障碍参数推断：https://gamma-web.iacs.umd.edu/BRVO/BRVO.pdf
Auto-Encoding Bayesian Inverse Games已有多模态后验vsMAP/MLE规划：https://arxiv.org/abs/2402.08902
SITH/IJRR2025已有不损失动作质量的自适应belief简化：https://arxiv.org/abs/2310.10274
2026已扩展有保证的open-loop简化/跳过重规划：https://arxiv.org/abs/2604.01352
因此没有把旧负结果简单改名“自适应belief压缩”当新颖点。

## 新实验20：真实感知误差的时间持续性及合法可推断性

不再只看同帧跨人相关，而检查同一原始tracklet的约1s残差相关。FROG四测试记录6507对，检测阈值.6、GT匹配<.5m，误差在检测自身径向/切向基底，未用世界odom消除误差。训练两记录的7叶100步回归去掉range/bearing/score/density条件均值后，四记录相关分别(.371,.336)、(.387,.404)、(.220,.351)、(.178,.235)，过“至少3记录>=200对且相关>=.2”现象门。标注本身误差、误差截断、oracle历史连接和bag时间仍是限制，不能单凭相关性说滤波必须改或宣布导航贡献。

随后检测误差>0.1m为标签，11-36训练、12-43验证/校准、其余四记录测试。先用5合法几何/score特征及2帧历史，同容量树、Current/History/二状态HMM/EWMA。HMM相对Current Brier改善仅0.0337%，没过5%实际门槛，未优于History。没有把这个弱特征模型的失败等同于传感器信息不可辨识。

补充原始激光输入：以检测位置为中心，21个横向[-.5,.5]m对应beam范围差，裁剪+/-1m；全为合法扫描，不用GT定位取patch。采用所有score>=.6检测的Hungarian CV关联，里程计严格as-of，速度门限2m/s*dt+.3，断档>1.25s重置；GT仅最后匹配/打当前误差标签，误检保留在合法历史而不冒充定位误差。26当前维、两个历史及mask，Current/History同15叶200步模型，val Platt校准，HMM/EWMA只val选参数。

Current测试AUROC .793/.763/.754/.818，证明raw scan确有误差辨识信息；但HMM验证选择persistence=0，即等于Current。四记录等权Brier约Current/HMM .076432、History .077353、EWMA .076731。未通过递归增量门，不接PPO。

为避免把错误iid emission归咎Bayes，额外做条件似然比：odds P(z_t|current,past2)/odds P(z_t|past2)，新context模型同容量、只训练/验证；递推以此incremental evidence更新。无信息ratio=1时保持stationary prior的数值断言通过。条件独立与Markov是假设，不称精确真实后验。该探索性补充仍val选persistence=0，Brier .078714，相对Current差2.985%，四记录配对CI也不支持改善。没有继续增加HMM复杂度。

结果：paper2_drow_20260916/frog_audit/{temporal_error,reliability_information,reliability_scan,conditional_reliability}/summary.json，合法输入及全部预测npz保留。该轮只判定这些具体模型未有增量，非所有时间相关滤波无效。
近邻：时间相关感知错误仿真已在IJCAI2020讨论：https://www.ijcai.org/Proceedings/2020/0483.pdf；不能把“采用Markov噪声仿真”直接作为新贡献。

## 新候选21：同一硬件的真实地面接触差异（进行中）

第18项只检验相同地面不同硬件，不能否决地面差异。现有元数据明确提供同一hardware2的yaw控制、black_pvc/concrete/gray_felt各10次。固定全部30条、不按成败选，UUID顺序每地面6train/2val/2test；单独wheelbot_surface目录，不覆盖velocity结果。现正获取CRC校验后的CSV。
预测0.5s相对Vicon旋转，Current/History/Oracle surface同容量与训练预算；历史.1/.25/.5s，所有臂相同给定力矩序列；这是接触条件Oracle，不是Bayesian控制或反事实返回。已有EVORA、ARCADE、在线Kalman动力学近邻不变，不能靠新地面标签宣称新颖。

全轮方法学提醒：这些连续候选筛查属于探索性研究，名义95%区间没有为“不断寻找阳性”提供总体错误率控制。任何候选即便过一个门，也必须锁方案后用未参与筛选的新数据/布局确认；不能将边看结果边迭代的测试集继续称最终独立测试。

### 实验18的未来控制混淆复核

记录中的未来力矩来自反馈控制器，不是事先独立随机指定的动作；因而原比较可能通过未来控制间接获知未来响应。追加同预算、同划分、同输入维度但未来力矩列统一置零的对照，只保留当前指令与传感器。
Current / History / Oracle硬件RMSE分别0.034699 / 0.034585 / 0.035260 rad；Oracle仍差1.951%，History-Oracle配对区间[-0.001657,+0.000022]，未通过。CPU安全加载九组权重复算，最大预测差1.79e-7。
结果：paper2_robot_dynamics_20260916/wheelbot/current_control_only/oracle_summary.json。这是观测策略下的预测，不是任意动作的因果动力学识别。第21项也运行同一去未来指令对照。

### 实验21完成第一关，继续合法推断检验

30条均CRC校验取回；yaw46仅3.326秒，力矩启用后不足2秒历史，按原协议0查询，不替换。实际17train/6val/6test。
给未来控制：Current/History/Oracle为0.196418/0.174800/0.162244rad；Oracle改善7.18%，但CI[-0.001604,+0.034003]跨零，未过门，且受一个History训练seed欠拟合影响。
不给未来控制：0.204973/0.202470/0.190907rad，Oracle改善5.71%，CI[0.005930,0.017425]，通过预测Oracle筛查。没有使用未来Vicon或未来指令作为输入。
这仅证明同一硬件的地面上下文在本0.5s闭环观测预测中有增量，不是反事实动作价值，也未证明合法Bayes可取出收益。继续第22项：冻结已有预测器，仅使用当前/过去机载传感器和过去动作做地面识别、条件似然比递推，比较MAP/FULL/短历史；不启动导航训练。

## 实验22：合法地面推断与后验压缩，未通过

沿用21的探索性划分，预测器固定为不含未来指令的三seed集成，因此绝对RMSE不与21的逐seed均值混比。类型分类器Current/History/Context同15叶200步HistGB，训练按记录/类别平衡，验证集选择概率温度。条件Bayes因子P(z|current,past)/P(z|past)，每五查询更新，验证选power；明确这是近似充分历史，不宣称精确真实后验。
自检修正：第一可用查询必须用已有history更新均匀先验，不能仅积累新增因子而丢失初始化证据。初版漏初始化的结果作废；修正后power=.1，零新增信息赔率不变断言通过。
测试六记录等权RMSE：History0.195004、Oracle0.183436、MAP0.186009、FULL0.185396、直接History分类概率混合0.185061、均匀混合0.187734。FULL相对MAP仅约0.33%，区间[-0.000036,+0.001824]含零，未通过。类型准确率Current78.6%、History80.2%、递推68.2%；说明合法信息存在，但该近似递推没有胜过直接历史模型。
相同验证残差协方差下，FULL混合NLL-2.494570，Moment-2.494115，仅差0.000455，不足以主张分布结构不可替代。没有反事实控制/导航验收，地面识别与Bayesian控制本身已有近邻，未批准论文主线。
结果：paper2_robot_dynamics_20260916/wheelbot_surface/legal_surface_belief/summary.json。代码SHA568e5b1c1b31803a5c2f89489d86d80591bca1d39d26ba62dcacc247b8e4cc57。

## 实验23：真实轮式执行的联合误差结构，未通过

取得DRIVE六份官方DataFrame，使用严格允许列表反序列化，转为无object的NPZ并记录SHA。三机器人、雪/瓷砖/碎石/冰；指令、编码器和原始ICP端点可用。原文已经有BLR滑移，不能把该组合当新颖点。来源：https://norlab.ulaval.ca/research/drive_dataset/ ，https://arxiv.org/abs/2309.10718 ，代码固定14680108287052c7a3a1720a627af1f08435a092。
先用同一非线性均值模型比较完整/对角协方差，另放同协方差Student5以控制尾厚，而不是让Gaussian充当弱对照。每记录按calib_step前50%训练、随后25%校准、最后25%测试，两处留2个控制段间隔。原表部分控制段2或4行，不强凑3行。只输入首时刻指令、首编码器、已知段内序号；不输入未来编码器、smoothed ICP velocity或派生slip。
固定cov时Student full比diag平均差0.01968nat，只有1/6记录略好。补转向条件cov，排除左右转相关性合并抵消：仍差0.01217nat，CI[-0.03822,+0.01305]，3/6略好。
进一步查到部分两秒窗口内指令切换，不能把全部端点误差叫打滑。最后仅保留双轮命令全40步恒定的窗口（不看轨迹结果），仍整段划分、同样方向分组和门槛。Student full平均差0.005346nat，CI[-0.02394,+0.01075]，2/6略好，未过门。
结论仅限这套端点误差结构模型，不能否定所有动力学Bayes；没有导航实验，也没有为过门增大学生训练。结果：paper2_robot_dynamics_20260916/drive/{joint_response,joint_response_directional,joint_response_directional_constant}/summary.json。

## 追加低成本近邻排除

“隐藏行人到达率的Gamma-Poisson后验+机器人探索”已有针对部分可观測计数、相关传感器及传感器模型不确定性的完整研究： https://link.springer.com/article/10.1007/s10514-022-10070-9 。因此未将这一通用统计组合直接列为新方向，也没有制造合成到达过程来宣称Bayes获胜。

## 实验24：Moment比较的信息含义复核

对已知三个非共线二维分支，归一化条件加均值的线性系统已经满秩，可精确恢复三概率，数值复算最大误差1.91e-17。因此不能泛称三个分支posterior必定比均值含更多信息；前提是分支支持本身已知，不能将未知支持偷加给Moment。
同时复核实际原生ORCA的八个固定早期状态、每个128个合法生成目标分支、16维未来轨迹。153维原始一二阶矩特征的相对1e-9数值秩为51–85，矩阵严重病态；并不是能稳定还原全部128概率。数值秩不是精确代数秩证明。
原决策代码中的Moment是把分布替换成256个联合高斯样本，不是“存矩后恢复原离散分布”。所以该对照检验的是高斯投影近似，不应扩张成所有低维表示均无法表达full后验。
另给出四点[-2,-1,1,2]的严格等矩对照：概率[.125,.5,0,.375]与[.375,0,.5,.125]同均值0、二阶矩2.5，正半轴概率.375与.625。它只证明信息结构可能不同，不证明真实观测可推断或导航可获益。
结果：paper2_neighbor_memory_20260916/moment_identifiability.json。这个修正用于筛选协议，不作为可投入方向。

## 新角度近邻复核：未直接开新训练

机器人多峰全局定位是真实且不同于行人模式的问题，但通用提案已经有直接近邻：M3P(2015)为被搬动/位置多解的机器人规划消歧动作 https://arxiv.org/abs/1506.01780 ；PLANS2020已有Bayesian位置belief与RL导航 https://www.ion.org/publications/abstract.cfm?articleID=17402 ；2026 BeliefDiffusion用多模态配置与规划处理感知混叠 https://arxiv.org/abs/2606.18888 。本地找到Gazebo world文件，但这不提供上述通用组合的创新性，因此未新搭一个迷宫宣布Bayes获胜。

另考虑利用belief价值函数的凸性改善学习。此思路也有直接近邻Convex Is Back(2025) https://arxiv.org/abs/2502.09298 ，以及UAI2026 Neural Value Iteration https://proceedings.mlr.press/v337/you26a.html 。特别注意：最优POMDP价值函数的凸性不能不加条件强加给任意固定belief-policy的on-policy PPO critic。没有为了制造结构创新给生产PPO增加ICNN。
以上是通用构想的近邻排除，不是证明这些领域没有任何研究空间。

## 实验25：真实狭窄交互Bi3探索性检查

HRI2026原文 https://arxiv.org/abs/2601.09856 的80人研究支持合作假设可能失效，但不支持复杂预测自动改善导航。关联Bi3公开数据为74人10.5小时，人数口径不能混用。入口 https://fluentrobotics.com/bi3dataset/ ，官方代码固定36eca4fb0774a440efad5640867574513e0984ad。
全量Dropbox为38.8GB且尝试的Range请求没有返回206；已停止全量下载。从部分ZIP完整恢复UM1-9的CV条件JSON，验证长度、CRC32、JSON解析。状态第三维是朝向，不是速度。官方最近邻下采样可能选未来帧，诊断改为严格as-of，并仅以过去位置差分估速。
跨场探索UM1-5训练、UM6选树容量、UM7-9检查，所有数据仍属于官方训练域，不是独立正式验收。两秒位置误差Current0.58068、History0.55384、Current+heading0.56723m；近机器人1m子集0.57759/0.56895/0.56216m。总体历史增量小，近距离更小。
再冻结History模型，检验持续加性个体偏差。每2.2秒用已完成的2秒预测误差更新，避免未来泄漏；固定/EWMA/共轭高斯Bayes误差0.55384/0.55393/0.55525m，NLL1.35723/1.34570/1.34452。没有强增量。这只约束常量残差偏差模型，不约束所有交互动力学。
代码paper2_bi3_probe.py，结果paper2_bi3_data/{extraction_audit,raw_audit,exploratory_information,exploratory_adaptation}.json。没有反事实导航实验，不批准方向。

进一步只用当前几何和过去速度设CV风险触发：2秒内最小中心距<0.6m，最近时刻>0.1秒。留出三个场次5794个人时刻中369个触发，固定模型Current/History/heading误差0.56616/0.56775/0.53803m。历史没有集中收益，heading三个场次均略好。此是观测朝向信息，不是Bayesian posterior收益；中心距触发也不是实际碰撞标签。未按风险子集调模型，未证明反事实导航价值。

自检修正：上述旧heading把robot heading与human heading一起加入，归因不干净。重跑已给所有臂共享机器人朝向、当前goal与turning flag，heading只加两个人朝向。当前全体误差0.56864/0.54475/0.55037m，风险子集0.56704/0.56785/0.53431m。原结果存*_before_proprio.json，当前JSON为修正结果。在线偏差适应亦重跑：固定/EWMA/Bayes误差0.54475/0.54363/0.54555m，NLL1.32858/1.31214/1.31389，Bayes仍未优于简单对照。没有将朝向增量当成Bayes贡献。

## 实验26：DRIVE时间相关误差与强物理对照

不同于实验23的终点跨坐标协方差，这次预测0.20至1.95秒八个横向位置，同样整控制段划分与恒定指令筛选。Student对角NLL7.59096、完整时间cov3.79291，六记录均改善。此初级gate确实通过，但只是排除按时刻对角近似。
同均值同边缘尺度的固定布朗相关NLL3.24889、积分白噪声相关2.94642，反而更好。物理公式已按真实20Hz采样时间修正重跑。没有证明必须学习复杂cov，没有Bayesian在线更新或导航收益，因此不批准为第二篇。
结果paper2_robot_dynamics_20260916/drive/temporal_response/summary.json。gate_passed字段仅表示初级结构筛查，不能解释为项目成功。

## 近邻27-28：训练不确定性与决策压缩

贝叶斯主动DAgger标注已有DropoutDAgger https://arxiv.org/abs/1709.06166 、BAgger https://personalrobotics.cs.washington.edu/workshops/mlmp2018/assets/docs/24_CameraReadySubmission_180928_BAgger.pdf 。强对照包括RND-DAgger https://arxiv.org/abs/2411.01894 与ConformalDAgger https://cmu-intentlab.github.io/conformalized-interactive-il/ 。当前没有更具体的技术差异，未启动重复训练。
Bayesian policy-gradient不确定性已有JMLR2016 https://www.jmlr.org/papers/volume17/10-245/10-245.pdf ，不是2026重新上传arXiv才提出。AC-Teach已有Bayesian critic选教师 https://www.robotics.stanford.edu/blog/acteach/ 。
Value-Directed Compression在NeurIPS2002已提出 https://papers.neurips.cc/paper_files/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d-Abstract.html ，Value-Directed Sampling也直接研究面向POMDP策略执行的粒子滤波 https://arxiv.org/abs/1301.2305 。所以“保动作质量、减少粒子”本身不够新。2026 evidential belief-function compression https://arxiv.org/abs/2608.10650 与Bayesian概率不是同一个形式体系，只作为邻近思想，不混为同一算法。
以上排除的是通用包装，不是宣称领域已无创新空间。总目标仍未达到，无获批方向。

## 实验29：SocNavData2026评分与个体偏好可识别性审计

官方 https://github.com/ljmanso/SocNavData2026-Trep382 固定33d2acf80cc8480c7a174f138106b108579c49f2。论文 https://arxiv.org/abs/2509.01251 为人工评价真实/仿真混合轨迹，不是全真实机器人实验。ratings包已下载并CRC校验，含all49人、selected22人。71个JSON实例中38个需容忍字符串内控制字符，保留原文件；不使用人口统计字段推断偏好。
all6481次评分，仅35个非控制(trajectory,context)有跨人重叠，selected4402评分仅19个。重复评分MAE0.15695/0.13917，跨人非控制MAE0.21346/0.21769，后者不能全部归因于持续偏好。
34人完整公共控制题探索：其他人均值预测MSE0.05456，用本人前5题校准常量偏移再预测后10题反而0.06434。未证明低维偏好可推断，也不证明所有偏好模型不可行。当前不足以批准个人偏好Bayesian导航主线，未造新reward或训练PPO。
数据与结果paper2_socnav_ratings_data/{extraction_audit,overlap_audit,control_bias_audit}.json。

后续结构化Gaussian随机效应测试补足“常量偏移太弱”的限制：其他33人估计15个公共评分项协方差与重复评分噪声，目标人前5题更新后验。后10题MSE0.054565→0.051478，但Spearman0.410204→0.410099，未改善排序。NLL每item -0.11134→-0.13033。仅新评分者/已知控制题探索，不是新轨迹或导航验证。
若导航仅优化线性偏好下的期望回报，Gaussian后验均值就是充分决策统计，MAP与均值相同；不能把完整posterior说成必需。脚本paper2_socnav_ratings_probe.py，结果structured_preference.json。方向未批准。

## 实验30：真实激光测量误差方向性

FROG两个recording校准、四个recording测试，保留旧的合法特征条件均值，测径向/切向残差。Student5同尾厚下，圆/视线定向椭圆/full covariance记录等权NLL为-4.311985/-4.366728/-4.366773。四条测试记录均支持方向性，但full对椭圆仅0.0000458nat。校准两个轴方差0.0007488/0.001812。
这补充的是实际传感器方向证据，不是旧的零噪声遮挡快照。仍有检测匹配截断与漏检未建模限制，也没有证明时序后验或导航可被静态margin替代。结果paper2_drow_20260916/frog_audit/directional_noise/summary.json；未批准新方向。

## 实验31：合法关联下的真实检测位置滤波

补测此前未直接测过的状态估计，而非可靠性分类。所有方法共用检测位置+因果CV关联，Hungarian门限2*dt+.3，无GT identity。里程计as-of映射同一坐标，GT仅用于匹配评分，原始检测匹配门限.5m。缓存1Hz。

12-43开发选参：EWMA alpha候选.5/.7/.85/.95/1；CV-KF白加速度std .03/.1/.3/1/3与测量std .03/.06/.1/.2，共20组；KF Joseph更新，初始化位置方差.01、速度方差1。选中alpha1与KF(3,.03)。没有根据测试成绩重调。

四条记录10-31/14-57/15-53/16-41，匹配数量2329/2114/2131/2433。等权MAE原始0.032396403、EWMA同原始、KF0.032396659米。未检出实际改善，不批准方向。

限制：只有匹配检测上的位置误差，没有漏检评估或闭环；里程计当固定量；普通CV-KF不是所有Bayes模型上界。缓存GT匹配口径与aligned_tracklet实验不同，样本数不能直接混用。测试记录已反复用于探索，不是新独立确认。结果`paper2_drow_20260916/frog_audit/state_filter/summary.json`；复现`probe.py --state-filter`。没有生产改动或RL训练。

## 实验32：同源真实检测的因果未来预测

另按forecast MAE选参，避免实验31当前位置目标偏好直接检测而漏掉预测收益。至少两帧合法关联历史，预测old_position+dt*old_velocity；dt在.75至1.25秒。当前帧进入滤波前计算误差，当前检测/GT不进入这一预测。原始CV采用最近检测差分速度；EWMA平滑速度，位置仍取当前检测；KF为原CV线性滤波。12-43开发选EWMA alpha=.8、KF q=.1/r=.03。

四记录预测n1856/1496/1599/1630，合计6581。等权MAE原始CV .3560362195，KF .3561227812，速度EWMA .3471687312米。EWMA四记录均好于两者。未提供Bayes相对简单方法收益，不启动导航/PPO。

评分条件依赖下一帧合法关联，存在身份切换/选择偏差，漏检期间不评价；不能声称GT身份条件下的纯预测误差。四记录重复探索、无独立新确认。脚本`probe.py --state-forecast`，结果`frog_audit/state_forecast/summary.json`。共享样本数与finite值断言已加入；生产git diff为空。

## 实验33：把Bayes移到IL至RL部署验收的近邻与数值筛查

问题真实：历史策略在RL后退化。但不能将简单Bayesian deployment gate当成新方法。已核查一手来源：
- PSyCo，2020预印本/2021Science of Computer Programming：Beta-Bernoulli模型检查约束满足概率，既指导学习又验证策略。https://arxiv.org/abs/2005.03898 ，https://doi.org/10.1016/j.scico.2021.102620 。不是仅泛泛安全RL。
- SPIBB，ICML2019：高不确定状态下回退baseline，研究安全策略改进。https://proceedings.mlr.press/v97/laroche19a/laroche19a.pdf 。不等于本文特定连续PPO实现，但排除泛化的“保持IL底座”创新措辞。
- DeepSPI，ICLR2026作者页面：online world-model/representation下的局部策略更新与改进分析。https://delgrange.me/publication/delgrange-2025-deepspisafepolicy/ 。本轮只核作者摘要，不将其所有理论条件视为已查证。

固定策略、固定样本数、独立同分布碰撞Bernoulli的最简验收数值：均匀prior的Bayes单侧95%上分位 vs Clopper-Pearson单侧95%上界。
0/100碰撞：.02922515 vs .02951305；1/100：.04610735 vs .04655981；2/100：.06102210 vs .06161920；0/200：.01479362 vs .01486704；1/200：.02338265 vs .02349847；0/500：.00596166 vs .00597355。
若要求上界<=.02且零碰撞，分别需148/149回合。复现公式scipy.stats.beta.ppf(.95,c+1,n-c+1)与beta.ppf(.95,c+1,n-c)，本轮实际执行。该差异不足以支持样本效率论文。Bayes credible与频率覆盖语义不同，不可混称同一保证；反复看结果停止、多checkpoint选择、policy变化或分布漂移都不满足这里的固定实验假设。

结论仅排除“加一个Beta验收门”作为目前可投入新主线，并不排除有额外结构与验证的安全策略改进研究。没有新网络、没有重新训练、没有批准方向。

## 实验34：真实Bi3上的递归非线性专家选择

复用`paper2_bi3_probe.py --model-bank`。UM1-5每记录一个二维HistGB专家(15leaves,150iterations)，统一基线全五记录训练同规格预测器。输入合法History及共有robot proprio；目标2秒位移。每专家似然方差由其他四训练记录残差估计，不使用自身拟合残差。UM6独立记录开发，UM7-9探索测试，官方仍属于train域。

更新仅使用target_time<=now且已完成的预测证据，间隔2.2s；先将旧权重与均匀prior按rate混合，再累加power*loglikelihood。开发rate=[0,.05,.2,.5,1]，power=[.1,.3,1]，MAP/FULL各自同等网格选NLL。没有读取未来机器人动作。训练记录专家不解释成真实人类类型。

测试记录/人等权2秒误差：统一.544754599，均匀.587258588，MAP.599448903，FULL.562420659米。均匀/MAP/FULL NLL1.583397665/1.517744826/1.485923298。FULL开发选择rate.05,power1。FULL相对MAP有收益，但统一预测器仍更好；整个库树数为统一5倍，不能声称容量匹配。缺统一等价校准NLL，所以只对点误差做统一比较，不将结果扩成概率密度全排序。

未通过实际投入标准，不接PPO。存在一手近邻Parallel Interacting Multiple Model-Based Human Motion Prediction for Motion Planning of Companion Robots：https://doi.org/10.1109/TASE.2016.2623599 ，摘要已经明确预测后接非线性MPC；不能把多模型Bayes预测接规划本身作为新颖点。其他标准滤波比较：https://dare.uva.nl/id/b7f307af-19a2-412e-aecf-20dcbb32f273 。本轮不是正式全面prior-art完成。

输出`paper2_bi3_data/exploratory_model_bank.json`。过程exit0，无生产改动或远程任务。

### 34补充：不能仅凭点误差否定分布价值

实际追加`--bank-density`，统一模型与所有专家均在UM6用零均值残差MSE校准对角方差，floor .01；测试不重拟合。保留原始34文件另存`exploratory_model_bank_density.json`。开发同时选原rate/power网格和统一Student自由度3/5/10/30(选30)。这是一套探索校准协议，不是独立正式验收。

NLL(记录/人等权)：统一Gaussian1.251452523，统一Student1.239598581，均匀混合1.572307681，MAP1.503064354，FULL1.432364416。FULL的精确一二阶矩Gaussian替代1.442041147。矩协方差包括分量内方差与分量均值间协方差，不只取对角。FULL混合相对其Moment有.009676731nat增量，但仍显著高于统一模型的记录均值，不声称统计显著性。FULL点误差.574810009，统一.544754599。

因此此前遗漏的概率评分强对照已补齐，不再仅以点误差决定投入。当前模型库没有通过这个新增检查；结论限于这五个记录专家及当前合法历史，不推及所有Bayes预测器。执行exit0；生产git diff为空。

## 实验35：真实重复评分上的主动查询及选择

34名完整15控制题评分者，全都仅前5题有重复评分。第一次答案为可查询反馈，第二次为保留验收。每人留出，其他33人的LedoitWolf协方差减重复噪声后PSD投影构建15维高斯prior，噪声仅由其他人估计。最多3个不重复查询，最后从5个方案选后验均值最大者。

对照population、100随机顺序均值、最大后验方差、相关高斯Knowledge Gradient、对角KG。KG用32点Gauss-Hermite积分估计查询后最大候选后验均值的期望，不是新算法。选择过程不读取第二次评分或人口属性。

平均第二次评分：population.840146716；random.838151742；variance.850661768；KG.851005596；diagKG.786283143。KG-pop差.010858880，34用户配对bootstrap10000次探索CI[-.008663351,.036612815]；variance-pop差.010515052，CI[-.011184756,.046803370]。KG未稳定超越population，且基本追平variance。hindsight最大评分.934441455含噪声选择偏差，不当真实偏好oracle。

仅限已知5控制方案与离线历史问答，没有新轨迹或实际导航。重复探索不是确认性CI；diag较差不证明所有非Bayes方法无效。当前不批准。

一手近邻：相关高斯KG(2009) https://pubsonline.informs.org/doi/abs/10.1287/ijoc.1080.0314 ；贝叶斯主动偏好适配RL导航(2020) https://www.europe.naverlabs.com/wp-content/uploads/2020/02/Fast-Adaptation-of-Deep-Reinforcement-Learning-Based-Navigation-Skills-to-Human-Preference.pdf ；FAPL(2022) https://arxiv.org/abs/2201.00469 。

复现`paper2_socnav_ratings_probe.py --active-choice`，结果`paper2_socnav_ratings_data/active_choice.json`，exit0，无生产改动或PPO训练。

## 数据扩展可获取性复核（不计否决实验）

Bi3官网https://fluentrobotics.com/bi3dataset/ 的Data链接再次访问。curl -L --max-time30 --max-filesize3000000，dl=0仍转向大文件，exit63被大小保护终止；未获得单独测试JSON，没有活动下载。此前9个UM CV记录仍然仅是官方train域，不改称独立确认。

发现ACME：https://arxiv.org/abs/2607.21964 ，v1 2026-07-24；https://arxiv.org/html/2607.21964v1 第11节明确目前审稿私有分享、计划HF公开。官方实验室2026-09新闻不等于数据已经开放。查询HF /api/datasets?search=ACME及Bi3&limit=100，未找到对应官方社会导航数据集；不以名称搜索穷尽性宣称所有访问路径关闭。

原文含跨机器人站点数据、机器人语音交互，可作为后续数据来源线索。但尚未获得数据、未核时序与反事实可识别条件，不能作为贝叶斯主线通过证据。本轮只是明确新确认数据缺口，不将数据访问失败当科学否决；无新训练、无生产修改。

## 实验36：两行人预测误差的联合结构及在线更新

Bi3每session对齐两人的query时间，使用同一History四输出HistGB(15leaves/150iter)预测两人2秒世界系位移。UM1-5训练，UM6校准均值偏差与LedoitWolf4x4cov，UM7-9测试。各arm均值相同，各人2D边缘相同，block只删除跨人cov。校准跨人相关最大.287288。

记录等权NLL：Gaussian full2.457312872/block2.433181222；Student5 full2.412024674/block-shared-scale2.395351732/两个独立2D Student2.395589119。不能把block Student称为完全独立，它仍有共同尺度。

为排除只因固定校准失效，追加递归Inverse-Wishart已知均值协方差后验及EWMA。IW初始E[Sigma]=校准cov，Psi=(nu-5)*cov，预测Student df=nu-3,shape=Psi/df；仅target_time<=now的误差outer-product进入更新，每2.2s一次。EWMA同样观测时序。UM6选择IW nu0=6/12/24/64及EWMA alpha=.01/.05/.1/.2，均选择最保守64/.01。

IW full2.377224647/block2.365351053；EWMA full2.365015325/block2.358612409。适应可改善评分，但跨人完整cov仍未赢block，IW未胜EWMA。这里没有证明导航价值，没有候选动作因果模型，不启动PPO。只有3个已反复探索测试session，非确认性统计；不扩大成联合分布无用。

代码`paper2_bi3_probe.py --joint-error`；结果`paper2_bi3_data/joint_error.json`，exit0。没有新增生产结构。

## Bi3输入语义审计：朝向不是速度，也不等于视觉朝向

重新核官方HMPNav，git ls-remote master=8b23db2e8d151e55fe04e60946837d99fde83438。在线node.py约468-478行读取human TF quaternion并算yaw；offline callback约366-368行则覆盖为速度方向，两条路径必须区分。原论文https://arxiv.org/html/2605.06863v1 III-D/V-A描述OptiTrack与记录方式。

本地9份CV数据逐记录以因果as-of .2秒差分计算速度，取speed>.2m/s；yaw与运动方向绝对圆差均值(度)26.77969/23.18169/26.12689/26.88065/22.25892/24.46980/23.81845/24.58867/22.27250，样本18314/12021/20761/17496/17461/16065/17430/20666/16516。支持其不是直接复制差分方向，但不能单凭此称身体/头部/gaze方向；稳妥语义为动捕标记yaw。实际机器人从视觉估计的准确性没有验证。

本地JSON仅8keys：robot_state,agent_states,predictions,robot_prediction,robot_goal,turning,time,logits。没有动作命令，尽管论文概述说记录了命令。不能把该概述补进实际数据，也不能将差分位移或机器人预测轨迹冒充已执行控制。此前预测试验仍是logged-policy条件，不是动作反事实。

源码固定链接：https://github.com/fluentrobotics/HMPNav/blob/8b23db2e8d151e55fe04e60946837d99fde83438/node.py 。本次为字段契约审计，不计新的方向否决或投入批准。

### 实验36补测：联合端点占据风险

执行`paper2_bi3_probe.py --joint-occupancy`，结果`paper2_bi3_data/joint_occupancy.json`已产生并读回。当前robot位置+2秒当前速度定义参考点，0.6m邻域内任一人的真实2秒端点为label。FULL/block共享预测均值、各人2D边缘及4096个Sobol Student5样本，差别仅跨人协方差。

UM7/8/9样本922/1099/876，正例44/67/26。Brier FULL/block分别.067357618/.067510976、.085803056/.085798732、.069178642/.069151725。记录均值.074113105/.074153811，差-0.000040706，仅一份记录改善。概率平均绝对差.001392/.001271/.001421。没有投入级证据，不能把微小Monte Carlo差距当成确定收益。

自检：人1为坐标原点、人2位置取current索引2:4、robot位置4:6、速度10:12；两人真实位移各加对应原点。Student shape=cov*3/5保持df5协方差。block仍有共同尺度依赖。此为端点占据，不是扫掠碰撞；参考点不是反事实机器人动作，真实人类未来属于原记录控制器。无生产代码改动，无PPO，无新方向批准。

### 实验36补测：真正按人独立Student

原block Student保留共同尺度，故新增independent arm：四个标准Normal仍由共同Sobol点产生，两个人分别除以独立chi2(df5)尺度。乘block covariance Cholesky；各自2D边缘不变。Sobol由5维改6维，原两臂数值因此也有Monte Carlo变化。原文件保留，新结果`paper2_bi3_data/joint_occupancy_independent.json`，命令同`--joint-occupancy`，exit0。

FULL/block/independent记录平均Brier .0741199389/.0742093202/.0740973463。独立对照在UM8和UM9略优于FULL，在UM7略差；总体几乎相同。此处没有支持共同尺度结构的投入级证据。依然只是静态校准密度下端点占据预测，不是贝叶斯在线学习或导航决策；未启动RL，生产架构不变。

### 仿真迁移重新筛选：后验、主动辨识与任务驱动的覆盖边界

此轮为文献补审，不增加实验编号。此前已排除泛化BayesSim/NPDR命名创新，此次检查主动采集和任务驱动是否留下可直接投入的差异。

- [NPDR](https://proceedings.mlr.press/v164/muratore22a.html)：参数后验适应与RL交替，支持相关参数、非固定分布族、不可微仿真器。不是只做点参数标定。
- [SPI-Active, CoRL2025](https://proceedings.mlr.press/v305/sobanbabu25a.html)：采样系统辨识，利用Fisher信息优化探索命令，提高真实轨迹辨识度。该文是主动辨识直接近邻，不等同已证明Bayesian full posterior必要。
- [AdaptSim, CoRL2023](https://arxiv.org/abs/2302.04903)：明确处理不可消除sim-real差距下动力学匹配未必带来任务收益的问题；通过RL元学习调整仿真参数分布，再利用少量真实交互适应。不能把它误称为Bayesian滤波器，但“按任务价值调仿真”本身已有直接先例。

核查了PMLR官方摘要和AdaptSim v2摘要/贡献文本；尚未做这些方法的复现比较。实际结论仅是上述三个宽泛创新主张不成立；不据此否定所有移动机器人sim2real。现有数据缺执行回执/独立位姿，且此前13/16执行预测未提供稳健posterior收益；目前不足以批准新训练或主线投入。

### 实验25补齐：History + human orientation

之前heading是Current+orientation，并非History+orientation。补齐这一对照，`main(combined=True)`拼接history与heading末4维(两人sin/cos)，共享全部robot proprioception，不把机器人朝向混作新增信息。两臂同HistGB预算，UM6选7/15叶，均15。运行`paper2_bi3_probe.py --history-heading` exit0；旧exploratory_information.json保留，新文件exploratory_history_heading.json。

History/组合2秒误差.544754599/.528552022，近距离.555783273/.533375107；风险触发369查询加权.567847950/.536568064。6个人-记录误差全改善，但只有3独立记录，不能视6个独立试验。开发UM6误差.579070414/.583069184，组合略差。反复探索，无确认性显著性声明。

意义限于“朝向可提供短历史之外的预测线索”，没有Bayesian优势、部署视觉测量或决策收益证据。下一步不得直接命名Bayesian姿态导航或长训PPO；需新记录与决策映射，而非在同三份记录继续调参寻找赢家。

### 实验25新记录确认：UM10-17已获得

原Dropbox目录有后续UM和LAAS条目，但ZIP次序在JSON之间夹视频。2GiB有界下载127.6秒获得UM10-17 CV JSON。stored-entry数据descriptor长度/CRC32逐项通过，JSON解析通过，保存`paper2_bi3_data/external/extraction_audit.json`；原始9份不变。不是网络失败或方向否决，更新先前尚未取得独立记录的限制。

执行`--history-heading-external`：原UM1-5训练，15叶由此前UM6决定，不重新选参。新记录不加入build默认路径，避免旧试验被静默改变。14348个人-时间查询，History/组合.634898466/.617944941；风险加权.600906362/.571361663；8/8记录改善，未把相邻帧视为独立试验做显著性声明。

官方分区分别：新训练域UM10-13 .664213219/.644080237(n7014)；验证UM14-15 .578237738/.563275139(n4068)；测试UM16-17 .632929689/.620344150(n3266)，测试风险.580906803/.557804619。仅UM站点，未覆盖LAAS。结果`external_history_heading.json`已读回，exit0。

结论限于额外朝向观测预测价值的跨记录复现。它不是Bayesian模型，本轮未证明部分可观测朝向推断/分布必要性/决策价值。未修改IL/PPO或主架构。本轮含视频下载前缀提取后清理，已验证JSON与旧243MB前缀保留。

### 朝向增强预测器上的递归偏差后验检查

沿用adaptation，新增`--heading-adaptation-external`，非生产改动。UM1-5拟合同一15叶History+heading预测器；UM6校准残差二阶矩(floor.01)并选参数；UM10-17仅评分及因果在线更新。更新仅已观察2秒目标，stride2.2秒。Gaussian常量个体偏差posterior初始std grid .01/.1/.3/1，EWMA grid0/.01/.03/.1/.3；各自以UM6 NLL选择。Student df3/5/10/30同样UM6选，shape保持相同二阶矩。

UM10-17：fixed error.617944941 NLL1.519066314；EWMA alpha.03 error.613934722 NLL1.509336592；Bayes initialstd.1 error.615832792 NLL1.511531447(plugin1.516122887)；fixedStudent df30 NLL1.497315204。UM16-17：EWMA/Bayes error .617074541/.618480146，NLL1.523092965/1.523888956；Student1.507684485。

完整参数方差比自身plugin有所改善，但未胜EWMA/固定重尾强基线。此为常量偏差近似，不据此否定所有行为Bayes；也不能因朝向point增量成立就批准Bayes主线。结果`external_heading_adaptation.json`，exit0，旧文件保留。未启动PPO。

### reciprocity负结果的场景边界与模型语义补审

实际代码`audit_saved.py:537`使用ActionHistory(BeliefEnv(...arm=no_belief),route=True)，只设置robot.visible=True；`reciprocity_reset`固定5人/nominal。`environment.py:48`默认baseline_circle。协议只有类型、动作扰动和固定步锚点，没有窄通道条件。因此历史类型Oracle负结果不能代表窄通道总问题。

但`bayesian_pilot/protocol.py:150`的non-reciprocal在委托ORCA之前删除state.human_states最后一个机器人邻居，完全不响应机器人。这不等同“拒绝提前让行但仍会紧急避碰”。直接把它放进狭窄通道，可能将不真实的碰撞压力作为posterior收益来源；不能把这种合成阳性当实机支持。

现实依据：[Stratton等2026](https://arxiv.org/abs/2601.09856)两站点80人研究报告约束空间中合作假设失效、预测ADE与导航指标不一致。它支持问题存在，但不证明删除ORCA邻居是准确人类模型。

另有直接邻近框架：[ETN, IJSR2023](https://link.springer.com/article/10.1007/s12369-023-00965-7)按人的注意状态选择意图传达，再比较预期/实际反应，失败则更换方式或改道。已读摘要及方法3节入口；其循环并非本次要发明的新概念，也不在此误称为Bayes RL。

本轮不启动窄通道仿真，不把文献边界当新方向通过。若重开，必须先有不同合作程度但保留物理避碰的响应模型依据及合法识别证据，再做Oracle决策测试；不能沿用旧二值删除机制制造收益。此为科学范围与实现语义核查，不增加实验编号。

### 实验36补测：当前几何对齐的联合协方差

执行`paper2_bi3_probe.py --joint-error-axis`。相同冻结均值模型，误差经由当前人1到人2向量定义的共同旋转映射到局部坐标；两人均同轴。仅当前几何，旋转det1，比较不受密度坐标体积影响。校准bias与cov同在该局部系，原世界系结果文件不覆盖。

Student5固定full/block 2.465157199/2.439554162；IW full/block 2.403154784/2.398240564；EWMA full/block2.392889910/2.388675063。相对block，full均只1/3记录较好，记录平均无增量。IW初始nu64、EWMA alpha.01仍仅由UM6选择。仅排除“换成当前相遇轴就能使这套联合cov胜出”的假设，不排除其他条件化模型。

结果`paper2_bi3_data/joint_error_geometry.json`，exit0，无导航/新方向批准。另复查原执行端13/16试验已有离散延迟、连续参数、ARX history与Student强对照，缺实际执行回执/独立位姿真值；没有重复启动同样执行端试验。

### 实验37：成熟GAMMA交互模型与历史后验的贡献拆分

为避免继续使用删除机器人邻居代表“不合作”，查到成熟责任分配模型。VR-ORCA(RA-L2021, https://kguo-cs.github.io/publication/multi-agent-trajectory-planning/ )调整双方责任且保持总和1；PORCA(https://adacomp.org/wp-content/uploads/2020/08/ram18.pdf )在人车接近时提高行人责任，配合POMDP。GAMMA v3(https://arxiv.org/pdf/1906.01566v3 )已将意图、注意范围、责任参数纳入贝叶斯推断。因此“Bayes推断让行程度”本身并非新的贡献。

官方仓库 https://github.com/AdaCompNUS/GAMMA 固定c797096a6c40a6368229b0883303efb588f2e2d6，克隆至`paper2_gamma_20260916`。CMake Release编译成功，默认ETH预测生成完整2099个查询帧输出。原启动会话已消失，不能追认其退出码；完整输出及后续对照进程exit0已确认。

源码审计：`Agent.cpp:843`同tag始终各承担.5，异tag根据距离和res_dec_rate归一化；注意范围仍可能排除邻居。`WorldBelief.cpp`在每次查询重新初始化先验，用history_size=2的窗口更新，有.0005平滑与似然地板。`Predict.cpp:301`取联合MAP参数，而非完整后验预测。目标候选是CV/CA外推，不是读取私有终点；开始四帧不预测。故不能把它说成完整分布进入导航的现成系统。

独立时间审计发现默认输出含当前帧，加11个未来帧，即.4–4.4秒，不能把含零误差的当前帧平均称作4.8秒预测成绩。本次剔除当前帧、未来真值10000占位及不完整序列，直接对齐原始轨迹，每个query相同11个未来点；CV用当前与上一帧差分。3776查询/303行人：默认GAMMA MAP ADE .553649082，CV .621590918；FDE1.091307607/1.209255097。不是导航，也不是论文指标正式复现。

只在临时克隆的`WorldBelief.cpp`增加`PAPER2_GAMMA_PRIOR_ONLY=1`开关，初始化后直接返回固定先验；其余模型、参数、MAP选择完全相同。生产项目未动。对照进程exit0，保存原输出`gamma_output_map.txt`，固定先验输出`gamma_output.txt`。结果`prior_ablation.json`含逐query记录，默认MAP/先验MAP/CV ADE分别.553649082/.553701184/.621590918，FDE1.091307607/1.091374218/1.209255097。仅1/303行人MAP更好；后验相对先验平均改善约0.000052m。

结论：现成成熟交互模型优于裸CV，但本配置历史后验几乎没有额外贡献，不能据11%预测改善批准Bayes主线。ETH均为People，同类型责任参数不影响配对责任，不能据此否定异构人机责任推断。此实验仅排除把默认ETH预测改善直接归因Bayes，既不是所有GAMMA配置否决，也不是新方向通过。尚无FULL对照、闭环RL或新主线批准。

实验37异构补测：AgentInfo.cpp新增PAPER2_GAMMA_UTOWN=1选择原生UTown，其余默认ETH不变。MAP/固定先验两次均exit0。相同评分器233完整查询，MAP/先验/CV ADE .374639506/.359545328/.364509211，FDE .894439255/.868990226/.890378857。moped159查询 .230325386/.208206244/.215480330；pedestrian74查询三者约.684719845。此小型随库数据仍未出现后验优势，不能批准贝叶斯责任主线；没有参数调优或真实导航。

复现评分已加入原`gamma_compute_accuracy.py --paired-audit <dataset> <map_output> <prior_output> <json>`，不改变无参数原评测行为。新增模式丢弃t=0，严格对齐原始位置与两臂真值，未来缺失排除。UTown产物`gamma_output_map_utown.txt`、`gamma_output.txt`及`prior_ablation_utown.json`。ETH先验输出已保留并重命名为`gamma_output_prior_eth.txt`，不再是gamma_output.txt。所有改动仅临时第三方克隆，无生产更改。

### 实验37实现自检修正：不能将原程序结果等同责任后验验收

继续逐行审计发现`Predict.cpp`四处GetBoundingBoxCorners将agent_num_in_frame传给frame_num。该函数确实用此参数访问pos/heading历史；应为predict_begin_frame。临时克隆仅修这四处后重新编译，UTown同233查询、两次exit0：后验MAP/先验/CV ADE .394287232/.372809180/.364509211，FDE .918438827/.884209903/.890378857。输出`gamma_output_map_utown_geometry.txt`、`gamma_output.txt`，结果`prior_ablation_utown_geometry.json`。原UTown先验输出移至`gamma_output_prior_utown.txt`保留。几何修复未产生后验优势，不改筛选门槛。

更深的范围限制：三参数PredictOneStepForOneAgentAtOneFrame在循环内向所有agent设置相同候选res_dec_rate，而不只是目标agent。Agent.cpp的双方责任raw值均为.5+(2.5-dist)*rate、同样floor.1，再归一化。因此候选rate相同时，异tag责任也恒.5；同tag本来就.5。此处候选责任参数不能通过该责任函数改变似然预测。此前“成熟代码已完整检验责任Bayes”的推断不成立；原代码有后验循环，不等于目标隐变量可辨识。注意范围也统一赋给所有邻居，不能把它描述成严格的单人参数条件似然。

本轮不把修开源bug作为论文贡献，不直接移植到IL/PPO。真实结论是基线实现需要隔离目标假设/邻居模型后才能用来验责任推断，原结果只评价其当前实现。没有新的可投入方向；上述证据防止基于失效似然再次错误否决科学问题。

实验37目标假设隔离实测：三参数一步预测仅向目标agent设置候选r_front/res_dec_rate，邻居保持addAgent的默认参数；不增加模型或调先验。UTown同233查询，MAP/固定先验/CV ADE .392108924/.372809180/.364509211，结果prior_ablation_utown_target.json。因固定先验直接返回、不调用似然，可沿用上一版本先验输出。该修正使责任候选不再由双方相同赋值必然抵消，但未证明它已对足够多真实片段可辨识。

随后发现PredictAtOneFrame的未来模拟循环使用agents_info_[i]取运动学参数，但i是活跃子集索引；正确应先getAgentID(i)后访问agents_info_[agent_id]。已修复并同时重跑两臂，均exit0。最终MAP/先验/CV ADE .394039837/.374968367/.364509211；FDE .919001982/.887634739/.890378857。moped159查询 .249088684/.230330077/.215480330，pedestrian74查询 .705488937/.685745235/.684719914。结果prior_ablation_utown_ids.json，原各版输出保留；当前gamma_output.txt为修正后先验，MAP为gamma_output_map_utown_ids.txt。

结论仍非投入级正证据。固定邻居默认参数是明确近似，非完整联合推断；这个小数据结果不能否定全部责任Bayes。到此完成三处具体实现修复及对照，不因未胜而继续随意改似然参数、增加复杂度或迁移主训练。后续若使用GAMMA须将这些本地差异与原方法区分。

### 主线候选的贡献范围再核查

本轮不增加实验编号，上一轮为有效进展（实际修复、重跑及保存结果），无运行任务需要等待。停止扩GAMMA补丁树，核查目前唯一跨新记录稳定的朝向预测线索能否直接转为贡献。

找到明确近邻：Conte/Furukawa，Autonomous Bayesian escorting of a human integrating intention and obstacle avoidance，JFR2022，https://doi.org/10.1002/rob.22070 。出版方搜索摘要明确以头部姿态推断意图，并融合意图预测与物理运动预测。直接页面返回403，本轮未声称全文审阅。它足以使“加入朝向并用Bayes融合”这个宽泛首创主张不成立，但不证明我们的具体场景、方法和问题已完全解决。Bi3的mocap朝向亦不能当作真实视觉头姿不确定性数据。

另读BNBRL+原文方法：https://arxiv.org/html/2403.10105v1 。已有有限FoV下位置历史预测、可见/不可见空间图、BNN、GRU与actor-critic。其说明不能把“遮挡belief+Bayesian RL”作为整体首次；是否存在合格的强消融或可比较公开实现仍需另核，不凭摘要宣称已复现。也不因原文写了Bayes就接受其必要性论证。

因此当前正向线索仍是“额外朝向观测改善预测”，缺部署感知/贝叶斯机制/决策收益三条，不能偷偷降低验收为预测增益即立项。现有失败均保留具体模型与任务边界，尤其不把GAMMA失败扩成真实责任不可推断。此次仅收紧贡献主张，未改生产架构、未启动RL、未批准方向。

### 跨站点数据获取及冻结确认：LAAS20记录

通过隔离安装的Playwright读取公开Dropbox目录，发现每层子目录有独立链接token，不能直接拼路径。找到JSON-only LAAS链接：https://www.dropbox.com/scl/fo/ix27cqjcknemh01u9i2f3/AB7cvvEif2aCHEVF76JIXUI/Bi3/jsons/laas?rlkey=b4pg0n0m1wjf3j5d6g8z3eqkb&dl=1 。返回laas.zip而非原42GB整包。下载达到预设2GiB后停止（下载进程exit1，明确是限额，不是完整ZIP成功）。从前缀提取所有20份cv.json，逐项size/CRC32/JSON解析通过，重新读回SHA256核验。浏览器初始16项目是未完全展开，不能据此说官方缺4份。文件在external/Bi3/jsons/laas，审计external/laas_extraction_audit.json。验证后删除2GiB前缀，原20份JSON保留。

仅修改现有paper2_bi3_probe.py增加site参数，默认UM行为保留。训练UM1-5、原UM6选择15叶及所有参数不变；LAAS20记录只用于冻结检验，编号与UM重叠不造成训练混入（分别构建列表）。--history-heading-laas exit0：44450人-时查询，History/History+heading误差.747493019/.730083558，19/20记录改善；合法CV风险触发1487查询加权.812430811/.787617440。跨站点预测线索再次成立，但不是Bayesian或闭环决策结论。

--heading-adaptation-laas exit0：UM6噪声校准、EWMA alpha.03、Bayes初始std.1、Student df30均不变；只在2秒真实未来已到达后、2.2秒stride更新。fixed error.730083558 NLL1.950851702；EWMA .670798970/1.727202869；Bayes .690956993/1.776616060(plugin1.786962248)；fixedStudent .730083558/1.925556839。实际新站点适配有收益，但这个常量偏差后验仍不敌EWMA，不能批准Bayes主线，也不扩大成所有在线Bayes无效。

产物laas_history_heading.json、laas_heading_adaptation.json。新增站点没有参与模型选择；未启动PPO，未改生产模型。此为既有25/适应筛查的真正跨站点补测，不重复包装为新研究变量。

### LAAS适应负结果的模型假设审计

沿用冻结预测器，--residual-dependence-laas只读分析非重叠2.2秒stride残差。40条人-记录，中心化lag1相关中位数.214243，31条超过各自200次顺序置换的95%参考值；首末四分之一时段均值距离中位数.669238米。描述性分析使用整记录中心化，绝不作为在线输入；未作多重比较校正，不以31条计独立显著发现。初次输出因numpy.int64 JSON序列化失败，修成int后重跑exit0，结果laas_residual_dependence.json。

这说明独立噪声+永久固定偏差并非充分假设，但相关性/时段差异不单独证明随机游走潜变量，可能由可见状态相关模型误差造成。因此上一轮只否决固定偏差适应，不能称合理动态Bayes已被否决。

同时完成稳态卡尔曼/EWMA代数单元核验：随机游走观测模型中后验P=alpha*R，过程Q=alpha²*R/(1-alpha)，则下一步增益恒alpha，均值更新与EWMA完全相同。alpha .01/.03/.1/.3各1000次二维更新，最大均值差4.58e-16。文件ewma_kalman_identity.json。这是合成数值单元检查，不是导航实验；只在所述稳态及固定噪声条件下成立，非所有Kalman/Bayes等价EWMA。

因此不再把“加过程噪声的滤波器”当独立新候选盲跑。如果重开动态适应，必须先明确未知噪声/切换或后验决策用途在何处超出这个强等价基线，并有数据支持；目前未满足。主架构不变，无PPO、无主线批准。

### 实验36跨站点补齐：联合风险与协方差适应分开判读

执行--joint-occupancy-laas，exit0。UM1-5四输出均值预测器、UM6均值偏差/协方差/超参全部沿用；LAAS20记录只评分与合法在线更新，不进入训练列表。22225同步查询，端点占据正例949。静态Student full/block/真正按人独立 Brier .101814161/.101821257/.101854703；full-block记录bootstrap差区间[-.000096566,.000083978]，full-independent[-.000173521,.000082385]，分别11/20和10/20记录改善。不能据此认为静态联合结构有稳定风险增益，且这里不是扫掠碰撞或反事实控制。

概率NLL则有新现象：IW full4.154909、IW block4.172022，EWMA full4.263188、EWMA block4.297541。IW full-EWMA full=-.108278，20/20记录改善，探索性记录bootstrap区间[-.126259,-.089757]。IW full-IW block=-.017113，区间[-.038048,+.004154]，14/20改善，联合结构本身尚不稳定。全部沿用UM6所选IW初始nu64、EWMA alpha.01。

这不是第二篇好消息：上述风险评分仍是静态模型，不能当作在线IW后验的决策评分；IW胜EWMA还可能只是累积协方差估计对本站点更合适，而非积分参数后验的价值。下一项若继续须先补“相同递推统计量的点估计/矩匹配密度”强对照，再有必要才测在线风险，不能直接接PPO。文件laas_joint_occupancy.json含逐记录结果及探索性bootstrap。本轮不宣称确认性显著结论，没有获批方向。

### 在线IW的同统计量强对照已完成

--joint-plugin-laas exit0，laas_joint_plugin.json。所有臂使用完全相同的IW递推矩阵、初始nu64、均值和每步协方差；分别用后验Student(df=nu-3)、同协方差Gaussian、固定Student（df3/5/10/30仅UM6选择，选10）评分。Gaussian协方差为原Student scale*df/(df-2)，固定Student shape再乘(10-2)/10，未偷改尺度。

LAAS20记录NLL：后验4.154909498，Gaussian4.181604138，固定Student10 4.138669777。后验-Gaussian=-.026694640，20/20记录改善，探索性记录bootstrap区间[-.032136,-.021551]；后验-固定Student=+.016239721，仅7/20改善，区间[+.000695,+.032143]。同一递推统计量的固定重尾对照已追平且均值更好，不能把上一轮IW胜EWMA解释为完整参数后验不可替代。

停止将这组概率分数作为PPO立项理由，不追加自由度搜索或LAAS调参。它只说明在线协方差估计/重尾建模有用，未完成Bayesian导航贡献。旧结果文件保留，主架构不变。

### Bi3多控制器能否提供反事实证据：实验设计与字段核查

重新读原文IV-B及实现细节（https://arxiv.org/html/2605.06863v1）：五控制器按平衡Latin square排序，非逐动作随机化；控制循环频率亦不同。因此不能把控制器条件直接当作任意机器人动作的无混淆随机工具变量。数据可扩充行为覆盖，但同一瞬时状态不同动作的反事实仍未被直接观测。

实际读取LAAS1 cv.json，仍只有robot_state/agent_states/predictions/robot_prediction/robot_goal/turning/time/logits八字段，无执行命令。对照固定commit 8b23db2e8d151e55fe04e60946837d99fde83438的HMPNav/node.py：CV分支cv_robot_prediction来自construct_cv_predictions，写出时存入robot_prediction；不是MPPI实际发出的控制序列。不能从它反推并宣称获得实际command。原论文提到命令记录不等于这份导出JSON含该字段，可能需要原始ROS数据，尚未获取验证。

本轮据此不启动将多控制器标签当因果工具的Bayes模型训练，不将预测轨迹冒充行动日志。这里是可用数据边界核查，不是新的方向否决或成功，也不能否定多控制器数据用于关联性预测的价值。

### 原始ROS数据缺口实际补齐：LAAS1 PR2命令与独立位姿

通过公开Dropbox逐层链接发现rosbags确实可用，先前不能获取是入口限制，不能再写数据不存在。仅下载LAAS1 cv.bag，155371325字节，ROS V2魔数正确，SHA256 8fc1eb07a1129f9d1d444611cbe116481fb19d08ac699ac0edd439804283a913。下载与解析exit0。文件external/laas1_cv.bag，下载审计同名前缀.download.json。没有下载全站点或视频。

实际240.494秒：/base_controller/command Twist9024条；/optitrack/bodies/pr2独立mocap28857条；/robot_pose_ekf/odom_combined7215条。还有base_scan、tracked_agents、tf与human_plans。提取command(t,vx,vy,w)、mocap/odom(t,x,y,yaw)，时间非递减且全有限；command幅值上限[.3,0,1.5]。mocap内部ts与bag时间差最大1.43e-14，导出记录显然使用统一归零时间，不能由此推断原网络延迟已知。Twist没有自身header，仍非执行ACK。

新资源的价值：可以使用真实发布命令与外部位姿检验有效执行动态，不必再用机器人预测轨迹或仅里程计替代。限制：单个PR2记录，不能直接外推TurtleBot；odom与mocap坐标不同，后续必须对齐或各自在本体坐标求增量；无逐步随机动作，不支持无假设的人类因果响应估计。

产物external/laas1_cv_signals.npz、laas1_cv_signal_audit.json已写入。该资源补齐不是研究方向通过；尚未检验模型可推断性或Bayesian决策价值，主IL/PPO不变。

### PR2 execution self-check

Existing robot_dynamics/probe.py --bi3-execution completed exit0. Chronological train ends119s, validation121--159s, test after161s; horizon boundary gaps applied. Independent mocap local pose increments, .2/.5/1s horizons. Ridge alpha selected only on validation from .01/1/100. Actual future command sequence is a known plant-identification input, not an online forecast feature.

1s position RMSE: velocity persistence .08674285m, integrated command+velocity .04576386, command+history .04250395, ordered command+history .04033843. Ordered model yaw RMSE .13314364rad. The .2/.5s ordered position errors are .008064/.017619m. Ordinary short-memory dynamics are already a strong baseline. Single recording, marker alignment and actuator ACK limitations remain; no Bayesian or decision-level approval. Result: external/laas1_execution_audit.json. Independent recordings required before claiming persistent latent execution regimes.
