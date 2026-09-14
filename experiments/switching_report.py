"""Generate the compact Chinese audit from preserved result tables."""
import json
from experiments.switching_validation import OUT,H


def run():
    def interval(value):
        return '['+', '.join(f'{v:.6f}' for v in value)+']' if value is not None else '不可估计'
    linear=json.loads((OUT/'summary.json').read_text())
    nonlinear=json.loads((OUT/'nonlinear/summary.json').read_text())
    lookup={x['subset']:x for x in nonlinear}
    names={'all_loso':'全部六记录留出','moving_loso':'移动记录整段留出','moving_temporal':'移动记录内部时间留出',
           'prefix_change_loso':'当前前缀变化较大片段','ordinary_loso':'其余普通片段'}
    lines=['切换动力学／Self-Aware Bayesian方案验证报告',
        '日期：2026-09-14；分支：switching-dynamics-validation-20260914',
        '', '一、结论先说',
        '模型库有小幅真实预测收益，不能宣布整个方向无效。但没有证据支持把它升级为开放世界自感知导航主线。',
        '同总树数的非线性模式库总体误差0.4067→0.4003米，改善约1.6%；前缀变化片段约4.5%。',
        'FULL相对同一模式库MAP改善约4.94毫米；时序递推相对逐帧独立模式加权仅改善约0.36毫米。',
        '异常检测不是无用，但简单残差/EWMA仍有竞争力；大误差无法独自证明出现了新动力学。',
        '这是预测原型筛查，不是三篇通信论文复现、完整SLDS实现、机器人闭环或SCI新颖性证明。',
        '', '二、做了什么，避免把坏实现当坏方向',
        '沿用上一轮固定查询和完整轨迹划分：1520条轨迹、6011个查询、26823个已发生位移更新。',
        '六次按记录留出，另作移动记录内部时间留出（16条轨迹、69查询，来自同一移动记录）。',
        '每种预测器三种种子11/29/47，不挑最好种子；统计先平均种子，不把种子当新数据。',
        '主实验：3/6个学习模式，Ridge专家alpha=1/10/100；验证集选择，测试不调参。',
        '补充：模式内换ExtraTrees，总96棵树，与单一树模型相同；叶子5/15只在验证集选择。',
        '这个补充在看到线性结果后登记，属于容量敏感性检查，不伪称事前确认。',
        'CV/CA/CT及简单贝叶斯运动库均保留。简单运动库不是完整IMM-Kalman，不代表所有IMM上限。',
        '模式由训练创新聚类得出，不用Avoid等整段标签；查询后验只处理截至当前的新位移。',
        '预测专家给各时域直接输出，不逐步采样全部未来模式切换；这是明确的原型范围限制。',
        'IID对照使用同一组按递推权重训练的专家，仅替换查询权重；不是重新优化后的IID上限。',
        '通过14项自检：包括未来扰动不影响过去、Bayes赔率、恒速、模式拆分不变、可学切换与预测正控制。',
        '另跑800条合成序列：正常、已知模式切换、未知侧向漂移、噪声增大各200条。',
        '', '三、预测主表（米；0.5/1/2/4秒平均，轨迹等权，越小越好）',
        '范围 | 轨迹 | CV | 单一线性 | 单一树 | 线性模式FULL | 非线性FULL | 非线性MAP | 非线性IID']
    for a,b in zip(linear,nonlinear):
        m=b['means']
        vals=[m['cv'],m['universal_ridge'],m['universal_trees'],a['means']['learned_full'],
              m['learned_full'],m['learned_map'],m['learned_iid']]
        lines.append(f"{names[b['subset']]} | {b['tracks']} | "+' | '.join(f'{v:.4f}' for v in vals))
    lines += ['', '1/2/4秒分开看（非线性；各行依次为单一树/FULL/MAP/IID）']
    for r in nonlinear[:3]:
        for j in [1,2,3]:
            values=[r['per_horizon'][m][j] for m in ['universal_trees','learned_full','learned_map','learned_iid']]
            lines.append(f"{names[r['subset']]} {H[j]:g}秒："+'/'.join(f'{v:.4f}' for v in values))
    lines += ['', '配对差及95%探索性区间（FULL减对照，负数才是改善）']
    for r in nonlinear:
        for base in ['universal_trees','learned_map','learned_iid']:
            key='learned_full-minus-'+base
            a=r['paired_track'][key];b=r['paired_session'][key]
            lines.append(f"{names[r['subset']]} vs {base}：轨迹等权 {a['mean']:.6f} {interval(a['ci95'])}；记录等权 {b['mean']:.6f} {interval(b['ci95'])}")
    lines += [
        '普通与变化分层由当前合法创新超过训练75分位定义，不按事后预测失败挑样本。',
        '同一个人可同时贡献普通和变化查询，所以分层轨迹数不能相加当总人数。',
        '六记录区间只作探索描述，交叉验证训练集有重叠；不能包装成大规模独立确认。',
        '移动只有一个记录，记录级区间留空；16轨迹时间留出不够支持稳定跨场景结论。',
        '非线性FULL在移动整记录留出仍差于CV（0.5781 vs 0.5458）；不能只说胜过0.5986的树模型。',
        '', '四、分布与固定几何风险，不只看点误差',
        '每折共享从验证集通用模型残差拟合的各时域Gaussian核；FULL不把模式均值当实际路径。',
        '核是条件核近似，不证明每个模式分别校准；不同核标定可能改变密度排序。',
        'NLL评分整个位移预测密度；Brier评分机器人当前速度外推线上0.7米圆盘的占据概率。',
        '几何探针不模拟行人对替代机器人动作的反应，不能称碰撞率、候选最优动作或导航收益。',
        '范围 | 查询时域命中数 | 单一树Brier | FULL Brier | MAP Brier | 单一树NLL | FULL NLL | MAP NLL']
    for r in nonlinear[:3]:
        m=r['means'];vals=[m[k] for k in ['universal_trees_brier','learned_full_brier','learned_map_brier',
                                         'universal_trees_nll','learned_full_nll','learned_map_nll']]
        lines.append(names[r['subset']]+f" | {r['probe_events']} | "+' | '.join(f'{v:.6f}' for v in vals))
    lines += ['命中是查询×时域的几何标签，不是独立碰撞事件；全部6011×4个标签只有182次命中。',
              '完整密度的NLL有收益，不等于机器人动作一定受益；Brier差值区间详见summary.json。',
              '', '五、是否更早知道模型失效',
              '真实数据没有逐帧切换真值。这里预测的是未来2秒CV误差超过训练90分位，不叫真实未知模式检测。',
              '阈值仅用验证集非事件分位固定。以下六记录等权均值不是统一保证5%测试误报。',
              '方法 | AUROC | AP | 实际FPR | TPR']
    import pandas as pd
    d=pd.DataFrame(json.loads((OUT/'detection.json').read_text()))
    agg=d[d.fold!='moving_temporal'].groupby('method')[['auc','ap','fpr','tpr']].mean()
    for method,r in agg.iterrows():
        lines.append(method+' | '+' | '.join(f'{v:.4f}' for v in r))
    lines += ['模型库NLL的总体AUROC较好，但AP与固定验证阈值下的召回不优于简单残差。',
              '移动记录整段留出：模型库NLL AUROC约0.664，EWMA约0.716；不能声称普遍更早。',
              '实现里的KL是posterior对预测prior的KL，不是通信论文KLDA的逐公式复现。',
              '合成已知切换时，残差和模型库均可在首个0.25秒更新报警，没有展示独立提前优势。',
              '合成未知侧向漂移时，KL首秒报警仅21.5%，残差/NLL为100%；噪声增大也让它们报警。',
              '单步5%阈值的正常15秒窗口也频繁报警，不能将单步误报率冒充整个交互的误报保证。',
              '因果边界：相同观测创新可由换目标、未观测交互或模型变化产生，单靠分数不能区分。',
              '', '六、三篇论文与最近邻核查',
              'Jammer Detection（2020）：已验证检测与定位；结论把刻画jammer、增量交互学习及抗干扰列为后续。',
              'Automatic Modulation（2021）：验证多GDBN分类；增量模型学习的全部闭环并非已完成成果。',
              'Semantic-Aware Resource Allocation：验证数字功率动作；物理轨迹规划明确属于后续工作。',
              '因此，三篇论文不能作为“未知行为自动学会并改善机器人导航”的现成证据。',
              '直接近邻：Kooij等Context-Based Path Prediction for Targets with Switching Dynamics，2018在线/2019卷期。',
              'https://link.springer.com/article/10.1007/s11263-018-1104-4',
              '它已将上下文DBN接入行人/骑行者SLDS预测，并区分看到切换后反应与利用前兆提前预测。',
              'Self-Awareness汽车研究在2018年也已有多模态生成模型与异常检测：',
              'https://arxiv.org/abs/1806.02609',
              '学习局部动力学及依赖状态的切换也已有recurrent SLDS：',
              'https://arxiv.org/abs/1610.08466',
              '本地report.md还记有旧GDBN与CV/K1对照；新数据和新实现不等于这个机制首次被尝试。',
              '', '七、哪些已经能判断，哪些不能装作判断了',
              '已能判断：本次学习模型库有小幅预测收益；只做线性专家会漏掉这项收益。',
              '已能判断：本次跨帧递推的新增预测收益极小，未建立“必须动态Bayes”的强机制证据。',
              '已能判断：异常分数不是未知类型后验；不加未知模型似然/拒识机制，已知模式权重仍只能和为1。',
              '尚不能判断：未来新类型自动增量学习、不同机器人动作下的人类反应、实际安全与导航收益。',
              'PeRoI缺少真实切换时刻、原始检测缺测标记和同状态反事实动作，不能靠自造模拟器补出这些事实。',
              '这不是“所有切换模型都不行”的判决。也是不能据此立即立项开放世界导航的具体原因。',
              '取舍：保留模型库为小幅预测改进结果；不将Self-Aware/自动学新规律写成已成立贡献，不升级正式MPC。',
              '', '八、复现与交付',
              '完整数据：官方PeRoI Zenodo记录18876411，dataset-PeRoI.zip；沿用上一轮下载并核对的完整发布包。',
              'https://zenodo.org/records/18876411',
              'python -m unittest experiments.test_switching_validation experiments.test_response_validation',
              'python -m experiments.switching_validation /path/to/peroi-full/data',
              'python -m experiments.switching_controls',
              'python -m experiments.switching_summary',
              'python -m experiments.switching_nonlinear /path/to/peroi-full/data',
              'python -m experiments.switching_summary --directory results/switching_validation/nonlinear',
              'python -m experiments.switching_report',
              'python -m experiments.verify_switching_delivery --data /path/to/peroi-full/data',
              '原始逐查询结果以queries.csv.gz发布，包含全部种子、各时域误差和概率评分；主结果与敏感性分目录。',
              'manifest.json记录实际数据和源码SHA256；verification.json记录同机器校验，不冒充跨环境复现。',
              '本轮未改变原导航器、旧数据、论文结论；没有重跑导航成功率，也没有上传外部PDF。']
    (OUT/'REPORT_ZH.txt').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':run()
