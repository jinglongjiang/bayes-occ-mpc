"""Content-verified merge of per-machine result blocks.

A same-named file appearing on both machines is not assumed identical: the
episode records are compared field by field.  Identical files are dropped,
differing files are reported as a conflict and neither copy is silently kept.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

FIELDS = ("case_id", "event", "success", "collision", "timeout", "nav_time",
          "path_length", "collision_union", "success_without_overlap")


def signature(path):
    d = json.load(open(path))
    e = d["episodes"]
    return [tuple(round(x[f], 9) if isinstance(x.get(f), float) else x.get(f)
                  for f in FIELDS) for x in sorted(e, key=lambda r: r["case_id"])]


def main():
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    target.mkdir(parents=True, exist_ok=True)
    added = identical = conflicts = 0
    for path in sorted(source.glob("*.json")):
        destination = target / path.name
        if not destination.exists():
            shutil.copy2(path, destination)
            added += 1
            continue
        try:
            same = signature(path) == signature(destination)
        except Exception as exc:
            print(f"  冲突 {path.name}: 无法比对 ({exc})")
            conflicts += 1
            continue
        if same:
            identical += 1
        else:
            conflicts += 1
            print(f"  ★ 冲突 {path.name}: 同名但内容不同，两份都保留待人工判定")
            shutil.copy2(path, target / (path.stem + ".REMOTE_CONFLICT.json"))
    print(f"合并 {source} -> {target}: 新增 {added}，内容一致 {identical}，冲突 {conflicts}")
    if conflicts:
        raise SystemExit(f"存在 {conflicts} 处同名内容冲突，合并未通过")


def transfer_report(root):
    import hashlib
    import numpy as np
    root = Path(root)
    arms = ['bayes', 'ewma', 'age_margin', 'conformal', 'static_cov']
    manifests = [json.loads((root/c/'protocol.json').read_text()) for c in ['clean','severe']]
    assert manifests[0]['source_hashes'] == manifests[1]['source_hashes']
    for name, digest in manifests[0]['source_hashes'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    expected = list(range(5100,5200))
    values, summary = {}, {}
    for condition in ['clean','severe']:
        summary[condition] = {}
        for arm in arms:
            block=json.loads((root/condition/(arm+'_sc3.json')).read_text())
            rows=sorted(block['episodes'],key=lambda r:r['case_id'])
            assert [r['case_id'] for r in rows] == expected
            assert block['frozen_sensor']
            assert block['point'] == manifests[0]['points'][arm]
            a=np.array([[r['success_without_overlap'],r['collision_union'],
                         r['nav_time'] if r['success_without_overlap'] else 25.] for r in rows],float)
            values[condition,arm]=a
            summary[condition][arm]={'successes':int(a[:,0].sum()),'collisions':int(a[:,1].sum()),
                'other_failures':int(len(a)-a[:,0].sum()-a[:,1].sum()),'penalized_time':float(a[:,2].mean())}
    rng=np.random.default_rng(2407)
    ix=rng.integers(0,100,size=(10000,100))
    comparisons={}
    for arm in arms[1:]:
        d=(values['severe','bayes']-values['clean','bayes'])-(values['severe',arm]-values['clean',arm])
        level=.975 if arm in ['ewma','age_margin'] else .95
        ci=np.quantile(d[ix].mean(axis=1),[(1-level)/2,1-(1-level)/2],axis=0)
        comparisons[arm]={'confidence':level,'metrics':{name:{'mean':float(d[:,j].mean()),'ci':ci[:,j].tolist()}
                         for j,name in enumerate(['success_change','collision_change','time_change'])}}
    result={'n_layouts':100,'n_episodes':1000,'summary':summary,'paired_change_comparisons':comparisons,
            'scope':'shared Bayesian tracker, frozen clean assumptions; risk representation migration only'}
    (root/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['\n\n执行结果：1000回合完成，case集合、冻结参数和源文件hash全部验证。',
           '方法 | clean成功/碰撞/其他失败 | severe成功/碰撞/其他失败 | clean/severe惩罚时间(s)']
    for arm in arms:
        a,b=summary['clean'][arm],summary['severe'][arm]
        lines.append(f"{arm} | {a['successes']}/{a['collisions']}/{a['other_failures']} | {b['successes']}/{b['collisions']}/{b['other_failures']} | {a['penalized_time']:.3f}/{b['penalized_time']:.3f}")
    for arm,c in comparisons.items():
        lines.append(f"Bayes相对{arm}的退化差（{100*c['confidence']:.1f}%区间）：")
        for metric,v in c['metrics'].items():
            scale=1 if metric=='time_change' else 100
            lines.append(f"  {metric}: {v['mean']*scale:.3f} [{v['ci'][0]*scale:.3f}, {v['ci'][1]*scale:.3f}]")
    report=root/'报告.txt'
    original=report.read_text().split('\n\n执行结果：')[0].replace('状态：执行中，尚无结论。','状态：已完成；最终判断见末尾。')
    report.write_text(original+'\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    if sys.argv[1] == '--transfer-report':
        transfer_report(sys.argv[2])
    else:
        main()
