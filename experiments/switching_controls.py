"""Known switches and nuisance shifts: tests of mechanism, not real navigation."""
import numpy as np
from experiments.switching_validation import Bank,filtered,save


def run():
    rng=np.random.default_rng(91023)
    means=np.array([[-.15,0],[0,0],[.15,0]])
    train={str(i):dict(t=np.arange(120)*.25,e=rng.normal(means[i%3],.025,(120,2)))
           for i in range(30)}
    bank=Bank(3,11).fit(train,list(train)); bank.residual_reference=.05
    single=Bank(1,11).fit(train,list(train)); single.residual_reference=.05
    names=['residual','ewma','cusum','bank_nll','update_kl','single_nll']

    def scores(e):
        s=dict(t=np.arange(len(e))*.25,e=e)
        _,_,d=filtered(bank,s)
        _,_,g=filtered(single,s)
        return np.c_[d,g[:,3]]

    calibration=np.concatenate([scores(rng.normal([0,0],.025,(120,2)))[20:]
                                for _ in range(30)])
    thresholds=np.quantile(calibration,.95,axis=0)
    output=[]
    for kind in ('no_change','known_mode_switch','unknown_lateral_drift','noise_increase'):
        values=[]
        for _ in range(200):
            e=rng.normal([0,0],.025,(120,2))
            if kind=='known_mode_switch': e[60:]+=np.array([.15,0])
            elif kind=='unknown_lateral_drift': e[60:]+=np.array([0,.15])
            elif kind=='noise_increase': e[60:]=rng.normal([0,0],.12,(60,2))
            values.append(scores(e))
        values=np.array(values)
        for j,name in enumerate(names):
            alerts=values[:,:,j]>thresholds[j]
            delays=[]
            for a in alerts[:,60:]:
                hits=np.flatnonzero(a)
                if len(hits): delays.append((hits[0]+1)*.25)
            output.append(dict(condition=kind,method=name,trials=200,
                prechange_step_alarm_rate=float(alerts[:,20:60].mean()),
                first_second_alarm_rate=float(alerts[:,60:64].any(1).mean()),
                detected_in_15s=float(alerts[:,60:].any(1).mean()),
                median_delay_given_detection=float(np.median(delays)) if delays else None))
    # Same innovations can arise from distinct unobserved causes; every score is identical.
    e=rng.normal([0,0],.025,(120,2)); e[60:]+=[0,.15]
    indistinguishable=float(np.max(np.abs(scores(e)-scores(e.copy()))))
    save('synthetic_controls.json',dict(results=output,thresholds=dict(zip(names,thresholds)),
        identical_observation_different_cause_score_difference=indistinguishable,
        limitations='Known change times are synthetic. No-change post60 alarm is not detection. '
                    'Single-step 5% calibration does not imply episode-level 5% false alarms. '
                    'No algorithm can anticipate an independent unsignalled random switch from identical prefixes.'))


if __name__=='__main__': run()
