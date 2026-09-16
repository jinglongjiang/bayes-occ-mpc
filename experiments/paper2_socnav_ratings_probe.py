"""Exploratory new-rater inference on shared control items, not new trajectories."""
import ast
import json
from pathlib import Path
import numpy as np
from scipy.stats import multivariate_normal, spearmanr
from sklearn.covariance import LedoitWolf

ROOT = Path('/home/abc/temp/paper2_socnav_ratings_data')
CODE = Path('/home/abc/temp/paper2_socnav_ratings_20260916/tools/data_analysis/check_quality.py')


def load(repeated=False):
    tree = ast.parse(CODE.read_text())
    relevant = next(ast.literal_eval(n.value) for n in tree.body
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
                    t.id == 'RELEVANT' for t in n.targets))
    keys = [(v[1], ' '.join(v[2].split())) for v in relevant]
    ratings, repeat_noise, second = [], [], []
    for path in sorted((ROOT / 'all_ratings').rglob('*.json')):
        d = json.loads(path.read_text(), strict=False)
        own = {}
        for q, v in sorted(d['answers'].items(), key=lambda pair: int(pair[0])):
            j = int(q)
            key = (d['indices'][j], ' '.join(d['descriptions'][j].split()))
            own.setdefault(key, []).append(v)
        if not all(k in own for k in keys):
            continue
        ratings.append([own[k][0] for k in keys])
        second.append([own[k][1] if len(own[k])>1 else np.nan for k in keys])
        repeat_noise.append(np.mean([(v[0]-v[1])**2/2 for v in own.values() if len(v)>1]))
    if repeated:return np.asarray(ratings),np.asarray(repeat_noise),np.asarray(second)
    return np.asarray(ratings), np.asarray(repeat_noise)


def main():
    y, noise = load()
    rows = []
    for i in range(len(y)):
        train = np.delete(y, i, 0)
        mu = train.mean(0)
        observation_noise = max(float(np.delete(noise, i).mean()), .0025)
        observed_cov = LedoitWolf().fit(train).covariance_
        # Remove estimated rating noise, then project to a valid latent covariance.
        eigenvalues, vectors = np.linalg.eigh(observed_cov - observation_noise*np.eye(15))
        cov = (vectors*np.maximum(eigenvalues, 1e-6)) @ vectors.T
        cross = cov[5:, :5]
        inverse = np.linalg.inv(cov[:5, :5] + observation_noise*np.eye(5))
        post_mean = mu[5:] + cross @ inverse @ (y[i, :5] - mu[:5])
        post_cov = cov[5:, 5:] - cross @ inverse @ cross.T
        total = post_cov + observation_noise*np.eye(10)
        for arm, pred, var in [('population', mu[5:], cov[5:, 5:] + observation_noise*np.eye(10)),
                               ('bayes', post_mean, total)]:
            err = y[i, 5:] - pred
            rows.append(dict(rater=i, arm=arm, mse=float(np.mean(err**2)),
                             clipped_mse=float(np.mean((y[i, 5:] - np.clip(pred, 0, 1))**2)),
                             spearman=float(spearmanr(pred, y[i, 5:]).correlation),
                             nll=float(-multivariate_normal.logpdf(err, cov=var)/10)))
    summary = dict(scope='exploratory common controls only; new rater, known items; no trajectory generalization or navigation',
                   protocol='leave one rater out; first5calibrate,last10predict; no demographic features',
                   limitations='Gaussian approximate bounded scores; noise from repeated ratings; only34raters; not a novel algorithm',
                   raters=len(y), rows=rows, arms={})
    for arm in ['population', 'bayes']:
        subset = [r for r in rows if r['arm']==arm]
        summary['arms'][arm] = {k:float(np.mean([r[k] for r in subset]))
                                for k in ['mse','clipped_mse','spearman','nll']}
    (ROOT / 'structured_preference.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary['arms'], indent=2))


def active_choice():
    y,noise,second=load(repeated=True)
    assert np.isfinite(second[:,:5]).all()
    from numpy.polynomial.hermite import hermgauss
    nodes,weights=hermgauss(32);nodes=nodes*np.sqrt(2);weights=weights/np.sqrt(np.pi)
    rows=[]
    for i in range(len(y)):
        train=np.delete(y,i,0);mu=train.mean(0)
        obsnoise=max(float(np.delete(noise,i).mean()),.0025)
        eig,vec=np.linalg.eigh(LedoitWolf().fit(train).covariance_-obsnoise*np.eye(15))
        cov=(vec*np.maximum(eig,1e-6))@vec.T
        for arm in ['population','random','variance','knowledge_gradient','diagonal_kg']:
            values=[];actions=[];queries=[]
            for seed in range(100 if arm=='random' else 1):
                rng=np.random.RandomState(2407+seed);mean=mu.copy()
                sigma=np.diag(np.diag(cov)) if arm=='diagonal_kg' else cov.copy()
                asked=[]
                for step in range(0 if arm=='population' else 3):
                    available=np.array([j for j in range(15) if j not in asked])
                    if arm=='random':j=int(rng.choice(available))
                    elif arm=='variance':j=int(available[np.argmax(np.diag(sigma)[available])])
                    else:
                        scores=[]
                        for q in available:
                            possible=mean[:5,None]+sigma[:5,q,None]/np.sqrt(sigma[q,q]+obsnoise)*nodes
                            scores.append(float(np.max(possible,axis=0)@weights))
                        j=int(available[np.argmax(scores)])
                    gain=sigma[:,j]/(sigma[j,j]+obsnoise)
                    mean=mean+gain*(y[i,j]-mean[j])
                    sigma=sigma-np.outer(gain,sigma[j,:]);sigma=(sigma+sigma.T)/2
                    asked.append(j)
                action=int(np.argmax(mean[:5]))
                values.append(float(second[i,action]));actions.append(action);queries.append(asked)
            rows.append(dict(rater=i,arm=arm,heldout_rating=float(np.mean(values)),
                             hindsight_best=float(np.max(second[i,:5])),actions=actions,queries=queries))
    arms={}
    base=np.array([r['heldout_rating'] for r in rows if r['arm']=='population'])
    rng=np.random.RandomState(2407);indices=rng.randint(len(y),size=(10000,len(y)))
    for arm in ['population','random','variance','knowledge_gradient','diagonal_kg']:
        val=np.array([r['heldout_rating'] for r in rows if r['arm']==arm]);delta=val-base
        arms[arm]=dict(rating=float(val.mean()),delta=float(delta.mean()),
                       exploratory_paired_ci=np.quantile(delta[indices].mean(1),[.025,.975]).tolist())
    result=dict(scope='real repeated ratings, known five control trajectories, no robot rollout or new-task generalization',
                protocol='leave-one-rater-out prior;3distinct queries of first ratings among15items;choose amongfirst5;score solely second ratings;100random orders',
                caveats=['34raters;Gaussian approximation;five control cases only',
                         'queries are answer lookups, not live user experiment',
                         'second ratings noisy;hindsight maximum is optimistic, not true utility',
                         'dataset repeatedly explored;CIs exploratory, not search-adjusted',
                         'knowledge gradient with Gaussian conditioning is existing methodology, not new algorithm'],
                arms=arms,hindsight_rating=float(np.mean([np.max(r[:5]) for r in second])),rows=rows)
    (ROOT/'active_choice.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__ == '__main__':
    import sys
    if '--active-choice' in sys.argv:active_choice()
    else:main()
