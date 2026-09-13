"""Finite-pool spatial-evidence diagnostic; not an online mixture controller."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key]='1'
import copy
import hashlib
import json
import pickle
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'archive/legacy')]
import numpy as np
from scipy.special import ndtr
from scipy.stats import ncx2
from experiments.occlusion_confirmation import OUT,verify,legacy
from experiments.occlusion_closeout import save

GRID=os.environ.get('OCCLUSION_FIXED_GRID')=='1'
ALL_CANDIDATES=os.environ.get('OCCLUSION_ALL_CANDIDATES')=='1'
DEST=OUT/'binding'/('fixed_grid_all' if GRID and ALL_CANDIDATES else 'fixed_grid' if GRID else 'shape')


def quadrature(order):
    if GRID:
        edges=np.linspace(-6.,6.,order+1)
        nodes=(edges[:-1]+edges[1:])/2
        weights=np.diff(ndtr(edges))
        weights/=weights.sum()
        return np.array(np.meshgrid(nodes,nodes)).reshape(2,-1).T, np.outer(weights,weights).ravel()
    nodes,weights=np.polynomial.hermite.hermgauss(order)
    return (np.array(np.meshgrid(nodes,nodes)).reshape(2,-1).T*np.sqrt(2.),
            np.outer(weights,weights).ravel()/np.pi)


def labels(points,snapshot):
    mx,my=snapshot['mesh'];grid=snapshot['sensor']
    col=np.floor((points[:,0]-mx[0,0])/(mx[0,1]-mx[0,0])+.5).astype(int)
    row=np.floor((points[:,1]-my[0,0])/(my[1,0]-my[0,0])+.5).astype(int)
    valid=(row>=0)&(col>=0)&(row<grid.shape[0])&(col<grid.shape[1])
    result=np.full(len(points),.5)
    result[valid]=grid[row[valid],col[valid]]
    return result


def circle(distance,variance,radius):
    variance=np.broadcast_to(variance,distance.shape)
    result=np.full(distance.shape,ndtr(-8.))
    near=(distance-radius)<8*np.sqrt(np.maximum(variance,0))
    central=(distance==0)&(variance>=1e-9)
    result[central]=-np.expm1(-radius**2/(2*variance[central]))
    mask=near&(variance>=1e-9)&~central
    result[mask]=ncx2.cdf(radius**2/variance[mask],2,distance[mask]**2/variance[mask])
    result=np.where(variance<1e-9,distance<=radius,result)
    if not np.isfinite(result).all():raise RuntimeError('Invalid circle probability')
    return result


def selection(planner,controls,positions,obs,hazard):
    hc=planner._human_clearance(controls,obs)
    occupancy=obs.occupancy_probability.sample(positions) if obs.occupancy_probability is not None else None
    cost=planner._cost(controls,obs,positions,hc,occupancy,hazard)
    full,first=planner._combined_clearance(controls,obs,positions,hc,occupancy,hazard)
    _,physical=planner._physical_clearance(controls,obs,positions,hc)
    if (full>=0).any():
        pool=np.flatnonzero(full>=0);winner=int(pool[np.argmin(cost[pool])]);level=0
    elif (first>=0).any():
        pool=np.flatnonzero(first>=0);pool=pool[full[pool]>=full[pool].max()-.02]
        winner=int(pool[np.argmin(cost[pool])]);level=1
    else:
        winner=int(planner._degraded_choice(cost,physical,controls,obs));level=2
    return dict(winner=winner,action=controls[winner,0].tolist(),level=level,
        full_feasible=int((full>=0).sum()),first_feasible=int((first>=0).sum()))


def evaluate(path):
    p=verify();snapshot=pickle.loads(Path(path).read_bytes());obs=snapshot['observation']
    planner=legacy.ContinuousCEMMPC(legacy.MPCConfig(**p['configs']['bayes']))
    seed=snapshot['case']*100003+snapshot['step']*97+1729
    rng=np.random.default_rng(seed)
    bases=np.concatenate((planner._initial_mean(obs)[None],planner._route_seeds(obs)))
    noise=rng.standard_normal((512,16,2));noise[:,1:]=.68*noise[:,:-1]+.32*noise[:,1:]
    raw=bases[np.arange(512)%len(bases)]+planner.cfg.init_std*noise
    seeds=planner._seed_trajectories(obs);raw[:len(seeds)]=seeds
    _,controls,positions=planner._rollout(raw,obs)
    oldhazard=planner._belief_collision_hazard(controls,obs,positions)
    original=selection(planner,controls,positions,obs,oldhazard)
    indices=np.unique(np.r_[np.linspace(0,511,23).astype(int),original['winner']])
    if ALL_CANDIDATES:indices=np.arange(512)
    controls=controls[indices];positions=positions[indices];oldhazard=oldhazard[indices]
    baseline=selection(planner,controls,positions,obs,oldhazard)
    tracks=[t for t in snapshot['tracks'].values() if t.visible or t.existence>=.15]
    assert len(tracks)==len(obs.entities)
    orders=[]
    for order in ((25,49,97) if GRID else (15,31,63)):
        xy,baseweights=quadrature(order)
        hazard_full=np.zeros_like(oldhazard);hazard_gauss=np.zeros_like(oldhazard)
        affected=0;normalizers=[];mirror_error=0.
        for i,track in enumerate(tracks):
            means=obs.human_segment_end[i]
            variance=np.trace(obs.human_position_covariance[i],axis1=1,axis2=2)/2
            radius=obs.robot_radius+obs.entities[i,4]+planner.cfg.human_margin
            q=circle(np.linalg.norm(positions-means[None],axis=2),variance[None],radius)
            qfull=q;qgauss=q
            if not track.visible and track.covariance[0,0]>0:
                v=track.covariance[0,0]
                offsets=np.sqrt(v)*xy
                lab=labels(track.mean[:2]+offsets,snapshot)
                free=baseweights[lab==0].sum();unknown=baseweights[lab==.5].sum()
                if free>=.01 and unknown>=.01:
                    affected+=1
                    w=baseweights*(lab!=0);normalizers.append(float(w.sum()));w/=w.sum()
                    mean=w@offsets
                    centered=offsets-mean
                    covariance=(centered*w[:,None]).T@centered
                    values,vectors=np.linalg.eigh(covariance)
                    gaussian_offsets=mean+xy@(vectors*np.sqrt(np.maximum(values,0))).T
                    qfull=np.zeros_like(q);qgauss=np.zeros_like(q)
                    for k in range(16):
                        time=(k+1)*planner.cfg.dt
                        A=np.hstack((np.eye(2),time*np.eye(2)))@track.covariance[:,:2]/v
                        conditional=obs.human_position_covariance[i,k]-v*A@A.T
                        sigma=float(np.trace(conditional)/2)
                        if sigma < -1e-10 or np.max(np.abs(conditional-sigma*np.eye(2)))>1e-8:
                            raise RuntimeError('Unexpected anisotropic conditional kernel')
                        for target,offset,weight in ((qfull,offsets,w),(qgauss,gaussian_offsets,baseweights)):
                            centers=means[k]+offset@A.T
                            dist=np.linalg.norm(positions[:,k,None,:]-centers[None],axis=2)
                            values=circle(dist,max(sigma,0.),radius)
                            target[:,k]=values@weight
                            # Coupled reflection is only a numerical invariance check.
                            if k==0:
                                reflected=np.linalg.norm((positions[:,k,None,:]-centers[None])*[-1,1],axis=2)
                                mirror_error=max(mirror_error,float(np.max(np.abs(reflected-dist))))
            r=obs.human_existence[i]
            hazard_full-=np.log1p(-np.clip(r*qfull,0,1-1e-12))
            hazard_gauss-=np.log1p(-np.clip(r*qgauss,0,1-1e-12))
        choices=dict(full=selection(planner,controls,positions,obs,hazard_full),
                     gaussian=selection(planner,controls,positions,obs,hazard_gauss))
        orders.append(dict(order=order,affected_tracks=affected,normalizers=normalizers,
            choices=choices,probability_full=(-np.expm1(-hazard_full)).tolist(),
            probability_gaussian=(-np.expm1(-hazard_gauss)).tolist(),mirror_distance_error=mirror_error))
    low,high=orders[-2:]
    errors={arm:float(np.max(np.abs(np.array(low['probability_'+arm])-np.array(high['probability_'+arm]))))
            for arm in ('full','gaussian')}
    stable=all(errors[a]<=.005 and low['choices'][a]['winner']==high['choices'][a]['winner'] for a in errors)
    shape_delta=float(np.linalg.norm(np.array(high['choices']['full']['action'])-high['choices']['gaussian']['action']))
    original_delta=float(np.linalg.norm(np.array(high['choices']['full']['action'])-baseline['action']))
    result=dict(scene=snapshot['scene'],case=snapshot['case'],step=snapshot['step'],
        candidates=indices.tolist(),baseline=baseline,orders=orders,probability_refinement_error=errors,
        stable=stable,full_vs_gaussian_action_delta=shape_delta,full_vs_old_action_delta=original_delta,
        qualified=stable and shape_delta>=.05 and high['choices']['full']['level']==0)
    save(DEST/(Path(path).stem+'.json'),result)
    return result


def main():
    DEST.mkdir(exist_ok=True)
    protocol=dict(scope='Development-only one-frame spatial-evidence intervention on existing 12 fixed snapshots',
        likelihood='Zero on current observed-free cells, one elsewhere, conditional on existence; planning existence held fixed',
        arms=['old Gaussian','conditioned spatial distribution','same conditioned distribution projected to full-covariance Gaussian'],
        candidates='All 512 original finite-pool candidates; no new CEM search or environment rollouts' if ALL_CANDIDATES else '23 evenly spaced original finite-pool candidates plus old winner; no new CEM search or environment rollouts',
        numerical_orders=[25,49,97] if GRID else [15,31,63],convergence='Two finest resolutions: max candidate-step risk error <= .005 and same selected candidate in both new arms',
        quadrature='Fixed uniform standardized spatial grid on [-6,6]^2; exact Gaussian cell masses, midpoint collision integration; omitted prior mass < 4e-9' if GRID else 'Gauss-Hermite',
        continuation='At least 3 numerically stable states from 2 layouts, fully feasible shape-vs-Gaussian action change >= .05 m/s; '
        'otherwise no online mixture implementation in this round. A pass indicates choice sensitivity, NOT navigation improvement.',
        mirror='Coupled distance reflection is an algebraic self-test only, not a novelty test.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        snapshots={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((OUT/'mechanism').glob('*.pkl'))})
    path=DEST/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise RuntimeError('Probe protocol changed')
    save(path,protocol)
    with ProcessPoolExecutor(6) as pool:
        results=[]
        for result in pool.map(evaluate,protocol['snapshots']):
            results.append(result)
            print('Completed',result['scene'],result['step'],'stable',result['stable'],
                  'action delta',result['full_vs_gaussian_action_delta'],flush=True)
    qualified=[r for r in results if r['qualified']]
    summary=dict(states=len(results),stable_states=sum(r['stable'] for r in results),
        qualified_states=len(qualified),qualified_layouts=len({(r['scene'],r['case']) for r in qualified}),
        passed=len(qualified)>=3 and len({(r['scene'],r['case']) for r in qualified})>=2,
        rows=[{k:r[k] for k in ('scene','case','step','stable','probability_refinement_error',
              'full_vs_gaussian_action_delta','full_vs_old_action_delta','qualified')} for r in results],
        limitation='Single-frame conditioning of old Gaussian, not a recursive negative-evidence filter. '
        'Numerical nonconvergence is not evidence against negative evidence. No new safety or Bayesian indispensability claim.')
    save(DEST/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
