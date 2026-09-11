"""Original exact risk and unchanged shared Hermite envelopes.

Bounds retain the audited floating padding, not interval certification.
"""
from __future__ import annotations
import math
from typing import Optional
import numpy as np
from scipy.special import ndtr, ive
from scipy.stats import ncx2
from .contracts import PlannerObservation

def exact_hazard(
    cfg,
    controls: np.ndarray,
    obs: PlannerObservation,
    positions: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Bernoulli-Gaussian collision hazard per step.

    The current filter's position covariance is isotropic, so squared
    radial distance follows a non-central chi-square distribution.  Its
    CDF gives the exact probability mass inside the collision disc.  The
    additive negative log-survival form preserves ordering when union
    probabilities round to one in dense crowds.
    """
    if (
        obs.human_position_covariance is None
        or obs.human_existence is None
        or not obs.entities.size
    ):
        return None
    if positions is None:
        positions = obs.robot_xy[None, None, :] + np.cumsum(
            controls * cfg.dt, axis=1
        )
    human_positions = obs.human_segment_end
    relative = positions[:, None, :, :] - human_positions[None, :, :, :]
    distance = np.linalg.norm(relative, axis=3)
    variance = 0.5 * np.trace(
        obs.human_position_covariance, axis1=2, axis2=3
    )
    collision_radius = (
        obs.robot_radius
        + obs.entities[None, :, None, 4]
        + cfg.human_margin
    )
    scaled_radius = np.square(collision_radius) / np.maximum(
        variance[None, :, :], 1e-9
    )
    noncentrality = np.square(distance) / np.maximum(
        variance[None, :, :], 1e-9
    )
    # A collision disc lies inside its near-side tangent half-plane.
    # Beyond 8 sigma use the upper bound Phi(-8), never drop a person.
    # Per-step hazard error is at most N * 6.23e-16 (roundoff aside).
    far = (distance - collision_radius) >= 8.0 * np.sqrt(variance)[None, :, :]
    conditional = np.full(distance.shape, ndtr(-8.0))
    near = ~far
    conditional[near] = ncx2.cdf(
        np.broadcast_to(scaled_radius, distance.shape)[near],
        2.0, noncentrality[near],
    )
    deterministic = variance < 1e-9
    if np.any(deterministic):
        conditional = np.where(
            deterministic[None, :, :],
            distance <= collision_radius,
            conditional,
        )
    existence = obs.human_existence
    if cfg.existence_override is not None:
        existence = np.full_like(existence, cfg.existence_override)
    component = conditional * existence[None, :, None]
    return -np.log1p(-np.clip(component, 0.0, 1.0 - 1e-12)).sum(axis=1)


class CompiledDiscRisk:
    """Per-observation Gaussian-disc table with analytic interpolation brackets.

    q(z)=P(||N((R+sigma*z,0),sigma^2 I)||<=R).
    The fourth derivative is bounded by sqrt(24)/2, from the zero mean
    and L2 norm of the fourth probabilists' Hermite polynomial. Cubic-Hermite
    interpolation error <= sqrt(24)*h^4/768. Floating table values and
    derivatives mean this implementation is NOT interval arithmetic.
    """
    def __init__(self,cfg,obs,step=.125):
        self.cfg,self.obs,self.step=cfg,obs,float(step)
        if not np.isclose(16./step,round(16./step)):
            raise ValueError('table step must divide 16')
        self.grid=np.linspace(-8.,8.,round(16./step)+1)
        self.variance=.5*np.trace(obs.human_position_covariance,axis1=2,axis2=3)
        self.s=np.sqrt(np.maximum(self.variance,1e-9))
        self.radius=obs.robot_radius+obs.entities[:,4,None]+cfg.human_margin
        self.a=np.broadcast_to(self.radius,self.variance.shape)/self.s
        unique_a, inverse = np.unique(self.a, return_inverse=True)
        self.component_map = inverse.reshape(self.variance.shape)
        centers=unique_a[:,None]+self.grid
        self.table=ncx2.cdf(unique_a[:,None]**2,2.,centers**2)
        self.derivative = (-unique_a[:,None] * np.sign(centers)
            * np.exp(-.5*(np.abs(centers)-unique_a[:,None])**2)
            * ive(1,unique_a[:,None]*np.abs(centers)))
        self.table_cdf_values=int(self.table.size)
        # d^4/dz^4 Gaussian set mass = E[1_A He_4(Z)]. Since E He_4=0,
        # its absolute value <= .5 E|He_4| <= sqrt(4!)/2.
        # Cubic Hermite error <= M4*h^4/384.
        self.interpolation_error=np.sqrt(24.)/2.*step**4/384.

    def conditional_bounds(self,positions):
        cfg,obs=self.cfg,self.obs
        d=np.linalg.norm(positions[:,None,:,:]-obs.human_segment_end[None,:,:,:],axis=3)
        z=(d-self.radius[None,:,:])/self.s[None,:,:]
        index=(z+8.)/self.step
        left=np.clip(np.floor(index).astype(np.int64),0,len(self.grid)-2)
        frac=np.clip(index-left,0.,1.)
        component=self.component_map
        table=self.table
        qleft=table[component[None,:,:],left]
        qright=table[component[None,:,:],left+1]
        derivative=self.derivative
        gleft=derivative[component[None,:,:],left]
        gright=derivative[component[None,:,:],left+1]
        t=frac; t2=t*t; t3=t2*t
        approx=((2*t3-3*t2+1)*qleft + (t3-2*t2+t)*self.step*gleft
                +(-2*t3+3*t2)*qright + (t3-t2)*self.step*gright)
        lower=approx-self.interpolation_error
        upper=approx+self.interpolation_error
        positive_left=self.a[None,:,:]+self.grid[left]>=0.
        lower=np.where(positive_left,np.maximum(lower,qright),lower)
        upper=np.where(positive_left,np.minimum(upper,qleft),upper)
        lower=np.maximum(0.,lower-1e-12)
        upper=np.minimum(1.,upper+1e-12)
        # A radius-gap of 8sigma contains a centered 8sigma ball.
        lower=np.where(z < -8.,1.-np.exp(-32.)-1e-12,lower)
        upper=np.where(z < -8.,1.,upper)
        # Reproduce reference's far-tail numerical convention exactly.
        far=z>=8.
        lower=np.where(far,ndtr(-8.),lower)
        upper=np.where(far,ndtr(-8.),upper)
        deterministic=self.variance[None,:,:]<1e-9
        lower=np.where(deterministic,d<=self.radius,lower)
        upper=np.where(deterministic,d<=self.radius,upper)
        return lower,upper

    def bounds(self,positions):
        cfg,obs=self.cfg,self.obs
        lower,upper=self.conditional_bounds(positions)
        existence=obs.human_existence
        if cfg.existence_override is not None:
            existence=np.full_like(existence,cfg.existence_override)
        if np.any(existence<0) or np.any(existence>1):
            raise ValueError('existence must be between zero and one')
        def aggregate(p):
            return -np.log1p(-np.clip(p*existence[None,:,None],0.,1.-1e-12)).sum(axis=1)
        return aggregate(lower),aggregate(upper)


def _keys(cost, full, first):
    feasible, first_feasible = full >= 0, first >= 0
    category = np.where(feasible, 0, np.where(first_feasible, 1, 2))
    violation = np.where(feasible, 0., np.where(first_feasible, -full, -first))
    return category, violation, cost


def _le_key(keys, cut):
    category, violation, cost = keys
    c, v, j = cut
    # Keep ALL exact ties, so sorting/tie-breaking remains the reference's job.
    return ((category < c) | ((category == c) & ((violation < v) |
            ((violation == v) & (cost <= j)))))
