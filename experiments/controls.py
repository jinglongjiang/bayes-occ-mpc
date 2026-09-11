"""Cached rectangle control, unchanged from the audited comparison."""
import numpy as np
from scipy.special import ndtr


class CachedRectangle:
    def __init__(self, cfg, obs):
        self.cfg, self.obs = cfg, obs
        self.variance = .5*np.trace(obs.human_position_covariance,axis1=2,axis2=3)
        self.s = np.sqrt(np.maximum(self.variance,1e-9))
        self.radius = obs.robot_radius+obs.entities[:,4,None]+cfg.human_margin
        self.inner = self.radius/np.sqrt(2.)
        self.inner_factor = 2*ndtr(self.inner/self.s)-1
        self.outer_factor = 2*ndtr(self.radius/self.s)-1
        self.table_cdf_values = 0

    def bounds(self, positions):
        obs = self.obs
        d = np.linalg.norm(positions[:,None]-obs.human_segment_end[None],axis=3)
        s, R, a = self.s[None], self.radius[None], self.inner[None]
        lo = (ndtr((a-d)/s)-ndtr((-a-d)/s))*self.inner_factor[None]
        hi = (ndtr((R-d)/s)-ndtr((-R-d)/s))*self.outer_factor[None]
        lo = np.maximum(0.,lo-1e-12); hi = np.minimum(1.,hi+1e-12)
        far = d-R >= 8*s
        lo = np.where(far,ndtr(-8.),lo); hi = np.where(far,ndtr(-8.),hi)
        deterministic = self.variance[None]<1e-9
        lo = np.where(deterministic,d<=R,lo); hi = np.where(deterministic,d<=R,hi)
        existence = obs.human_existence
        if self.cfg.existence_override is not None:
            existence = np.full_like(existence,self.cfg.existence_override)
        assert np.all((existence>=0)&(existence<=1))
        def aggregate(q):
            return -np.log1p(-np.clip(q*existence[None,:,None],0.,1.-1e-12)).sum(axis=1)
        return aggregate(lo),aggregate(hi)
