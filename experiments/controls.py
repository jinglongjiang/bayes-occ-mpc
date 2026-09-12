"""Cached rectangle control, unchanged from the audited comparison."""
import numpy as np
from scipy.special import ndtr
from nav.risk import CompiledDiscRisk, ncx2, exact_hazard
from nav.planner import MPCPlanner


class DirectDiscRisk(CompiledDiscRisk):
    """Experimental T: original nodes, direct cubic point value, no intervals."""
    def conditional(self, positions):
        obs = self.obs
        d = np.linalg.norm(positions[:, None] - obs.human_segment_end[None], axis=3)
        z = (d-self.radius[None])/self.s[None]
        index = (z+8.)/self.step
        left = np.clip(np.floor(index).astype(np.int64), 0, len(self.grid)-2)
        t = np.clip(index-left, 0., 1.)
        c = self.component_map[None]
        qleft, qright = self.table[c,left], self.table[c,left+1]
        gleft, gright = self.derivative[c,left], self.derivative[c,left+1]
        t2, t3 = t*t, t*t*t
        q = ((2*t3-3*t2+1)*qleft + (t3-2*t2+t)*self.step*gleft
             + (-2*t3+3*t2)*qright + (t3-t2)*self.step*gright)
        q = np.clip(q, 0., 1.)
        # The positive far tail and deterministic convention match the reference.
        far = d-self.radius[None] >= 8.*np.sqrt(self.variance)[None]
        deterministic = np.broadcast_to(self.variance[None] < 1e-9, d.shape)
        outside = (z < -8.) & ~far & ~deterministic
        self.outside_values = getattr(self, 'outside_values', 0) + int(outside.sum())
        if outside.any():
            threshold = np.broadcast_to(self.radius[None]**2 / np.maximum(
                self.variance[None], 1e-9), d.shape)[outside]
            noncentral = d**2 / np.maximum(self.variance[None], 1e-9)
            q[outside] = ncx2.cdf(threshold, 2., noncentral[outside])
        q = np.where(far, ndtr(-8.), q)
        return np.where(deterministic, d <= self.radius[None], q)

    def hazard(self, positions):
        q = self.conditional(positions)
        existence = self.obs.human_existence
        if self.cfg.existence_override is not None:
            existence = np.full_like(existence, self.cfg.existence_override)
        return -np.log1p(-np.clip(q*existence[None,:,None], 0., 1.-1e-12)).sum(axis=1)


class DirectMPC(MPCPlanner):
    """Only T's evaluator changes; the production plan/score loop is inherited."""
    def __init__(self, cfg):
        super().__init__(cfg, envelope=DirectDiscRisk)
        self._point_table = None

    def _build_envelope(self, obs):
        self._point_table = super()._build_envelope(obs)
        return self._point_table

    def _belief_collision_hazard(self, controls, obs, positions=None):
        if self._point_table is None:
            return super()._belief_collision_hazard(controls, obs, positions)
        return self._point_table.hazard(positions)

    def _evaluate_batch(self, *args):
        compiled = self._compiled
        self._compiled = None
        try:
            # In this deliberately approximate control the inherited mask means
            # evaluated under T's rule, NOT reference-exact or certified.
            return super()._evaluate_batch(*args)
        finally:
            self._compiled = compiled


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
