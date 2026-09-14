import unittest
import numpy as np
from experiments.switching_validation import (normalize_update,sequence,ct_path,
    probability_scores,filtered,Bank)


class SwitchingTests(unittest.TestCase):
    def test_bayes_odds(self):
        p,e=normalize_update(np.array([.2,.8]),np.log([.6,.3]))
        np.testing.assert_allclose(p,[1/3,2/3])
        self.assertAlmostEqual(e,np.log(.36))

    def test_identical_likelihood_preserves_odds(self):
        p,_=normalize_update(np.array([.01,.99]),np.array([-10000.,-10000.]))
        np.testing.assert_allclose(p,[.01,.99])

    def test_future_does_not_change_prefix(self):
        t=np.arange(0,5,.1); p=np.c_[t,t*t]
        cutoff=t[23]
        a=sequence(t,p,cutoff)
        p[t>cutoff]=100000
        b=sequence(t,p,cutoff)
        for key in a: np.testing.assert_array_equal(a[key],b[key])
        self.assertTrue((a['t']<=2.3+1e-8).all())

    def test_constant_velocity(self):
        t=np.arange(0,5,.1); p=np.c_[t,2*t]
        s=sequence(t,p,t[-1])
        np.testing.assert_allclose(s['e'],0,atol=1e-13)
        np.testing.assert_allclose(s['r'],0,atol=1e-13)
        np.testing.assert_allclose(ct_path(np.array([1,2]),np.array([1,2]),np.array([1,2])),[[1,2],[2,4]])

    def test_mode_split_invariance(self):
        modes=np.zeros((2,1,4,2)); y=np.ones((2,4,2)); robot=np.zeros_like(y)
        a=probability_scores(modes,np.ones((2,1)),y,robot,np.ones(4)*.1)
        b=probability_scores(np.repeat(modes,2,axis=1),np.tile([.3,.7],(2,1)),y,robot,np.ones(4)*.1)
        for x,z in zip(a,b): np.testing.assert_allclose(x,z)

    def test_learnable_mode_positive_control(self):
        rng=np.random.default_rng(51)
        e=np.r_[rng.normal([-1,0],.03,(100,2)),rng.normal([1,0],.03,(100,2))]
        seqs={'x':dict(t=np.arange(200)*.25,e=e)}
        bank=Bank(2,11).fit(seqs,['x']); bank.residual_reference=.1
        p,_,_=filtered(bank,seqs['x'])
        assigned=bank.model.means_[p.argmax(1),0]>0
        self.assertGreater(np.mean(assigned==np.r_[np.zeros(100),np.ones(100)]),.98)


if __name__=='__main__': unittest.main()
