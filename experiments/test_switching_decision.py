import unittest
from types import SimpleNamespace
import joblib
import numpy as np
from experiments.switching_navigation import Online,envbase,Direct,mixture
from experiments.switching_decision import OUT,H,source
from nav.contracts import MPCConfig
from integration.crowdnav import BayesObservationAdapter


class DecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.model=joblib.load(OUT/'deployment.joblib')

    def test_online_offline_prefix_equivalence(self):
        engine=Online(self.model,'FULL')
        t=np.arange(0,3.01,.25);p=np.c_[t,.05*t*t];r=np.zeros_like(p)
        obs=SimpleNamespace(robot_xy=np.zeros(2))
        for ti,pi in zip(t,p):
            output,count=engine.update(ti,[dict(id=7,px=pi[0],py=pi[1])],obs,[7])
        f,rot,cv,_,_,_=source.old.prefix_features(t,p,r)
        ages=np.minimum(np.array([4,3,2,1,0]),t[-1]-t[0])
        longp=(source.old.interpolate(t,p,t[-1]-ages)-p[-1])@rot
        longr=(source.old.interpolate(t,r,t[-1]-ages)-p[-1])@rot
        x=np.r_[f,longp.ravel(),longr.ravel(),ages,t[-1],(p[-1]-p[0])@rot,np.zeros(3)]
        modes=source.expert_predict(self.model['experts'],x[None],cv[None])[0]@rot.T+p[-1]
        prob,_,_=source.filtered(self.model['bank'],source.sequence(t,p,t[-1]))
        np.testing.assert_allclose(output[0][0][:,[1,3,7,15]],modes,atol=1e-12)
        np.testing.assert_allclose(output[0][1],prob[-1],atol=1e-12)

    def test_missing_is_not_an_observation(self):
        engine=Online(self.model,'FULL');obs=SimpleNamespace(robot_xy=np.zeros(2))
        for t in np.arange(0,2,.25):engine.update(t,[dict(id=7,px=t,py=0)],obs,[7])
        length=len(engine.past[7])
        result,n=engine.update(2.5,[],obs,[7])
        self.assertEqual(result,{})
        self.assertEqual(len(engine.past[7]),length)
        result,n=engine.update(3.,[dict(id=7,px=3.,py=0)],obs,[7])
        self.assertEqual(result,{})

    def test_direct_risk_and_mode_invariance(self):
        cfg=MPCConfig(horizon=16)
        env=envbase.environment(dict(case=0,people=5,scene='circle_crossing'))
        obs=BayesObservationAdapter(cfg).read(env)
        self.assertGreater(len(obs.entities),0)
        centers=obs.human_segment_end[0:1]
        var=np.ones(16)*.1
        one={0:(centers,np.ones(1),var)}
        split={0:(np.repeat(centers,2,axis=0),np.array([.3,.7]),var)}
        rng=np.random.default_rng(11)
        positions=rng.normal(size=(40,16,2))
        a=-np.expm1(-Direct(cfg,obs,one).exact(positions))
        b=-np.expm1(-Direct(cfg,obs,split).exact(positions))
        exact=-np.expm1(-mixture.MixtureEnvelope(cfg,obs,one).exact(positions))
        np.testing.assert_allclose(a,b,atol=1e-10)
        self.assertLess(np.max(np.abs(a-exact)),1e-4)


if __name__=='__main__':unittest.main()
