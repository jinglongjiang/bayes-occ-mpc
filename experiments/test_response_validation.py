"""Controls that distinguish a failed hypothesis from a broken diagnostic."""
import unittest
import numpy as np
from experiments.peroi_response_validation import (PhysicalScaler, prefix_features,
    conditional_features, regressor, energy)
from experiments.response_data_audit import structural_checks


class ResponseTests(unittest.TestCase):
    def test_static_robot_scaling(self):
        x = np.arange(100)[:,None]*1e-16
        scaler = PhysicalScaler().fit(x)
        self.assertGreaterEqual(scaler.scale_[0],.01)
        self.assertLess(abs(scaler.transform([[.1]])[0,0]),11)

    def test_cv_coordinates(self):
        t = np.linspace(0,1,21)
        p = np.c_[2*t,3*t]
        r = np.zeros_like(p)
        f,rot,cv,ca,dist,closing = prefix_features(t,p,r)
        self.assertTrue(np.allclose(cv[:,0],np.array([.5,1,2])*np.sqrt(13)))
        self.assertTrue(np.allclose(cv[:,1],0))
        self.assertTrue(np.allclose(ca,cv))
        self.assertTrue(np.allclose(rot.T@rot,np.eye(2)))

    def test_known_label_positive_control(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(900,5))
        z = rng.integers(0,3,900)
        y = x[:,0:1]+np.array([-2,0,2])[z,None]
        train,test = np.arange(600),np.arange(600,900)
        pooled = regressor('ridge',1).fit(x[train],y[train])
        conditional = regressor('ridge',1).fit(conditional_features(x[train],z[train]),y[train])
        a = np.mean((pooled.predict(x[test])-y[test])**2)
        b = np.mean((conditional.predict(conditional_features(x[test],z[test]))-y[test])**2)
        self.assertLess(b,a*.01)

    def test_finite_energy_degeneracy(self):
        modes = np.array([[[[1.,0.]],[[1.,0.]],[[1.,0.]]]])
        self.assertTrue(np.allclose(energy(modes,np.array([[.2,.3,.5]]),np.zeros((1,1,2))),1))

    def test_exact_structure(self):
        results = structural_checks()
        self.assertLess(results['group_additive_expected_cost_max_error'],1e-12)
        self.assertEqual(results['sensor_count_confounding_max_error'],0)


if __name__=='__main__':
    unittest.main()
