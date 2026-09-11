import unittest

import numpy as np

from analyze_modern_main import block_sign_flip, holm, metrics, paired


def record(case, success, collision, seed=0):
    return dict(task=dict(scene=["scene"],case=case,seed=seed,occluded=True),
                result=dict(success_without_overlap=success,collision_union=collision,
                            timeout=int(not success and not collision),nav_time=10.))


class StatisticsTests(unittest.TestCase):
    def test_failure_does_not_get_fast_arrival_credit(self):
        np.testing.assert_array_equal(metrics(record(1,0,1)["result"]), [0,1,0,25])

    def test_holm_preserves_input_order(self):
        np.testing.assert_allclose(holm([.2,.01,.04]), [.2,.03,.08])

    def test_paired_events_and_sign(self):
        a=[record(i,1,0) for i in range(6)]
        b=[record(i,0,1) for i in range(6)]
        result=paired(a,b,100)
        self.assertEqual(result["events"]["SR"]["p_exact"], .03125)
        np.testing.assert_array_equal(result["left_minus_right"], [1,-1,0,-15])

    def test_repeated_seeds_are_not_independent_trials(self):
        a=[record(i,1,0,s) for i in range(4) for s in range(3)]
        b=[record(i,0,1,s) for i in range(4) for s in range(3)]
        result=paired(a,b,100)
        self.assertEqual(result["layouts"],4)
        self.assertIsNone(result["events"]["SR"]["p_exact"])
        self.assertEqual(result["events"]["SR"]["p_block_sign_flip"],.125)

    def test_scene_variants_share_environment_seed(self):
        a=[record(i,1,0) for i in (1,2,1,2)]
        b=[record(i,0,1) for i in (1,2,1,2)]
        for rows in (a,b):
            for r in rows[2:]:
                r["task"]["scene"]=["second_scene"]
        result=paired(a,b,100)
        self.assertEqual(result["layouts"],4)
        self.assertEqual(result["seed_blocks"],2)
        self.assertEqual(result["events"]["SR"]["p_block_sign_flip"],.5)
        self.assertIsNone(result["events"]["SR"]["p_exact"])

    def test_zero_difference_sign_flip(self):
        self.assertEqual(block_sign_flip([0,0,0]),1.)

    def test_duplicate_key_is_rejected(self):
        a=[record(1,1,0),record(1,1,0)]
        with self.assertRaises(ValueError):
            paired(a,a)


if __name__ == "__main__":
    unittest.main()
