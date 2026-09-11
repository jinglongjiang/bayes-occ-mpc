import tempfile
import unittest
from pathlib import Path

from modern_main_table import candidates, digest, rank, task, verify_sources


class MainTableProtocolTests(unittest.TestCase):
    def test_candidates_keep_native_identity_and_horizon(self):
        rows = candidates()
        self.assertEqual(len(rows), 58)
        self.assertEqual(len({r['id'] for r in rows}), len(rows))
        for family, expected in [('bayes', 10), ('tmpc', 24), ('shmpc', 24)]:
            group = [r for r in rows if r['family'] == family]
            self.assertEqual(len(group), expected)
            self.assertEqual({r['horizon'] for r in group}, {8, 16})
            if family != 'bayes':
                self.assertTrue(all(r['arm'].startswith(family + '_repair_') for r in group))

    def test_pair_key_does_not_collapse_conditions_or_seeds(self):
        c = candidates()[0]
        scene = ('s', 5, 'circle_crossing', 4., 10.)
        keys = {task('T', c, scene, 21000, seed=s, occluded=o)['key']
                for s in range(3) for o in (True, False)}
        self.assertEqual(len(keys), 6)
        self.assertNotEqual(task('T', c, scene, 21000)['key'],
                            task('T', c, ('s2', 5, 'circle_crossing', 6., 10.), 21000)['key'])

    def test_selection_penalizes_failure_and_uses_only_supplied_rows(self):
        records = []
        for label, outcomes in [('fast_failure', [(0, 1, .5), (1, 0, 2.)]),
                                ('reliable', [(1, 0, 10.), (1, 0, 11.)])]:
            for sr, cr, t in outcomes:
                records.append(dict(task={'candidate': {'id': label}},
                    result={'success_without_overlap': sr, 'collision_union': cr, 'nav_time': t}))
        self.assertEqual(rank(records)[0], 'reliable')

    def test_digest_is_order_invariant(self):
        self.assertEqual(digest({'b': 2, 'a': 1}), digest({'a': 1, 'b': 2}))

    def test_changed_frozen_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source'
            path.touch()
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                verify_sources({str(path): 'incorrect_hash'})


if __name__ == '__main__':
    unittest.main()
