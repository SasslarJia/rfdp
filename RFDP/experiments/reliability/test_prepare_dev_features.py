"""Model-free contract checks; run with Python's standard library only."""
import unittest

import prepare_dev_features as prep


class SplitTests(unittest.TestCase):
    def records(self):
        return [prep.record_from_name(f'{pid:04d}_c{camera}s1_{frame:06d}_00.jpg')
                for pid in range(1, 7) for camera, frame in ((1, 1), (1, 2), (2, 1))]

    def split(self, rows):
        return prep.build_split(rows, expected_identities=6, expected_images=18, fit_count=4)

    def test_deterministic_identity_split_and_full_image_coverage(self):
        rows, split = self.split(self.records())
        rows_reversed, again = self.split(list(reversed(self.records())))
        self.assertEqual(rows, rows_reversed)
        self.assertEqual(split, again)
        fit, val = split['partitions']['fit'], split['partitions']['val']
        self.assertFalse(set(fit['identity_ids']) & set(val['identity_ids']))
        self.assertEqual((fit['query_count'], fit['gallery_count']), (4, 8))
        self.assertEqual((val['query_count'], val['gallery_count']), (2, 4))
        used = []
        for info in (fit, val):
            used += info['query_indices'] + info['gallery_indices']
            self.assertTrue(all(n > 0 for n in info['query_cross_camera_positive_counts']))
            for index in info['query_indices']:
                self.assertTrue(rows[index]['name'].endswith('c1s1_000001_00.jpg'))
        self.assertEqual(sorted(used), list(range(18)))

    def test_duplicate_image_rejected(self):
        rows = self.records()
        rows[-1] = rows[0]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.split(rows)

    def test_multiple_camera_queries_and_singleton_fallback(self):
        camera_counts = {1: {1: 2, 2: 2, 3: 1}, 2: {1: 1, 2: 1, 3: 1}, 3: {2: 2, 3: 1}}
        rows = [prep.record_from_name(f'{pid:04d}_c{camera}s1_{frame:06d}_00.jpg')
                for pid, cameras in camera_counts.items()
                for camera, count in cameras.items() for frame in range(1, count + 1)]
        ordered, split = prep.build_split(rows, expected_identities=3, expected_images=11, fit_count=2)
        query_indices = [i for part in split['partitions'].values() for i in part['query_indices']]
        queries = {(ordered[i]['pid'], ordered[i]['camera'], ordered[i]['name']) for i in query_indices}
        self.assertEqual({(pid, camera) for pid, camera, name in queries}, {(1, 1), (1, 2), (2, 3), (3, 2)})
        self.assertTrue(all(name.endswith('_000001_00.jpg') for pid, camera, name in queries))
        fallback_ids = [pid for part in split['partitions'].values() for pid in part['all_singleton_camera_fallback_ids']]
        self.assertEqual(fallback_ids, [2])
        self.assertTrue(all(n > 0 for part in split['partitions'].values()
                            for n in part['query_cross_camera_positive_counts']))
        used = [i for part in split['partitions'].values() for role in ('query', 'gallery')
                for i in part[role + '_indices']]
        self.assertEqual(sorted(used), list(range(11)))

    def test_single_camera_identity_rejected(self):
        rows = self.records()
        rows[2] = prep.record_from_name('0001_c1s1_000003_00.jpg')
        with self.assertRaisesRegex(ValueError, 'cross-camera'):
            self.split(rows)

    def test_filename_metadata_mismatch_rejected(self):
        rows = self.records()
        rows[0]['camera'] = 6
        with self.assertRaisesRegex(ValueError, 'metadata mismatch'):
            self.split(rows)

    def test_missing_image_rejected(self):
        with self.assertRaisesRegex(ValueError, 'training images'):
            self.split(self.records()[:-1])

    def test_training_junk_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Junk/distractor'):
            prep.record_from_name('0000_c1s1_000001_00.jpg')


if __name__ == '__main__':
    unittest.main(verbosity=2)
