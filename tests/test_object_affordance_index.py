import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from data.affordance_alignment import align_masks, canonical_indices
from data.multi_affordance_audit import audit_manifest
from data.object_affordance_index import load_index, load_object_arrays, relative_path, read_point_source


def fixture(root, split='train', oid='mesh-a', offset=0):
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.2, 0.2, 1]], dtype=float) + offset
    masks = np.array([[1, 0], [0, 1], [0, 0], [1, 1], [0.2, 0.8]])
    np.save(root / f'{oid}.npy', np.column_stack([points, masks]))
    for name in ('grasp', 'lift'):
        Image.new('RGB', (12, 10), color=(30, 60, 90)).save(root / f'{oid}-{name}.png')
    row = {'object_id': oid, 'category': 'Mug', 'setting': 'Seen', 'split': split,
           'identity_verified': True, 'identity_evidence': 'synthetic verified pairing fixture',
           'point_source': {'path': f'{oid}.npy', 'format': 'piadv2_npy'},
           'mask_columns': {'grasp': 0, 'lift': 1},
           'images': {name: [f'{oid}-{name}.png'] for name in ('grasp', 'lift')}}
    return {'version': 1, 'dataset': 'piadv2', 'affordance_vocabulary': ['grasp', 'lift'], 'records': [row]}, points, masks


class ObjectIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest, self.points, self.masks = fixture(self.root)
        self.index_path = self.root / 'index.json'

    def build(self, **kwargs):
        return audit_manifest(self.manifest, self.root, 'train', 'Seen', self.root / 'audit',
                              index_path=self.index_path, **kwargs)

    def test_roundtrip_and_checksum(self):
        summary = self.build()
        self.assertTrue(summary['relation_ready'])
        index = load_index(self.index_path)
        points, masks = load_object_arrays(self.index_path, index['objects'][0])
        np.testing.assert_allclose(points, self.points)
        np.testing.assert_allclose(masks, self.masks.T)
        with (self.root / index['objects'][0]['artifact']).open('ab') as stream:
            stream.write(b'tampered')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_object_arrays(self.index_path, index['objects'][0])

    def test_unverified_identity_rejected(self):
        self.manifest['records'][0]['identity_verified'] = False
        with self.assertRaisesRegex(ValueError, 'No accepted'):
            self.build()
        self.assertFalse(json.loads((self.root / 'audit/summary.json').read_text())['relation_ready'])

    def test_leakage_checked_across_entire_manifest(self):
        test = copy.deepcopy(self.manifest['records'][0])
        test['split'] = 'test'
        self.manifest['records'].append(test)
        with self.assertRaisesRegex(ValueError, 'No accepted'):
            self.build()
        leaks = json.loads((self.root / 'audit/split_leakage.json').read_text())
        self.assertEqual(len(leaks), 1)

    def test_unknown_mask_semantics_rejected(self):
        self.manifest['records'][0]['mask_columns'] = {'grasp': 0}
        with self.assertRaisesRegex(ValueError, 'No accepted'):
            self.build()

    def test_minimum_coverage_gate(self):
        self.assertFalse(self.build(min_multi_objects=2)['relation_ready'])

    def test_piad_manifest_content_identity_checked(self):
        row = self.manifest['records'][0]
        self.manifest['dataset'] = 'piad'
        path = self.root / 'piad.txt'
        path.write_text('\n'.join('mesh-a Mug ' + ' '.join(map(str, values)) for values in np.column_stack([self.points, self.masks])))
        row['point_source'] = {'path': 'piad.txt', 'format': 'piad_txt'}
        self.assertTrue(self.build()['relation_ready'])
        row['object_id'] = 'wrong-id'
        with self.assertRaisesRegex(ValueError, 'No accepted'):
            self.build()

    def test_reorder_preserves_point_mask_alignment(self):
        permutation = [3, 1, 4, 0, 2]
        aligned, ids, weights, quality = align_masks(self.points, self.points[permutation], self.masks[permutation])
        self.assertEqual(quality['status'], 'reordered')
        np.testing.assert_allclose(aligned, self.masks)
        np.testing.assert_allclose((self.masks[permutation][ids] * weights[:, :, None]).sum(1), aligned)

    def test_interpolation_is_opt_in_and_bidirectional(self):
        with self.assertRaisesRegex(ValueError, 'explicit'):
            align_masks(self.points, self.points + .01, self.masks)
        aligned, _, _, quality = align_masks(self.points, self.points + .01, self.masks, .1)
        self.assertEqual(quality['status'], 'interpolated')
        self.assertTrue(np.isfinite(aligned).all())
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            align_masks(self.points[:1], self.points, self.masks, .1)

    def test_conflicting_duplicate_coordinates_rejected(self):
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            align_masks(self.points, np.zeros((2, 3)), np.array([[0], [1]]), 5)

    def test_interpolation_artifacts_replay(self):
        row = copy.deepcopy(self.manifest['records'][0])
        row['point_source']['path'] = 'shifted.npy'
        row['images'] = {'lift': row['images']['lift']}
        self.manifest['records'][0]['images'].pop('lift')
        np.save(self.root / 'shifted.npy', np.column_stack([self.points + .01, self.masks]))
        self.manifest['records'].append(row)
        self.build(max_distance=.1, canonical_count=3)
        obj = load_index(self.index_path)['objects'][0]
        self.assertEqual(obj['num_points'], 3)
        with np.load(self.root / obj['artifact']) as arrays:
            replay = (self.masks[arrays['indices_1'], 1] * arrays['weights_1']).sum(1)
            np.testing.assert_allclose(arrays['masks'][1], replay, atol=1e-6)

    def test_windows_paths_and_escape_rejection(self):
        self.assertEqual(relative_path('a\\b.npy'), 'a/b.npy')
        for value in ('../a.npy', 'C:\\data\\a.npy', '/a.npy'):
            with self.assertRaises(ValueError):
                relative_path(value)

    def test_piad_string_metadata(self):
        (self.root / 'source.txt').write_text('uuid Mug 0 0 0 1 0\nuuid Mug 1 0 0 0 1\n')
        points, masks = read_point_source(self.root, {'path': 'source.txt', 'format': 'piad_txt'})
        self.assertEqual(points.shape, (2, 3))
        np.testing.assert_equal(masks, [[1, 0], [0, 1]])

    def test_deterministic_canonical_sampling(self):
        np.testing.assert_equal(canonical_indices(self.points, 3), canonical_indices(self.points, 3))
        self.assertEqual(len(np.unique(canonical_indices(self.points, 3))), 3)

    def test_corrupt_nonnumeric_npy_rejected(self):
        np.save(self.root / 'bad.npy', np.array([['x', 'y', 'z', 'mask']]))
        with self.assertRaisesRegex(ValueError, 'real numeric'):
            read_point_source(self.root, {'path': 'bad.npy', 'format': 'piadv2_npy'})


if __name__ == '__main__':
    unittest.main()
