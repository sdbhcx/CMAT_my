from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.verify_piadv2_object_groups import coordinate_hash, summarize, verify_group


class VerifyPIADV2Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.points = np.column_stack([np.arange(2048), np.zeros(2048), np.ones(2048)]).astype(float)
        self.mask = (np.arange(2048) % 2).astype(float)
        self.fingerprint = coordinate_hash(self.points, True)
        self.pools = {('Seen', split, 'Mug', aff): {'category_prompt.jpg'}
                      for split in ('train', 'test') for aff in ('grasp', 'lift')}

    def row(self, affordance, split='train', name='a', mask=None, permutation=None):
        points = self.points if permutation is None else self.points[permutation]
        values = self.mask if mask is None else mask
        if permutation is not None:
            values = values[permutation]
        path = f'Seen/Point/{split}/Mug/3D_Aff/{affordance}/{name}.npy'
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, np.column_stack([points, values]))
        return {'path': path, 'category': 'Mug', 'source_dataset': '3D_Aff',
                'setting': 'Seen', 'split': split, 'mask_semantics': [affordance],
                'ordered_coordinate_hash': coordinate_hash(points)}

    def test_category_affordance_cues_do_not_require_instance_identity(self):
        rows = [self.row('grasp'), self.row('lift', mask=1-self.mask)]
        group = verify_group(self.root, self.fingerprint, rows, self.pools)
        self.assertFalse(group['issues'])
        self.assertFalse(group['original_mesh_id_recovered'])
        part = group['partitions'][0]
        self.assertTrue(part['eligible_before_split_filter'])
        self.assertTrue(part['same_point_order'])
        self.assertFalse(part['mask_pairs'][0]['identical_masks'])

    def test_conflicting_same_affordance_masks_rejected(self):
        rows = [self.row('grasp'), self.row('grasp', name='b', mask=1-self.mask)]
        group = verify_group(self.root, self.fingerprint, rows, self.pools)
        self.assertEqual(group['partitions'][0]['conflicting_duplicate_affordances'], ['grasp'])
        self.assertFalse(group['partitions'][0]['eligible_before_split_filter'])

    def test_permutation_aligns_masks(self):
        rows = [self.row('grasp'), self.row('lift', permutation=np.arange(2048)[::-1])]
        group = verify_group(self.root, self.fingerprint, rows, self.pools)
        part = group['partitions'][0]
        self.assertFalse(part['same_point_order'])
        self.assertTrue(part['mask_pairs'][0]['identical_masks'])

    def test_split_exclusion_is_report_only(self):
        rows = [self.row('grasp'), self.row('lift'), self.row('grasp', split='test')]
        group = verify_group(self.root, self.fingerprint, rows, self.pools)
        summary, exclusions = summarize([group], [])
        self.assertEqual(summary['same_geometry_split_overlaps'], {'Seen/test-train': 1})
        self.assertEqual(exclusions['Seen'], [group['geometry_id']])
        self.assertEqual(summary['coverage_by_partition']['Seen/train'].get('eligible_A_ge_2_after_proposed_filter', 0), 0)
        self.assertFalse(summary['original_data_or_splits_modified'])


if __name__ == '__main__':
    unittest.main()
