import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from data.raw_affordance_audit import audit_raw_dataset


class RawAuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'Seen').mkdir()

    def test_v2_geometry_is_candidate_not_identity(self):
        paths = []
        for affordance, rows in [('grasp', [[0, 0, 0, 1], [1, 0, 0, 0]]),
                                 ('lift', [[1, 0, 0, 1], [0, 0, 0, 0]])]:
            path = f'Seen/Point/train/Mug/3D_Aff/{affordance}/1.npy'
            (self.root / path).parent.mkdir(parents=True)
            np.save(self.root / path, rows)
            paths.append('Data/' + path)
        (self.root / 'Seen/Point_train.txt').write_text('\n'.join(paths))
        summary = audit_raw_dataset(self.root, 'piadv2', self.root / 'report')
        self.assertEqual(summary['coordinate_match_groups'], 1)
        self.assertEqual(summary['candidate_groups_with_at_least_k_nonempty_affordances']['2'], 1)
        self.assertFalse(summary['relation_ready'])
        groups = json.loads((self.root / 'report/coordinate_matches.json').read_text())
        self.assertEqual(groups[0]['point_order'], 'reordered')
        self.assertFalse(groups[0]['identity_verified'])
        single = audit_raw_dataset(self.root, 'piadv2', self.root / 'single', workers=1)
        self.assertEqual(summary, single)
        self.assertEqual((self.root / 'report/point_inventory.json').read_text(),
                         (self.root / 'single/point_inventory.json').read_text())

    def test_piad_uuid_and_split_overlap(self):
        for split in ('train', 'test'):
            path = f'Seen/Point/{split}/Mug/1.txt'
            (self.root / path).parent.mkdir(parents=True)
            (self.root / path).write_text('mesh-uuid Mug 0 0 0 ' + ' '.join(['1'] + ['0'] * 16) + '\n')
            (self.root / f'Seen/Point_{split}.txt').write_text('Data/' + path)
        summary = audit_raw_dataset(self.root, 'piad', self.root / 'report')
        self.assertEqual(summary['recovered_original_object_ids'], 1)
        self.assertEqual(summary['object_id_partition_overlaps'], 1)
        self.assertFalse(summary['relation_ready'])
        stats = json.loads((self.root / 'report/object_summary.json').read_text())
        self.assertEqual(stats['within_setting_split_overlap_groups'], {'Seen': 1})
        self.assertFalse(stats['usable_for_training'])

    def test_missing_source_is_reported(self):
        (self.root / 'Seen/Point_train.txt').write_text('Data/Seen/Point/train/Mug/source/grasp/missing.npy')
        summary = audit_raw_dataset(self.root, 'piadv2', self.root / 'report', max_files=1)
        self.assertEqual(summary['anomaly_count'], 1)
        self.assertTrue(summary['sampled'])


if __name__ == '__main__':
    unittest.main()
