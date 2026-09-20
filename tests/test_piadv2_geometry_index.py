import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from data.indexed_affordance_dataset import ObjectAffordanceDataset
from data.object_affordance_index import load_index, validate_index, write_json
from tools.build_piadv2_geometry_index import build_indexes
from tools.verify_piadv2_object_groups import coordinate_hash, verify_group


class GeometryIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.groups = []
        for split, offset in [('train', 0), ('test', .25)]:
            rows, pools, image_paths = [], {}, []
            xyz = np.column_stack([np.arange(2048), np.zeros(2048), np.ones(2048)]) + offset
            for aff, stride in [('grasp', 2), ('lift', 3)]:
                point_path = f'Seen/Point/{split}/Mug/3D_Aff/{aff}/point.npy'
                image_path = f'Seen/Img/{split}/Mug/PIAD/{aff}/cue.png'
                for path in (point_path, image_path):
                    (self.root / path).parent.mkdir(parents=True, exist_ok=True)
                np.save(self.root / point_path, np.column_stack([xyz, (np.arange(2048) % stride == 0).astype(float)]))
                Image.new('RGB', (8, 8), (20, 50, 80)).save(self.root / image_path)
                image_paths.append('Data/' + image_path)
                pools[('Seen', split, 'Mug', aff)] = {image_path}
                rows.append({'path': point_path, 'category': 'Mug', 'source_dataset': '3D_Aff',
                             'setting': 'Seen', 'split': split, 'mask_semantics': [aff],
                             'ordered_coordinate_hash': coordinate_hash(xyz)})
            (self.root / 'Seen' / f'Img_{split}.txt').write_text('\n'.join(image_paths))
            self.groups.append(verify_group(self.root, coordinate_hash(xyz, True), rows, pools))
        self.verification = self.root / 'audit'
        write_json(self.verification / 'verified_groups.json', self.groups)
        self.output = self.root / 'index'

    def build(self):
        return build_indexes(self.root, self.verification, self.output, setting='Seen', workers=1)

    def test_v2_roundtrip_loader_and_cue_pool(self):
        report = self.build()
        self.assertTrue(report['partitions']['train']['relation_ready'])
        index = load_index(self.output / 'train.json')
        self.assertEqual(index['version'], 2)
        self.assertFalse(index['objects'][0]['original_mesh_id_recovered'])
        self.assertNotIn('images', index['objects'][0]['affordances'][0])
        dataset = ObjectAffordanceDataset(self.output / 'train.json', self.root, num_points=32,
                                          training=True, min_affordances=2, image_size=(8, 8))
        self.assertEqual(dataset[0]['masks'].shape, (2, 32))
        self.assertEqual(dataset[0]['images'].shape, (2, 3, 8, 8))

    def test_cue_category_mismatch_rejected(self):
        self.build()
        index = load_index(self.output / 'train.json')
        index['image_pools']['Mug/grasp']['category'] = 'Bottle'
        with self.assertRaisesRegex(ValueError, 'category'):
            validate_index(index)

    def test_overlap_does_not_silently_change_splits(self):
        group = self.groups[0]
        part = copy.deepcopy(group['partitions'][0])
        part['split'] = 'test'
        group['partitions'].append(part)
        write_json(self.verification / 'verified_groups.json', self.groups)
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            self.build()


if __name__ == '__main__':
    unittest.main()
