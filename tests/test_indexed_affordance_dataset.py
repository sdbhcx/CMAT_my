import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.multi_affordance_audit import audit_manifest
from data.indexed_affordance_dataset import ObjectAffordanceDataset
from data.multi_affordance_dataset import multi_affordance_collate, get_fbd_dataloader
from data.multi_affordance_collate import flatten_object_batch
from tests.test_object_affordance_index import fixture


class IndexedDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        manifest, _, self.masks = fixture(self.root)
        other, _, _ = fixture(self.root, split='test', oid='mesh-b')
        manifest['records'] += other['records']
        self.index_path = self.root / 'train.json'
        for split in ('train', 'test'):
            audit_manifest(manifest, self.root, split, 'Seen', self.root / 'audit' / split,
                           index_path=self.root / f'{split}.json')

    def dataset(self, **kwargs):
        return ObjectAffordanceDataset(self.index_path, self.root, num_points=3, image_size=(8, 10), **kwargs)

    def test_all_masks_share_indices(self):
        sample = self.dataset()[0]
        np.testing.assert_allclose(sample['masks'].numpy(), self.masks.T[:, sample['sample_indices']])
        self.assertEqual(sample['images'].shape, (2, 3, 8, 10))
        self.assertEqual(sample['category_id'], 0)

    def test_train_seed_epoch_and_eval_determinism(self):
        first, second = self.dataset(training=True), self.dataset(training=True)
        for epoch in (0, 1, 9):
            first.set_epoch(epoch)
            second.set_epoch(epoch)
            torch.testing.assert_close(first[0]['points'], second[0]['points'])
        before = first[0]['points']
        first.set_epoch(10)
        self.assertFalse(torch.equal(before, first[0]['points']))
        evaluation = self.dataset()
        before = evaluation[0]
        evaluation.set_epoch(30)
        for key in ('points', 'images', 'masks', 'affordance_ids'):
            torch.testing.assert_close(before[key], evaluation[0][key])

    def test_eval_chunks_cover_all_affordances_with_same_points(self):
        dataset = self.dataset(max_affordances=1)
        self.assertEqual(len(dataset), 2)
        torch.testing.assert_close(dataset[0]['points'], dataset[1]['points'])
        self.assertEqual([dataset[i]['affordance_ids'].item() for i in range(2)], [0, 1])

    def test_las_flatten_preserves_fixed_sample(self):
        sample = self.dataset()[0]
        batch = multi_affordance_collate([sample])
        flat = flatten_object_batch(batch)
        torch.testing.assert_close(flat['points'][0], flat['points'][1])
        torch.testing.assert_close(flat['gt_mask'].squeeze(-1), sample['masks'])
        torch.testing.assert_close(flat['image'], sample['images'])

    def test_worker_count_does_not_change_samples(self):
        dataset = self.dataset(training=True)
        dataset.set_epoch(7)
        a = next(iter(DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=multi_affordance_collate)))
        b = next(iter(DataLoader(dataset, batch_size=1, num_workers=2, collate_fn=multi_affordance_collate)))
        torch.testing.assert_close(a['points'], b['points'])
        torch.testing.assert_close(a['images'], b['images'])

    def config(self):
        return {'dataset_type': 'piadv2', 'paths': {'data_root': str(self.root)},
                'data': {'index_paths': {s: str(self.root / f'{s}.json') for s in ('train', 'test')},
                         'max_affordances_per_object': 2, 'num_points': 3, 'image_size': [8, 8]},
                'training': {'batch_size_objects': 1, 'num_workers': 0}, 'loss': {}}

    def test_factory_and_las_compatibility(self):
        config = self.config()
        loader, sampler = get_fbd_dataloader(config, 'train')
        self.assertIsNone(sampler)
        self.assertEqual(next(iter(loader))['masks'].shape, (1, 2, 3))
        config['data']['object_las_compat'] = True
        loader, _ = get_fbd_dataloader(config, 'test')
        self.assertEqual(next(iter(loader))['gt_mask'].shape, (2, 3, 1))

    def test_factory_refuses_wrong_split(self):
        config = self.config()
        config['data']['index_paths']['test'] = config['data']['index_paths']['train']
        with self.assertRaisesRegex(ValueError, 'split'):
            get_fbd_dataloader(config, 'test')

    def test_overlap_selection_includes_high_and_low_pairs(self):
        dataset = self.dataset(training=True)
        obj = {'affordances': [{'affordance_id': i} for i in range(6)],
               'overlap_pairs': [{'affordance_ids': [0, 1], 'iou': .9},
                                 {'affordance_ids': [2, 3], 'iou': .1}]}
        self.assertEqual(dataset._select_affordances(obj, np.random.default_rng(42)), [0, 1, 2, 3])

    def test_indexed_batch_reaches_stage_a_backward(self):
        from models.relational_affordance_model import RelationalAffordanceHead
        from losses.functional_basis_loss import FunctionalBasisLoss
        batch = multi_affordance_collate([self.dataset()[0]])
        # Lightweight encoded features exercise the actual A>1 data/loss contract.
        head = RelationalAffordanceHead(point_dim=3, query_dim=3, num_basis=4,
                                       basis_hidden_dim=8, selector_hidden_dim=8)
        queries = batch['images'].mean(dim=(-1, -2))
        outputs = head(batch['points'], queries, batch['valid'])
        loss, _ = FunctionalBasisLoss()(outputs, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))


if __name__ == '__main__':
    unittest.main()
