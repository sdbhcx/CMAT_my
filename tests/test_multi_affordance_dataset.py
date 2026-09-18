import unittest

import torch
from torch.utils.data import Dataset

from data.multi_affordance_dataset import (
    SingleAffordanceSetDataset,
    multi_affordance_collate,
)


class _LegacyDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "image": torch.full((3, 4, 5), float(index)),
            "points": torch.randn(6, 3),
            "gt_mask": torch.tensor([[0], [1], [0], [1], [0], [1]]).float(),
            "affordance_id": index + 3,
            "instance_id": index + 10,
        }


class MultiAffordanceDatasetTest(unittest.TestCase):
    def test_legacy_adapter_and_collate_contract(self):
        dataset = SingleAffordanceSetDataset(_LegacyDataset())
        sample = dataset[0]
        self.assertEqual(sample["images"].shape, (1, 3, 4, 5))
        self.assertEqual(sample["masks"].shape, (1, 6))

        batch = multi_affordance_collate([dataset[0], dataset[1]])
        self.assertEqual(batch["points"].shape, (2, 6, 3))
        self.assertEqual(batch["images"].shape, (2, 1, 3, 4, 5))
        self.assertEqual(batch["masks"].shape, (2, 1, 6))
        self.assertTrue(batch["valid"].all())
        self.assertTrue(
            torch.equal(batch["affordance_ids"], torch.tensor([[3], [4]]))
        )

    def test_adapter_rejects_unverified_multi_column_masks(self):
        class MultiColumnDataset(_LegacyDataset):
            def __getitem__(self, index):
                sample = super().__getitem__(index)
                sample["gt_mask"] = torch.zeros(6, 2)
                return sample

        with self.assertRaisesRegex(ValueError, "exactly one mask column"):
            SingleAffordanceSetDataset(MultiColumnDataset())[0]

    def test_variable_affordance_padding(self):
        dataset = SingleAffordanceSetDataset(_LegacyDataset())
        single = dataset[0]
        double = dict(dataset[1])
        double["images"] = double["images"].repeat(2, 1, 1, 1)
        double["masks"] = double["masks"].repeat(2, 1)
        double["affordance_ids"] = torch.tensor([4, 5])
        double["valid"] = torch.tensor([True, True])

        batch = multi_affordance_collate([single, double])
        self.assertEqual(batch["images"].shape[:2], (2, 2))
        self.assertTrue(torch.equal(batch["valid"], torch.tensor([[1, 0], [1, 1]]).bool()))
        self.assertEqual(batch["affordance_ids"][0, 1].item(), -1)
        self.assertTrue(torch.equal(batch["masks"][0, 1], torch.zeros(6)))


if __name__ == "__main__":
    unittest.main()
