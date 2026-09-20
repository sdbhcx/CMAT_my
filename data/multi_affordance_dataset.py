"""Object-shaped batching for functional-basis training.

Audited indexes provide true instance sets. Without an index, each legacy sample
remains an A=1 set, preserving existing pairing semantics.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class SingleAffordanceSetDataset(Dataset):
    """Adapt a legacy single-affordance dataset to the object-set contract."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        image = sample["image"]
        points = sample["points"]
        mask = sample["gt_mask"]

        if image.ndim != 3:
            raise ValueError(f"image must have shape [3, H, W], got {tuple(image.shape)}")
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError(f"points must have shape [N, 3], got {tuple(points.shape)}")
        if mask.ndim == 2 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        if mask.ndim != 1:
            raise ValueError(
                "The safe MVP requires exactly one mask column per legacy sample; "
                f"got gt_mask shape {tuple(sample['gt_mask'].shape)}. Run the data "
                "audit before enabling multi-column PIADv2 files."
            )
        if mask.shape[0] != points.shape[0]:
            raise ValueError("points and gt_mask must contain the same number of points")

        affordance_id = int(sample.get("affordance_id", -1))
        instance_id = int(sample.get("instance_id", index))
        return {
            "points": points,
            "images": image.unsqueeze(0),
            "masks": mask.unsqueeze(0),
            "affordance_ids": torch.tensor([affordance_id], dtype=torch.long),
            "valid": torch.ones(1, dtype=torch.bool),
            "object_id": str(instance_id),
            "category_id": -1,
        }


def multi_affordance_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad the affordance dimension and collate object-level samples."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    batch_size = len(batch)
    max_affordances = max(item["images"].shape[0] for item in batch)
    point_shape = batch[0]["points"].shape
    image_shape = batch[0]["images"].shape[1:]

    points = torch.stack([item["points"] for item in batch])
    images = batch[0]["images"].new_zeros(
        (batch_size, max_affordances, *image_shape)
    )
    masks = batch[0]["masks"].new_zeros(
        (batch_size, max_affordances, point_shape[0])
    )
    affordance_ids = torch.full(
        (batch_size, max_affordances), -1, dtype=torch.long
    )
    valid = torch.zeros((batch_size, max_affordances), dtype=torch.bool)

    for batch_index, item in enumerate(batch):
        if item["points"].shape != point_shape:
            raise ValueError("All point clouds in a batch must share shape [N, 3]")
        if item["images"].shape[1:] != image_shape:
            raise ValueError("All images in a batch must share shape [3, H, W]")
        affordance_count = item["images"].shape[0]
        if item["masks"].shape != (affordance_count, point_shape[0]):
            raise ValueError("Each affordance image must have one aligned point mask")

        images[batch_index, :affordance_count] = item["images"]
        masks[batch_index, :affordance_count] = item["masks"]
        affordance_ids[batch_index, :affordance_count] = item["affordance_ids"]
        valid[batch_index, :affordance_count] = item["valid"]

    return {
        "points": points,
        "images": images,
        "masks": masks,
        "affordance_ids": affordance_ids,
        "valid": valid,
        "object_ids": [item["object_id"] for item in batch],
        "category_ids": torch.tensor(
            [item.get("category_id", -1) for item in batch], dtype=torch.long
        ),
    }


def get_fbd_dataloader(
    config,
    split: str,
    rank: int = 0,
    world_size: int = 1,
):
    """Use an audited object index when configured, otherwise preserve A=1."""
    data_config = config.get('data', {})
    index_paths = data_config.get('index_paths', {})
    index_path = index_paths.get(split)
    if index_paths and not index_path:
        raise ValueError(f'Missing data.index_paths.{split}; refusing legacy fallback')
    if data_config.get('index_path') and not index_paths:
        raise ValueError('Use data.index_paths.train/test to avoid training/evaluation index reuse')
    if index_path:
        from data.indexed_affordance_dataset import ObjectAffordanceDataset
        from data.multi_affordance_collate import las_object_collate

        dataset = ObjectAffordanceDataset(
            index_path, config['paths']['data_root'],
            num_points=int(data_config.get('num_points', 2048)),
            max_affordances=int(data_config.get('max_affordances_per_object', 4)),
            image_size=data_config.get('image_size', (224, 224)),
            training=(split == 'train'), augment=data_config.get('use_augmentation', True),
            seed=int(config.get('seed', 42)), rank=rank,
            min_affordances=int(data_config.get('min_train_affordances', 1)) if split == 'train' else 1,
        )
        if len(dataset) == 0:
            raise ValueError('No objects satisfy the configured affordance count')
        if dataset.index['split'] != split:
            raise ValueError(f'Index split does not match requested {split}')
        if dataset.index['dataset'] != config.get('dataset_type', 'piadv2'):
            raise ValueError('Index dataset does not match dataset_type')
        from data.object_affordance_index import load_index
        current_ids = {obj['object_id'] for obj in dataset.index['objects']}
        for other_split, other_path in index_paths.items():
            if other_split == split:
                continue
            other = load_index(other_path)
            if any(other[key] != dataset.index[key] for key in ('affordance_vocabulary', 'category_vocabulary')):
                raise ValueError('Index vocabularies must match across splits')
            if current_ids & {obj['object_id'] for obj in other['objects']}:
                raise ValueError('Object ID leakage between configured indexes')
        relation_enabled = any(float(config.get('loss', {}).get(name, 0)) != 0
                               for name in ('relation', 'coefficient_relation'))
        if relation_enabled and not dataset.index['audit'].get('relation_ready', False):
            raise ValueError('Relation losses require a GO audit')
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=(split == 'train'), seed=int(config.get('seed', 42))) if world_size > 1 else None
        training_config = config['training']
        loader = DataLoader(
            dataset, batch_size=training_config.get('batch_size_objects', training_config.get('batch_size', 4)),
            shuffle=(split == 'train' and sampler is None), sampler=sampler,
            num_workers=training_config.get('num_workers', config.get('num_workers', 4)),
            pin_memory=config.get('hardware', {}).get('pin_memory', True),
            drop_last=(split == 'train'),
            generator=torch.Generator().manual_seed(int(config.get('seed', 42)) + rank),
            collate_fn=las_object_collate if data_config.get('object_las_compat', False) else multi_affordance_collate,
        )
        return loader, sampler
    max_affordances = int(config.get("data", {}).get("max_affordances_per_object", 1))
    if max_affordances != 1:
        raise ValueError(
            "The safe compatibility dataset currently supports "
            "max_affordances_per_object: 1 only. Build an audited object index "
            "before enabling A > 1."
        )
    dataset_type = str(config.get("dataset_type", "piadv2")).lower()
    if dataset_type == "piad":
        from data.piad_dataset import get_piad_dataloader

        base_loader = get_piad_dataloader(config, split=split)
    elif dataset_type in {"piadv2", "piad_v2", "piad2"}:
        from data.piadv2_dataset import get_dataloader

        base_loader = get_dataloader(config, split=split)
    else:
        raise ValueError("The functional-basis MVP supports PIAD and PIADv2 only")

    dataset = SingleAffordanceSetDataset(base_loader.dataset)
    distributed = world_size > 1
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=(split == "train"),
        )

    training_config = config["training"]
    batch_size = training_config.get(
        "batch_size_objects", training_config.get("batch_size", 4)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train" and sampler is None),
        sampler=sampler,
        num_workers=training_config.get(
            "num_workers", config.get("num_workers", 4)
        ),
        pin_memory=config.get("hardware", {}).get("pin_memory", True),
        drop_last=(split == "train"),
        collate_fn=multi_affordance_collate,
    )
    return dataloader, sampler
