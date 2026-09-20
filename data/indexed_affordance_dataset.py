"""Audited instance-level PIAD/PIADv2 sets with shared point sampling."""
from __future__ import annotations

import hashlib
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset

from .object_affordance_index import load_index, load_object_arrays, resolve_path, get_affordance_images


class ObjectAffordanceDataset(Dataset):
    def __init__(self, index_path, data_root, num_points=2048, max_affordances=4,
                 image_size=(224, 224), training=False, augment=True, seed=42, rank=0, min_affordances=1):
        self.index_path, self.data_root = index_path, data_root
        self.index = load_index(index_path)
        if num_points < 1 or not 1 <= max_affordances <= 4:
            raise ValueError('num_points must be positive; max_affordances must be in [1, 4]')
        if not 1 <= min_affordances <= max_affordances:
            raise ValueError('min_affordances must be between 1 and max_affordances')
        self.num_points, self.max_affordances = num_points, max_affordances
        self.image_size = tuple(image_size)
        self.training, self.augment = training, augment and training
        self.seed, self.rank, self.epoch = seed, rank, 0
        self.items = []
        for i, obj in enumerate(self.index['objects']):
            count = len(obj['affordances'])
            if training:
                if count >= min_affordances:
                    self.items.append((i, None))
            else:
                # Evaluate every affordance, in stable chunks of at most Amax.
                for start in range(0, count, max_affordances):
                    self.items.append((i, list(range(start, min(count, start + max_affordances)))))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.items)

    def _rng(self, object_id):
        salt = int.from_bytes(hashlib.sha256(object_id.encode()).digest()[:8], 'little')
        return np.random.default_rng(np.random.SeedSequence([
            self.seed, self.epoch if self.training else 0, self.rank if self.training else 0, salt]))

    def _select_affordances(self, obj, rng):
        count = len(obj['affordances'])
        if count <= self.max_affordances:
            return list(range(count))
        chosen = set()
        pairs = sorted(obj.get('overlap_pairs', []), key=lambda pair: (pair['iou'], pair['affordance_ids']))
        positions = {a['affordance_id']: i for i, a in enumerate(obj['affordances'])}
        if pairs and self.max_affordances >= 2:
            # Prefer high- and low-overlap pairs when both fit the set budget.
            for pair in (pairs[-1], pairs[0]):
                candidate = chosen | {positions[aid] for aid in pair['affordance_ids']}
                if len(candidate) <= self.max_affordances:
                    chosen = candidate
        remaining = [i for i in range(count) if i not in chosen]
        chosen.update(rng.choice(remaining, self.max_affordances - len(chosen), replace=False).tolist())
        return sorted(chosen)

    def __getitem__(self, item):
        object_pos, selected = self.items[item]
        obj = self.index['objects'][object_pos]
        points, masks = load_object_arrays(self.index_path, obj)
        rng = self._rng(obj['object_id'])
        # Draw points before affordance selection so eval chunks share indices.
        if len(points) > self.num_points:
            indices = rng.choice(len(points), self.num_points, replace=False)
        elif len(points) < self.num_points:
            indices = np.concatenate([np.arange(len(points)), rng.choice(len(points), self.num_points - len(points))])
        else:
            indices = np.arange(len(points))
        points = points[indices].astype(np.float32)
        points -= points.mean(axis=0, keepdims=True)
        radius = np.linalg.norm(points, axis=1).max()
        points /= max(float(radius), 1e-8)
        if self.augment:
            angle = rng.uniform(0, 2 * np.pi)
            rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                                 [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], dtype=np.float32)
            points = points @ rotation.T
            points += rng.normal(0, 0.01, points.shape).astype(np.float32)
        if selected is None:
            selected = self._select_affordances(obj, rng)
        images, aids = [], []
        for position in selected:
            affordance = obj['affordances'][position]
            paths = get_affordance_images(self.index, obj, affordance)
            path = paths[int(rng.integers(len(paths)))] if self.training else paths[0]
            with Image.open(resolve_path(self.data_root, path)) as image:
                image = image.convert('RGB').resize((self.image_size[1], self.image_size[0]), Image.Resampling.BILINEAR)
                if self.augment:
                    image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.8, 1.2)))
                array = np.asarray(image, dtype=np.float32) / 255.0
            array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
            images.append(torch.from_numpy(array.transpose(2, 0, 1).copy()))
            aids.append(affordance['affordance_id'])
        return {'points': torch.from_numpy(points), 'images': torch.stack(images),
                'masks': torch.from_numpy(masks[selected][:, indices].copy()),
                'affordance_ids': torch.tensor(aids, dtype=torch.long),
                'valid': torch.ones(len(aids), dtype=torch.bool),
                'object_id': obj['object_id'],
                'category_id': self.index['category_vocabulary'].index(obj['category']),
                'sample_indices': torch.from_numpy(indices.copy())}
