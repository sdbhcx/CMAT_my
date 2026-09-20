"""Deterministic coordinate alignment in original coordinate units."""
from __future__ import annotations
import numpy as np


def _points(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected finite nonempty [N, 3] coordinates')
    return values


def nearest_neighbors(target, source, k=1):
    """Bounded-memory, deterministic kNN; ties resolve by source row index."""
    target, source = _points(target), _points(source)
    k = min(k, len(source))
    all_indices, all_distances = [], []
    chunk_size = max(1, min(256, 2_000_000 // len(source)))
    for start in range(0, len(target), chunk_size):
        squared = ((target[start:start + chunk_size, None] - source[None]) ** 2).sum(-1)
        # Stable sorting also makes duplicate-coordinate behavior reproducible.
        ids = np.argsort(squared, axis=1, kind='stable')[:, :k]
        all_indices.append(ids)
        all_distances.append(np.sqrt(np.take_along_axis(squared, ids, axis=1)))
    return np.concatenate(all_indices), np.concatenate(all_distances)


def canonical_indices(points, count=2048):
    """Deterministic farthest-point sampling, with deterministic repeat padding."""
    points = _points(points)
    if count < 1:
        raise ValueError('canonical point count must be positive')
    if count >= len(points):
        return np.arange(count, dtype=np.int64) % len(points)
    selected = np.empty(count, dtype=np.int64)
    selected[0] = np.lexsort(points.T[::-1])[0]
    distances = np.full(len(points), np.inf)
    for i in range(1, count):
        distances = np.minimum(distances, ((points - points[selected[i - 1]]) ** 2).sum(1))
        distances[selected[:i]] = -1
        selected[i] = distances.argmax()
    return selected


def align_masks(canonical, source, masks, max_distance=None):
    """Return aligned masks, replayable interpolation map, and bidirectional quality.

    Exact reorder requires a bijection. Approximate alignment is disabled unless
    a caller supplies a dataset-specific maximum distance; no ICP/normalization
    is used to make unrelated shapes appear aligned.
    """
    canonical, source = _points(canonical), _points(source)
    masks = np.asarray(masks, dtype=np.float32)
    if masks.ndim == 1:
        masks = masks[:, None]
    if masks.ndim != 2 or masks.shape[0] != len(source) or not np.isfinite(masks).all():
        raise ValueError('Source masks must be finite [N, A]')
    if max_distance is not None and (not np.isfinite(max_distance) or max_distance < 0):
        raise ValueError('max_distance must be finite and nonnegative')
    if np.array_equal(canonical, source):
        ids = np.arange(len(source))[:, None]
        distances = np.zeros((len(source), 1))
        status, reverse_max = 'exact', 0.0
    else:
        # Duplicate coordinates with different labels have no unique reorder.
        unique, inverse = np.unique(source, axis=0, return_inverse=True)
        if len(unique) != len(source):
            low = np.full((len(unique), masks.shape[1]), np.inf)
            high = np.full_like(low, -np.inf)
            np.minimum.at(low, inverse, masks)
            np.maximum.at(high, inverse, masks)
            if not np.array_equal(low, high):
                raise ValueError('Duplicate coordinates carry conflicting masks')
        canonical_order = np.lexsort(canonical.T[::-1])
        source_order = np.lexsort(source.T[::-1])
        if len(source) == len(canonical) and np.array_equal(canonical[canonical_order], source[source_order]):
            ids = np.empty((len(canonical), 1), dtype=np.int64)
            ids[canonical_order, 0] = source_order
            distances = np.zeros_like(ids, dtype=np.float64)
            status, reverse_max = 'reordered', 0.0
        else:
            ids, distances = nearest_neighbors(canonical, source, k=3)
            _, reverse = nearest_neighbors(source, canonical)
            reverse_max = float(reverse.max())
            status = 'interpolated'
            if max_distance is None:
                raise ValueError('Nonidentical point sets require explicit max_distance')
            if max(float(distances[:, 0].max()), reverse_max) > max_distance:
                raise ValueError('Bidirectional alignment distance exceeds max_distance')
    weights = 1.0 / np.maximum(distances, 1e-12)
    exact = distances[:, 0] == 0
    weights[exact] = 0
    weights[exact, 0] = 1
    weights /= weights.sum(1, keepdims=True)
    aligned = (masks[ids] * weights[:, :, None]).sum(1).astype(np.float32)
    quality = {'status': status, 'max_distance': float(distances[:, 0].max()),
               'mean_distance': float(distances[:, 0].mean()),
               'reverse_max_distance': reverse_max}
    return aligned, ids, weights.astype(np.float32), quality
