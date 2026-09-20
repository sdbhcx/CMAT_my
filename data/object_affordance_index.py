"""Portable, versioned object index. No torch dependency in audit tooling."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

INDEX_VERSION = 1
PIAD_AFFORDANCES = ['grasp', 'contain', 'lift', 'open', 'lay', 'sit', 'support',
                    'wrapgrasp', 'pour', 'move', 'display', 'push', 'listen',
                    'wear', 'press', 'cut', 'stab']


def relative_path(value):
    """Normalize portable relative paths, rejecting escapes and drive paths."""
    value = str(value).replace('\\', '/')
    path = PurePosixPath(value)
    if (not value or value == '.' or path.is_absolute() or
            PureWindowsPath(value).drive or '..' in path.parts):
        raise ValueError(f'Expected a root-relative path: {value}')
    return path.as_posix()


def resolve_path(root, value):
    root = Path(root).resolve()
    result = (root / relative_path(value)).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Path escapes its root')
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_point_source(root, source):
    """Read PIAD text (two leading metadata columns) or PIADv2 npy."""
    path = resolve_path(root, source['path'])
    if source['format'] == 'piad_txt':
        # Metadata columns may be strings; only xyz and labels are numeric.
        rows = [line.split()[2:] for line in path.read_text().splitlines() if line.strip()]
        values = np.asarray(rows, dtype=np.float64)
    elif source['format'] == 'piadv2_npy':
        values = np.load(path, allow_pickle=False)
    else:
        raise ValueError('format must be piad_txt or piadv2_npy')
    if values.ndim != 2 or len(values) == 0 or values.shape[1] < 4:
        raise ValueError('Point file must contain nonempty [N, xyz + masks]')
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError('Point coordinates and masks must be real numeric values')
    if not np.isfinite(values).all():
        raise ValueError('Point file contains NaN/Inf')
    points, masks = values[:, :3].astype(np.float64), values[:, 3:].astype(np.float32)
    if np.any((masks < 0) | (masks > 1)):
        raise ValueError('Mask values must lie in [0, 1]')
    return points, masks


def get_affordance_images(index, obj, affordance):
    if index.get('version') == 2:
        key = affordance.get('image_pool_key')
        pool = index.get('image_pools', {}).get(key)
        if not pool or pool['category'] != obj['category'] or pool['affordance'] != affordance['name']:
            raise ValueError('Cue pool must match object category and affordance')
        if pool['setting'] != index['setting'] or pool['split'] != index['split']:
            raise ValueError('Cue pool must belong to the same setting/split')
        return pool['images']
    return affordance['images']


def validate_index(index):
    if index.get('version') not in (INDEX_VERSION, 2):
        raise ValueError('Unsupported object index version')
    for key in ('dataset', 'split', 'setting'):
        if not isinstance(index.get(key), str) or not index[key]:
            raise ValueError(f'Missing index {key}')
    for key in ('affordance_vocabulary', 'category_vocabulary'):
        values = index.get(key, [])
        if not values or any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
            raise ValueError(f'Invalid {key}')
    if not index.get('objects'):
        raise ValueError('Index contains no accepted objects')
    if not isinstance(index.get('audit'), dict) or not isinstance(index.get('alignment_policy'), dict):
        raise ValueError('Missing audit/alignment policy')
    if index['version'] == 2:
        if index.get('image_matching_policy') != 'category_affordance' or index.get('point_identity_policy') != 'exact_sampled_geometry':
            raise ValueError('Version 2 indexes require explicit point and image policies')
        if not index.get('image_pools'):
            raise ValueError('Missing shared image pools')
        for pool in index['image_pools'].values():
            if not pool.get('images'):
                raise ValueError('Empty cue pool')
            for path in pool['images']:
                relative_path(path)
    ids = set()
    for obj in index['objects']:
        if not isinstance(obj.get('object_id'), str) or not obj['object_id'] or obj['object_id'] in ids:
            raise ValueError('Duplicate or missing object_id')
        ids.add(obj['object_id'])
        if obj.get('identity_verified') is not True or not obj.get('identity_evidence'):
            raise ValueError('Object identity must have explicit evidence')
        if obj['category'] not in index['category_vocabulary']:
            raise ValueError('Unknown category')
        if index['version'] == 2:
            fingerprint = obj.get('geometry_sha256', '')
            if (obj.get('identity_kind') != 'exact_sampled_geometry' or len(fingerprint) != 64
                    or any(c not in '0123456789abcdef' for c in fingerprint)):
                raise ValueError('Missing exact-geometry identity evidence')
        relative_path(obj['artifact'])
        if len(obj.get('artifact_sha256', '')) != 64 or obj.get('num_points', 0) < 1:
            raise ValueError('Missing artifact checksum/point count')
        aids = [a['affordance_id'] for a in obj['affordances']]
        if not aids or aids != sorted(set(aids)):
            raise ValueError('Affordances must be unique and sorted by ID')
        for a in obj['affordances']:
            aid = a['affordance_id']
            if type(aid) is not int or not 0 <= aid < len(index['affordance_vocabulary']):
                raise ValueError('Invalid affordance ID')
            images = get_affordance_images(index, obj, a)
            if a['name'] != index['affordance_vocabulary'][aid] or not images:
                raise ValueError('Affordance name/images mismatch')
            if index['version'] == 1:
                for image in images:
                    relative_path(image)
            quality = a['alignment']
            if quality['status'] not in ('exact', 'reordered', 'interpolated'):
                raise ValueError('Unaccepted mask alignment')
            for field in ('max_distance', 'mean_distance', 'reverse_max_distance'):
                if not np.isfinite(quality[field]) or quality[field] < 0:
                    raise ValueError('Invalid alignment quality')
            if quality['status'] == 'interpolated':
                limit = index['alignment_policy'].get('max_distance')
                if limit is None or max(quality['max_distance'], quality['reverse_max_distance']) > limit:
                    raise ValueError('Interpolation exceeds audited distance policy')
    return index


def load_index(path):
    return validate_index(read_json(path))


def save_index(path, index):
    write_json(path, validate_index(index))


def load_object_arrays(index_path, obj):
    path = resolve_path(Path(index_path).parent, obj['artifact'])
    if sha256_file(path) != obj['artifact_sha256']:
        raise ValueError(f"Artifact checksum mismatch for {obj['object_id']}; rebuild index")
    with np.load(path, allow_pickle=False) as data:
        points, masks = data['points'].copy(), data['masks'].copy()
    if points.shape != (obj['num_points'], 3) or masks.shape != (len(obj['affordances']), len(points)):
        raise ValueError('Artifact shapes do not match index')
    if not np.isfinite(points).all() or not np.isfinite(masks).all() or np.any((masks < 0) | (masks > 1)):
        raise ValueError('Invalid artifact values')
    return points, masks
