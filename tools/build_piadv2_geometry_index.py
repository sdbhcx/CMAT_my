"""Build PIADv2 exact-coordinate object indexes with category/affordance cues."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from data.object_affordance_index import read_json, read_point_source, resolve_path, save_index, write_json
from tools.verify_piadv2_object_groups import coordinate_hash, read_image_pools

AFFORDANCES = ['grasp', 'contain', 'lift', 'open', 'lay', 'sit', 'support',
               'wrapgrasp', 'pour', 'move', 'display', 'push', 'listen', 'wear',
               'press', 'cut', 'stab', 'carry', 'ride', 'clean', 'play', 'beat', 'speak', 'pull']


def build_object(root, output, group, part, pools):
    if group['issues'] or part['conflicting_duplicate_affordances']:
        raise ValueError('Unresolved audit issues: ' + group['geometry_id'])
    category = part['category']
    source = group['sources'][0]
    fingerprint = group['geometry_id'].split(':', 1)[1]
    object_id = f'{category}:{source}:{fingerprint}'
    points = None
    masks, affordances = [], []
    quality = {'status': 'exact', 'max_distance': 0.0, 'mean_distance': 0.0, 'reverse_max_distance': 0.0}
    for name in sorted(part['affordances'], key=AFFORDANCES.index):
        sources = sorted(part['paths_by_affordance'][name])
        mask = None
        for path in sources:
            xyz, labels = read_point_source(root, {'path': path, 'format': 'piadv2_npy'})
            if xyz.shape != (2048, 3) or labels.shape != (2048, 1):
                raise ValueError('Unexpected point/mask schema: ' + path)
            if coordinate_hash(xyz, True) != fingerprint:
                raise ValueError('Coordinates changed after audit: ' + path)
            if points is None:
                points = xyz
                canonical_source = path
            if not np.array_equal(points, xyz):
                raise ValueError('This index builder requires verified identical point order: ' + path)
            if mask is not None and not np.allclose(mask, labels[:, 0], atol=1e-6, rtol=0):
                raise ValueError('Conflicting masks: ' + path)
            mask = labels[:, 0]
        pool_key = f'{category}/{name}'
        if pool_key not in pools:
            raise ValueError('Missing category/affordance cue pool: ' + pool_key)
        masks.append(mask)
        affordances.append({'affordance_id': AFFORDANCES.index(name), 'name': name,
                            'image_pool_key': pool_key, 'alignment': dict(quality),
                            'mask_sources': [{'point_source': {'path': p, 'format': 'piadv2_npy'},
                                              'mask_column': 0, 'alignment': dict(quality)} for p in sources]})
    points = points.astype(np.float32)
    masks = np.stack(masks).astype(np.float32)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, points=points, masks=masks, canonical_indices=np.arange(len(points)))
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    artifact = Path(output) / 'objects' / (digest + '.npz')
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if not artifact.exists():
        artifact.write_bytes(payload)
    pairs = []
    for i, j in combinations(range(len(affordances)), 2):
        left, right = masks[i] >= .5, masks[j] >= .5
        union = np.logical_or(left, right).sum()
        pairs.append({'affordance_ids': [affordances[i]['affordance_id'], affordances[j]['affordance_id']],
                      'iou': float(np.logical_and(left, right).sum() / union) if union else 1.0})
    return {'object_id': object_id, 'category': category, 'source_dataset': source,
            'identity_verified': True, 'identity_kind': 'exact_sampled_geometry',
            'identity_evidence': 'All source xyz rows re-read and equal; category/source agree. Image cues use category + affordance.',
            'original_mesh_id_recovered': False, 'geometry_sha256': fingerprint,
            'num_points': len(points), 'artifact': artifact.relative_to(output).as_posix(),
            'artifact_sha256': digest, 'canonical_point_source': {'path': canonical_source, 'format': 'piadv2_npy'},
            'overlap_pairs': pairs, 'affordances': affordances}


def build_indexes(data_root, verification_dir, output_dir, setting='Unseen_obj', workers=8,
                  overlap_policy='error', min_multi_objects=1):
    if workers < 1 or min_multi_objects < 1:
        raise ValueError('workers/min_multi_objects must be positive')
    if overlap_policy not in ('error', 'drop-train'):
        raise ValueError('overlap_policy must be error or drop-train')
    groups = read_json(Path(verification_dir) / 'verified_groups.json')
    partitions = {}
    excluded = []
    for group in groups:
        parts = [p for p in group['partitions'] if p['setting'] == setting]
        names = {p['split'] for p in parts}
        if len(names) > 1:
            if overlap_policy == 'error' or len(names - {'train'}) > 1:
                raise ValueError(f'Overlapping point group in {setting}: {group["geometry_id"]}; splits {sorted(names)}')
            excluded.append(group['geometry_id'])
            parts = [p for p in parts if p['split'] != 'train']
        for part in parts:
            partitions.setdefault(part['split'], []).append((group, part))
    if not {'train', 'test'} <= partitions.keys():
        raise ValueError('Both train and test partitions are required')
    categories = sorted({p['category'] for entries in partitions.values() for _, p in entries})
    raw_pools, image_errors = read_image_pools(data_root)
    relevant_errors = [e for e in image_errors if e['path'].startswith(setting + '/')]
    if relevant_errors:
        raise ValueError(f'Image path audit failed: {relevant_errors[:3]}')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for split, entries in sorted(partitions.items()):
        needed = {(part['category'], name) for _, part in entries for name in part['affordances']}
        pools = {}
        for category, name in sorted(needed):
            images = sorted(raw_pools.get((setting, split, category, name), ()))
            if not images:
                raise ValueError(f'No cue images for {setting}/{split}/{category}/{name}')
            pools[f'{category}/{name}'] = {'category': category, 'affordance': name,
                                         'setting': setting, 'split': split, 'images': images}
        # Decode each image once instead of once for every object using the pool.
        image_paths = sorted({path for pool in pools.values() for path in pool['images']})
        def check_image(path):
            with Image.open(resolve_path(data_root, path)) as image:
                image.convert('RGB').load()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for start in range(0, len(image_paths), 128):
                list(executor.map(check_image, image_paths[start:start + 128]))
        objects = []
        entries.sort(key=lambda entry: entry[0]['geometry_id'])
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for start in range(0, len(entries), 128):
                jobs = [executor.submit(build_object, data_root, output_dir, group, part, pools)
                        for group, part in entries[start:start + 128]]
                objects.extend(job.result() for job in jobs)
                if start % 2048 == 0:
                    print(f'{setting}/{split}: built {len(objects)}/{len(entries)} objects', flush=True)
        coverage = {str(k): sum(len(obj['affordances']) >= k for obj in objects) for k in (1, 2, 3, 4)}
        audit = {'decision': 'GO' if coverage['2'] >= min_multi_objects else 'SEGMENTATION_ONLY',
                 'relation_ready': coverage['2'] >= min_multi_objects,
                 'objects_with_at_least_k_affordances': coverage, 'min_multi_objects': min_multi_objects,
                 'leakage_count': 0, 'excluded_training_geometries': len(excluded),
                 'image_pairing': 'category_affordance', 'image_count': len(image_paths)}
        index = {'version': 2, 'dataset': 'piadv2', 'setting': setting, 'split': split,
                 'image_matching_policy': 'category_affordance',
                 'point_identity_policy': 'exact_sampled_geometry',
                 'affordance_vocabulary': AFFORDANCES, 'category_vocabulary': categories,
                 'image_pools': pools, 'objects': objects, 'audit': audit,
                 'alignment_policy': {'method': 'exact_rows', 'max_distance': 0.0, 'units': 'source_coordinates'}}
        save_index(output_dir / (split + '.json'), index)
        reports[split] = audit
    report = {'dataset': 'piadv2', 'setting': setting, 'overlap_policy': overlap_policy,
              'excluded_training_geometries': excluded, 'partitions': reports,
              'original_data_modified': False}
    write_json(output_dir / 'build_report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--verification-dir', default='work/data_audit/piadv2/object_verification')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--setting', default='Unseen_obj')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--overlap-policy', choices=['error', 'drop-train'], default='error')
    parser.add_argument('--min-multi-objects', type=int, default=1)
    args = parser.parse_args()
    import json
    print(json.dumps(build_indexes(args.data_root, args.verification_dir, args.output_dir,
                                    args.setting, args.workers, args.overlap_policy, args.min_multi_objects), indent=2))
