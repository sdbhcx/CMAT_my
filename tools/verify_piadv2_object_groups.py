"""Re-read PIADv2 to verify exact point-carrier groups and category/affordance cues.

This is an audit, not a split editor or a training-index builder. Geometry keys
identify exact sampled point clouds, not recovered original mesh UUIDs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np

from data.object_affordance_index import read_json, read_point_source, relative_path, resolve_path, write_json


def coordinate_hash(points, unordered=False):
    values = np.asarray(points, dtype=np.float64).copy()
    values[values == 0] = 0
    if unordered:
        values = values[np.lexsort(values.T[::-1])]
    return hashlib.sha256(values.astype('<f8').tobytes()).hexdigest()


def filename_diagnostics(rows):
    numbers, geometries = defaultdict(list), defaultdict(list)
    for row in rows:
        prefix = (row['setting'], row['split'], row['category'], row['source_dataset'].casefold())
        suffix = Path(row['path']).stem.rsplit('_', 1)[-1]
        numbers[(*prefix, suffix)].append(row)
        geometries[(*prefix, row['unordered_coordinate_hash'])].append(row)
    stats = defaultdict(Counter)
    examples = []
    for key, entries in numbers.items():
        if len({r['unordered_coordinate_hash'] for r in entries}) > 1:
            stats[key[0] + '/' + key[1]]['same_number_different_geometry_keys'] += 1
    for key, entries in geometries.items():
        if len({r['mask_semantics'][0] for r in entries}) < 2:
            continue
        suffixes = {Path(r['path']).stem.rsplit('_', 1)[-1] for r in entries}
        part = key[0] + '/' + key[1]
        stats[part]['multi_affordance_geometry_groups'] += 1
        stats[part]['groups_with_different_filename_numbers'] += len(suffixes) > 1
        if len(examples) < 5 and len(suffixes) > 1:
            examples.append({'partition': part, 'files': [r['path'] for r in entries]})
    return {'statistics_by_partition': dict(stats), 'examples': examples,
            'conclusion': 'Filename suffix is not a reliable cross-affordance object identifier'}


def read_image_pools(root):
    pools = defaultdict(set)
    anomalies = []
    for listing in sorted(Path(root).glob('*/Img_*.txt')):
        setting, split = listing.parent.name, listing.stem.split('_', 1)[1].lower()
        for line in listing.read_text().splitlines():
            if not line.strip():
                continue
            path = relative_path(line.strip())
            path = path[5:] if path.startswith('Data/') else path
            parts = Path(path).parts
            if len(parts) != 7 or parts[0] != setting or parts[2].lower() != split:
                anomalies.append({'path': path, 'reason': 'Unexpected image partition/layout'})
                continue
            if not resolve_path(root, path).is_file():
                anomalies.append({'path': path, 'reason': 'Missing image'})
                continue
            pools[(setting, split, parts[3], parts[-2])].add(path)
    return pools, anomalies


def verify_group(root, fingerprint, rows, image_pools):
    categories = {r['category'] for r in rows}
    sources = {r['source_dataset'].casefold() for r in rows}
    result = {'geometry_id': 'xyz-sha256:' + fingerprint,
              'original_mesh_id_recovered': False,
              'categories': sorted(categories), 'sources': sorted(sources),
              'files_checked': len(rows), 'issues': [], 'partitions': [],
              'cross_partition_annotation_conflicts': []}
    if len(categories) != 1 or len(sources) != 1:
        result['issues'].append('Exact geometry has conflicting categories/source datasets')
    canonical = None
    partitions = defaultdict(list)
    all_labels = {}
    for row in sorted(rows, key=lambda r: r['path']):
        path = row['path']
        try:
            points, masks = read_point_source(root, {'path': path, 'format': 'piadv2_npy'})
            if points.shape != (2048, 3) or masks.shape != (2048, 1):
                raise ValueError('Expected PIADv2 [2048, xyz + one mask]')
            if coordinate_hash(points, True) != fingerprint or coordinate_hash(points) != row['ordered_coordinate_hash']:
                raise ValueError('Coordinates changed since raw inventory')
            if canonical is None:
                canonical = points
            same_order = np.array_equal(canonical, points)
            if same_order:
                mask = masks[:, 0]
            else:
                source_order = np.lexsort(points.T[::-1])
                canonical_order = np.lexsort(canonical.T[::-1])
                if not np.array_equal(points[source_order], canonical[canonical_order]):
                    raise ValueError('Coordinate hash match failed direct array comparison')
                if len(np.unique(points, axis=0)) != len(points):
                    raise ValueError('Reorder with duplicate coordinates is ambiguous')
                mask = np.empty(len(points), dtype=np.float32)
                mask[canonical_order] = masks[source_order, 0]
            affordance = row['mask_semantics'][0]
            if Path(path).parts[-2] != affordance:
                raise ValueError('Affordance label disagrees with path')
            partition = (row['setting'], row['split'])
            partitions[partition].append((row, mask, same_order))
            if affordance in all_labels:
                previous_path, previous_mask = all_labels[affordance]
                if not np.allclose(previous_mask, mask, rtol=0, atol=1e-6):
                    result['cross_partition_annotation_conflicts'].append({
                        'affordance': affordance, 'reference': previous_path, 'path': path,
                        'max_absolute_difference': float(np.max(np.abs(previous_mask - mask)))})
            else:
                all_labels[affordance] = (path, mask)
        except (ValueError, OSError, KeyError, IndexError) as error:
            result['issues'].append({'path': path, 'reason': str(error).replace(str(root), '<data_root>')})
    for (setting, split), entries in sorted(partitions.items()):
        masks_by_aff, paths_by_aff = {}, defaultdict(list)
        conflicts = []
        for row, mask, _ in entries:
            aff = row['mask_semantics'][0]
            paths_by_aff[aff].append(row['path'])
            if aff in masks_by_aff and not np.allclose(mask, masks_by_aff[aff], rtol=0, atol=1e-6):
                conflicts.append(aff)
            masks_by_aff[aff] = mask
        labels = sorted(masks_by_aff)
        category = entries[0][0]['category']
        cue_counts = {aff: len(image_pools.get((setting, split, category, aff), ())) for aff in labels}
        pairs = []
        for a, b in combinations(labels, 2):
            ma, mb = masks_by_aff[a], masks_by_aff[b]
            ba, bb = ma >= .5, mb >= .5
            union = np.logical_or(ba, bb).sum()
            pairs.append({'affordances': [a, b],
                          'iou_at_0_5': float(np.logical_and(ba, bb).sum() / union) if union else 1.0,
                          'soft_mask_mae': float(np.abs(ma - mb).mean()),
                          'identical_masks': bool(np.allclose(ma, mb, rtol=0, atol=1e-6))})
        record = {'setting': setting, 'split': split, 'category': category,
                  'affordances': labels, 'paths_by_affordance': dict(paths_by_aff),
                  'same_point_order': all(entry[2] for entry in entries),
                  'nearest_neighbor_error': 0.0,
                  'conflicting_duplicate_affordances': sorted(set(conflicts)),
                  'image_pool_counts': cue_counts,
                  'nonempty_affordances': [aff for aff in labels if np.any(masks_by_aff[aff] > 0)],
                  'mask_pairs': pairs,
                  'eligible_before_split_filter': not result['issues'] and not conflicts and all(cue_counts.values())}
        result['partitions'].append(record)
    return result


def summarize(groups, image_anomalies):
    coverage = defaultdict(Counter)
    overlap = Counter()
    excluded = defaultdict(list)
    pair_counts = Counter()
    sources = defaultdict(Counter)
    for group in groups:
        by_setting = defaultdict(list)
        for part in group['partitions']:
            key = part['setting'] + '/' + part['split']
            stats = coverage[key]
            stats['geometry_groups'] += 1
            stats['missing_image_pool_groups'] += not all(part['image_pool_counts'].values())
            stats['conflicting_mask_groups'] += bool(part['conflicting_duplicate_affordances'])
            stats['eligible_groups_before_split_filter'] += part['eligible_before_split_filter']
            for k in (2, 3, 4):
                stats[f'groups_A_ge_{k}'] += len(part['affordances']) >= k
                stats[f'eligible_A_ge_{k}_before_split_filter'] += part['eligible_before_split_filter'] and len(part['affordances']) >= k
            stats['multi_affordance_same_order_groups'] += len(part['affordances']) >= 2 and part['same_point_order']
            sources[key][group['sources'][0]] += len(part['affordances']) >= 2
            for pair in part['mask_pairs']:
                pair_counts['pairs'] += 1
                pair_counts['identical_masks'] += pair['identical_masks']
            by_setting[part['setting']].append(part)
        for setting, parts in by_setting.items():
            names = sorted({p['split'] for p in parts})
            for a, b in combinations(names, 2):
                overlap[f'{setting}/{a}-{b}'] += 1
            # Proposal only: retain held-out sets, remove overlap from training.
            overlap_train = 'train' in names and any(name in ('val', 'test') for name in names)
            if overlap_train:
                excluded[setting].append(group['geometry_id'])
            for part in parts:
                if part['split'] == 'train' and not overlap_train and part['eligible_before_split_filter']:
                    stats = coverage[setting + '/train']
                    stats['eligible_groups_after_proposed_filter'] += 1
                    for k in (2, 3, 4):
                        stats[f'eligible_A_ge_{k}_after_proposed_filter'] += len(part['affordances']) >= k
    return {'image_matching_policy': 'same category + affordance, within each setting/split',
            'point_group_identity': 'Exact 2048-point coordinate carrier; not original mesh provenance',
            'files_rechecked': sum(g['files_checked'] for g in groups),
            'geometry_groups': len(groups),
            'groups_with_source_or_coordinate_issues': sum(bool(g['issues']) for g in groups),
            'image_path_anomalies': len(image_anomalies),
            'groups_with_cross_partition_annotation_conflicts': sum(bool(g['cross_partition_annotation_conflicts']) for g in groups),
            'coverage_by_partition': dict(coverage),
            'multi_affordance_groups_by_source': dict(sources),
            'same_geometry_split_overlaps': dict(overlap),
            'proposed_train_exclusions': {k: len(v) for k, v in excluded.items()},
            'mask_pair_statistics': dict(pair_counts),
            'original_data_or_splits_modified': False,
            'limitations': ['No original mesh IDs recovered from file names.',
                            'Different sampled point clouds of the same mesh may be missed by exact grouping.',
                            'Cross-setting reuse is not treated as within-protocol leakage.',
                            'This report does not enable training or change the existing index schema.']}, dict(excluded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--inventory', default='work/data_audit/piadv2/all/point_inventory.json')
    parser.add_argument('--output-dir', default='work/data_audit/piadv2/object_verification')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    rows = read_json(args.inventory)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['unordered_coordinate_hash']].append(row)
    image_pools, image_anomalies = read_image_pools(args.data_root)
    entries = sorted(grouped.items())
    verified = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(entries), 128):
            jobs = [pool.submit(verify_group, args.data_root, fingerprint, records, image_pools)
                    for fingerprint, records in entries[start:start + 128]]
            verified.extend(job.result() for job in jobs)
            if start % 2048 == 0:
                print(f'Verified {len(verified)}/{len(entries)} geometry groups', flush=True)
    summary, excluded = summarize(verified, image_anomalies)
    output = Path(args.output_dir)
    write_json(output / 'summary.json', summary)
    write_json(output / 'verified_groups.json', verified)
    write_json(output / 'image_anomalies.json', image_anomalies)
    write_json(output / 'proposed_train_exclusions.json', excluded)
    write_json(output / 'filename_id_diagnostics.json', filename_diagnostics(rows))
    import json
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
