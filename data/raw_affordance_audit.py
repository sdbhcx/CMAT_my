"""Inventory original PIAD file lists without inventing instance/cue mappings."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import numpy as np

from .object_affordance_index import PIAD_AFFORDANCES, relative_path, resolve_path, read_point_source, write_json


def write_raw_object_summary(inventory, output_dir):
    """Derive point-object candidates and partition-aware statistics from inventory.

    These are NOT training indexes: image identity is still unknown.
    Can also regenerate reports without rereading large source datasets.
    """
    groups = defaultdict(list)
    for entry in inventory:
        key = entry.get('original_object_id') or ('coordinate:' + entry['unordered_coordinate_hash'])
        groups[key].append(entry)
    vocabulary = sorted({label for e in inventory for label in (e.get('mask_semantics') or [])})
    label_ids = {name: i for i, name in enumerate(vocabulary)}
    cooccurrence = defaultdict(lambda: np.zeros((len(vocabulary), len(vocabulary)), dtype=np.int64))
    candidates, within = [], Counter()
    for key, entries in sorted(groups.items()):
        by_partition = defaultdict(set)
        splits_by_setting = defaultdict(set)
        for entry in entries:
            partition = (entry['setting'], entry['split'])
            splits_by_setting[entry['setting']].add(entry['split'])
            semantics = entry.get('mask_semantics')
            if semantics:
                by_partition[partition].update(semantics[c] for c in entry['nonempty_mask_columns'])
        for setting, splits in splits_by_setting.items():
            if len(splits) > 1:
                within[setting] += 1
        for (setting, split), names in by_partition.items():
            ids = [label_ids[name] for name in names]
            cooccurrence[f'{setting}/{split}'][np.ix_(ids, ids)] += 1
        candidates.append({'point_object_key': key,
                           'original_id_recovered': bool(entries[0].get('original_object_id')),
                           'image_identity_verified': False,
                           'categories': sorted({e['category'] for e in entries}),
                           'point_sources': [e['path'] for e in entries],
                           'partitions': sorted({(e['setting'], e['split']) for e in entries}),
                           'coordinate_variants': len({e['unordered_coordinate_hash'] for e in entries}),
                           'nonempty_affordances': sorted(set().union(*by_partition.values())) if by_partition else []})
    stats = {'point_object_candidates': len(candidates),
             'within_setting_split_overlap_groups': dict(within),
             'identity_basis': 'PIAD original UUID; otherwise exact-coordinate candidate only',
             'usable_for_training': False}
    write_json(Path(output_dir) / 'point_object_candidates.json', candidates)
    write_json(Path(output_dir) / 'object_summary.json', stats)
    write_json(Path(output_dir) / 'cooccurrence.json', {
        'vocabulary': vocabulary, 'counts_by_partition': {k: v.tolist() for k, v in sorted(cooccurrence.items())},
        'definition': 'Nonempty mask co-occurrence per point-object candidate; not verified image/object sets'})
    return stats


def _prefetch_points(root, dataset, paths, workers):
    """Bounded IO concurrency; ordered results keep reports deterministic."""
    def read(raw_path):
        path = relative_path(raw_path)
        path = path[5:] if path.startswith('Data/') else path
        try:
            points, masks = read_point_source(root, {'path': path, 'format': 'piad_txt' if dataset == 'piad' else 'piadv2_npy'})
            return path, points, masks, None
        except (ValueError, OSError, IndexError) as error:
            return path, None, None, str(error).replace(str(root), '<data_root>')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(paths), 64):
            yield from pool.map(read, paths[start:start + 64])


def audit_raw_dataset(data_root, dataset, output_dir, max_files=None, workers=4):
    root = Path(data_root)
    if dataset not in ('piad', 'piadv2'):
        raise ValueError('dataset must be piad/piadv2')
    if max_files is not None and max_files < 1:
        raise ValueError('max_files must be positive')
    if workers < 1:
        raise ValueError('workers must be positive')
    lists = sorted(p for p in root.glob('*/*.txt') if p.stem.lower().startswith('point_'))
    if not lists:
        raise ValueError('No <setting>/Point_<split>.txt lists found under data root')
    inventory, anomalies, image_counts = [], [], Counter()
    image_records, candidate_groups = [], defaultdict(list)
    raw_ids = defaultdict(set)
    partitions = Counter()
    for listing in sorted(p for p in root.glob('*/*.txt') if p.stem.lower().startswith('img_')):
        for line in listing.read_text().splitlines():
            if not line.strip():
                continue
            path = relative_path(line.strip())
            path = path[5:] if path.startswith('Data/') else path
            parts = Path(path).parts
            if len(parts) < 6:
                anomalies.append({'path': path, 'reason': 'Unrecognized image layout'})
                continue
            category, affordance = parts[3], parts[-2]
            key = (listing.parent.name, listing.stem.split('_', 1)[1].lower(), category, affordance)
            image_counts[key] += 1
            if not resolve_path(root, path).is_file():
                anomalies.append({'path': path, 'reason': 'Missing image'})
    for key, count in sorted(image_counts.items()):
        image_records.append(dict(zip(('setting', 'split', 'category', 'affordance', 'count'), (*key, count))))
    processed = 0
    for listing in lists:
        setting, split = listing.parent.name, listing.stem.split('_', 1)[1].lower()
        paths = sorted(set(line.strip() for line in listing.read_text().splitlines() if line.strip()))
        partitions[f'{setting}/{split}'] = len(paths)
        if max_files is not None:
            paths = paths[:max_files]
        for path, points, masks, read_error in _prefetch_points(root, dataset, paths, workers):
            entry = {'path': path, 'setting': setting, 'split': split}
            try:
                if read_error:
                    raise ValueError(read_error)
                parts = Path(path).parts
                entry.update(category=parts[3], num_points=len(points), mask_columns=masks.shape[1])
                if dataset == 'piad':
                    metadata = {tuple(line.split()[:2]) for line in resolve_path(root, path).read_text().splitlines() if line.strip()}
                    if len(metadata) != 1:
                        raise ValueError('Inconsistent object ID/category inside point file')
                    oid, category = next(iter(metadata))
                    if category != entry['category']:
                        raise ValueError('Content category differs from path')
                    entry['original_object_id'] = oid
                    raw_ids[oid].add((setting, split))
                    # Label order follows the repository PIAD adapter, only if column count agrees.
                    entry['mask_semantics'] = PIAD_AFFORDANCES if masks.shape[1] == len(PIAD_AFFORDANCES) else None
                    entry['semantics_evidence'] = 'repository PIAD loader label order; requires dataset documentation confirmation'
                    if entry['mask_semantics'] is None:
                        anomalies.append({'path': path, 'reason': 'Mask column count differs from PIAD loader vocabulary'})
                else:
                    entry['original_object_id'] = None
                    entry['source_dataset'] = parts[-3]
                    entry['mask_semantics'] = [parts[-2]] if masks.shape[1] == 1 else None
                    entry['semantics_evidence'] = 'single mask + affordance directory' if masks.shape[1] == 1 else 'unknown multi-column schema'
                    if entry['mask_semantics'] is None:
                        anomalies.append({'path': path, 'reason': 'PIADv2 multi-column semantics require explicit schema'})
                # Geometry hashes are ONLY candidate matches, never object IDs.
                coordinates = points.copy()
                coordinates[coordinates == 0] = 0  # canonicalize negative zero
                order = np.lexsort(coordinates.T[::-1])
                ordered_hash = hashlib.sha256(coordinates.astype('<f8').tobytes()).hexdigest()
                geometry_hash = hashlib.sha256(coordinates[order].astype('<f8').tobytes()).hexdigest()
                entry.update(ordered_coordinate_hash=ordered_hash, unordered_coordinate_hash=geometry_hash)
                entry['nonempty_mask_columns'] = np.flatnonzero(masks.max(0) > 0).tolist()
                # Full per-file soft masks can overlap; report thresholded IoU.
                active = np.flatnonzero(masks.max(0) > 0)
                binary = masks[:, active] >= 0.5
                overlap = []
                for i in range(len(active)):
                    for j in range(i + 1, len(active)):
                        union = np.logical_or(binary[:, i], binary[:, j]).sum()
                        overlap.append({'columns': [int(active[i]), int(active[j])],
                                        'iou': float(np.logical_and(binary[:, i], binary[:, j]).sum() / union) if union else 1.0})
                entry['mask_overlap'] = overlap
                candidate_groups[geometry_hash].append(len(inventory))
                inventory.append(entry)
            except (ValueError, OSError, IndexError) as error:
                anomalies.append({'path': path, 'reason': str(error).replace(str(root), '<data_root>')})
            processed += 1
            if processed % 2000 == 0:
                print(f'Audited {processed} point files', flush=True)
    geometry_matches, geometry_leakage = [], []
    for fingerprint, positions in candidate_groups.items():
        if len(positions) < 2:
            continue
        entries = [inventory[i] for i in positions]
        partitions_here = sorted({(e['setting'], e['split']) for e in entries})
        entry = {'coordinate_hash': fingerprint, 'paths': [e['path'] for e in entries],
                 'partitions': partitions_here,
                 'point_order': 'same' if len({e['ordered_coordinate_hash'] for e in entries}) == 1 else 'reordered',
                 'nearest_neighbor_error': 0.0,
                 'identity_verified': False}
        geometry_matches.append(entry)
        if len(partitions_here) > 1:
            geometry_leakage.append(entry)
    object_leakage = [{'object_id': oid, 'partitions': sorted(parts)} for oid, parts in sorted(raw_ids.items()) if len(parts) > 1]
    # Counts are geometry candidates / point-file annotations, never trainable sets.
    candidate_coverage = {str(k): 0 for k in (2, 3, 4)}
    for positions in candidate_groups.values():
        groups = defaultdict(set)
        for i in positions:
            entry = inventory[i]
            semantics = entry['mask_semantics']
            if semantics:
                groups[(entry['setting'], entry['split'])].update(semantics[c] for c in entry['nonempty_mask_columns'])
        for labels in groups.values():
            for k in candidate_coverage:
                candidate_coverage[k] += len(labels) >= int(k)
    summary = {'dataset': dataset, 'decision': 'NO_GO', 'relation_ready': False,
               'reason': 'No verified mapping from HOI images to original point-cloud object IDs. Supply an explicit manifest.',
               'sampled': max_files is not None, 'max_files_per_partition': max_files,
               'listed_unique_files_per_partition': dict(partitions), 'scanned_point_files': processed,
               'valid_point_files': len(inventory), 'anomaly_count': len(anomalies),
               'mask_column_histogram': dict(Counter(str(e['mask_columns']) for e in inventory)),
               'recovered_original_object_ids': len(raw_ids),
               'object_id_partition_overlaps': len(object_leakage),
               'coordinate_match_groups': len(geometry_matches), 'coordinate_partition_overlaps': len(geometry_leakage),
               'candidate_groups_with_at_least_k_nonempty_affordances': candidate_coverage,
               'verified_trainable_multi_affordance_objects': 0,
               'alignment_scope': 'Exact coordinate matches only. Approximate alignment requires externally verified object IDs.',
               'partition_overlap_note': 'Cross-setting overlap may reflect reused experimental protocols; review separately from within-setting split leakage.'}
    for filename, value in [('summary.json', summary), ('point_inventory.json', inventory),
                            ('image_counts.json', image_records), ('anomalies.json', anomalies),
                            ('coordinate_matches.json', geometry_matches),
                            ('coordinate_partition_overlaps.json', geometry_leakage),
                            ('object_id_partition_overlaps.json', object_leakage)]:
        write_json(Path(output_dir) / filename, value)
    write_raw_object_summary(inventory, output_dir)
    return summary
