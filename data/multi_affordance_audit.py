"""Audit explicit instance mappings before building object-level training data."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import io
from itertools import combinations
from pathlib import Path

import numpy as np

from .affordance_alignment import align_masks, canonical_indices
from .object_affordance_index import (INDEX_VERSION, read_point_source, relative_path,
                                     resolve_path, sha256_file, write_json, save_index)


def audit_manifest(manifest, data_root, split, setting, output_dir,
                   max_distance=None, canonical_count=2048, min_multi_objects=1,
                   index_path=None):
    """Audit ALL manifest splits; optionally persist the selected split's index.

    Identity is an externally verified assertion, never inferred from geometry.
    A single rejected source rejects its whole object, preventing partial masks
    from silently changing the object's union target.
    """
    if manifest.get('version') != 1 or manifest.get('dataset') not in ('piad', 'piadv2'):
        raise ValueError('Expected manifest version 1 and dataset piad/piadv2')
    vocabulary = manifest.get('affordance_vocabulary', [])
    if not vocabulary or len(vocabulary) != len(set(vocabulary)) or any(not isinstance(x, str) or not x for x in vocabulary):
        raise ValueError('Manifest needs a unique affordance vocabulary')
    if min_multi_objects < 1 or canonical_count < 1:
        raise ValueError('min_multi_objects and canonical_count must be positive')
    if max_distance is not None and (not np.isfinite(max_distance) or max_distance < 0):
        raise ValueError('max_distance must be finite and nonnegative')
    records = manifest.get('records', [])
    if not records:
        raise ValueError('Manifest is empty')
    grouped = defaultdict(list)
    for row in records:
        for key in ('object_id', 'category', 'split', 'setting'):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f'Every manifest record needs a nonempty {key}')
        grouped[row['object_id']].append(row)
    categories = sorted({r['category'] for r in records})
    leakage, anomalies, object_report, column_report = [], [], [], []
    for oid, rows in sorted(grouped.items()):
        partitions = sorted({(r['setting'], r['split']) for r in rows})
        if len(partitions) > 1:
            leakage.append({'object_id': oid, 'partitions': partitions})
    leaked_ids = {entry['object_id'] for entry in leakage}
    selected = {oid: rows for oid, rows in grouped.items()
                if any(r['split'] == split and r['setting'] == setting for r in rows)}
    objects, overlaps = [], []
    cooccurrence = np.zeros((len(vocabulary), len(vocabulary)), dtype=np.int64)
    image_owners = defaultdict(set)
    source_owners = defaultdict(set)
    for row in records:
        source_owners[relative_path(row['point_source']['path'])].add(row['object_id'])
        for paths in row.get('images', {}).values():
            for path in paths:
                image_owners[relative_path(path)].add(row['object_id'])
    for oid, rows in sorted(selected.items()):
        try:
            if oid in leaked_ids:
                raise ValueError('Object ID occurs in multiple split/setting partitions')
            if len({r['category'] for r in rows}) != 1:
                raise ValueError('Object ID has conflicting categories')
            if any(r.get('identity_verified') is not True or not r.get('identity_evidence') for r in rows):
                raise ValueError('Object/image identity has no verified provenance')
            rows = sorted(rows, key=lambda r: relative_path(r['point_source']['path']))
            loaded = []
            for row in rows:
                source = row['point_source']
                source_path = relative_path(source['path'])
                if len(source_owners[source_path]) != 1:
                    raise ValueError('A point source is assigned to multiple object IDs')
                points, masks = read_point_source(data_root, source)
                if source['format'] == 'piad_txt':
                    identities = {tuple(line.split()[:2]) for line in resolve_path(data_root, source_path).read_text().splitlines() if line.strip()}
                    if identities != {(oid, row['category'])}:
                        raise ValueError('PIAD content object ID/category differs from manifest')
                columns = row.get('mask_columns', {})
                column_report.append({'object_id': oid, 'path': source_path,
                                      'mask_column_count': masks.shape[1], 'semantics': columns})
                if (not columns or any(name not in vocabulary for name in columns) or
                        any(type(c) is not int for c in columns.values()) or
                        sorted(columns.values()) != list(range(masks.shape[1]))):
                    raise ValueError('mask_columns must explicitly name EVERY mask column exactly once')
                images = row.get('images', {})
                if not images or any(name not in columns for name in images):
                    raise ValueError('Images must reference declared mask columns')
                for paths in images.values():
                    if not isinstance(paths, list) or not paths:
                        raise ValueError('Each selected affordance needs an image list')
                    for path in paths:
                        path = relative_path(path)
                        if len(image_owners[path]) != 1:
                            raise ValueError('An image is assigned to multiple object IDs')
                        # Decode to catch corrupted images before training.
                        from PIL import Image
                        with Image.open(resolve_path(data_root, path)) as image:
                            image.verify()
                loaded.append((row, points, masks))
            canonical = loaded[0][1]
            masks_by_name, images_by_name, alignments, sources = {}, defaultdict(set), {}, defaultdict(list)
            maps = {}
            for source_index, (row, points, masks) in enumerate(loaded):
                aligned, indices, weights, quality = align_masks(canonical, points, masks, max_distance)
                maps[f'indices_{source_index}'] = indices
                maps[f'weights_{source_index}'] = weights
                for name, paths in row['images'].items():
                    mask = aligned[:, row['mask_columns'][name]]
                    if name in masks_by_name and not np.allclose(masks_by_name[name], mask, atol=1e-6, rtol=0):
                        raise ValueError(f'Conflicting duplicate masks for {name}')
                    masks_by_name[name] = mask
                    images_by_name[name].update(relative_path(p) for p in paths)
                    previous = alignments.get(name)
                    severity = {'exact': 0, 'reordered': 1, 'interpolated': 2}
                    if previous is None or (severity[quality['status']], quality['max_distance']) >= (severity[previous['status']], previous['max_distance']):
                        alignments[name] = quality
                    sources[name].append({'point_source': row['point_source'],
                                          'mask_column': row['mask_columns'][name],
                                          'map_index': source_index, 'alignment': quality})
            # Only independently sampled clouds need a resampled canonical cloud.
            needs_canonical = any(q['status'] == 'interpolated' for q in alignments.values())
            sample_ids = canonical_indices(canonical, canonical_count) if needs_canonical else np.arange(len(canonical))
            points = canonical[sample_ids].astype(np.float32)
            names = sorted(masks_by_name, key=vocabulary.index)
            stacked = np.stack([masks_by_name[n][sample_ids] for n in names])
            object_pairs = []
            for i, j in combinations(range(len(names)), 2):
                a, b = stacked[i] >= 0.5, stacked[j] >= 0.5
                union = np.logical_or(a, b).sum()
                iou = float(np.logical_and(a, b).sum() / union) if union else 1.0
                pair = {'affordance_ids': [vocabulary.index(names[i]), vocabulary.index(names[j])], 'iou': iou}
                object_pairs.append(pair)
                overlaps.append({'object_id': oid, **pair})
            aids = [vocabulary.index(name) for name in names]
            cooccurrence[np.ix_(aids, aids)] += 1
            obj = {'object_id': oid, 'category': rows[0]['category'],
                   'identity_verified': True,
                   'identity_evidence': sorted({r['identity_evidence'] for r in rows}),
                   'num_points': len(points), 'canonical_point_source': rows[0]['point_source'],
                   'overlap_pairs': object_pairs,
                   'affordances': [{'affordance_id': vocabulary.index(name), 'name': name,
                                    'images': sorted(images_by_name[name]), 'alignment': alignments[name],
                                    'mask_sources': sources[name]} for name in names]}
            if index_path is not None:
                buffer = io.BytesIO()
                np.savez_compressed(buffer, points=points, masks=stacked,
                                    canonical_indices=sample_ids,
                                    **{k: v[sample_ids] for k, v in maps.items()})
                payload = buffer.getvalue()
                digest = hashlib.sha256(payload).hexdigest()
                artifact = Path(index_path).parent / 'objects' / f'{digest}.npz'
                artifact.parent.mkdir(parents=True, exist_ok=True)
                if not artifact.exists():
                    artifact.write_bytes(payload)
                obj['artifact'] = artifact.relative_to(Path(index_path).parent).as_posix()
                obj['artifact_sha256'] = sha256_file(artifact)
            objects.append(obj)
            object_report.append({'object_id': oid, 'category': obj['category'], 'accepted': True,
                                  'num_affordances': len(names), 'num_points': len(points),
                                  'images_per_affordance': {n: len(images_by_name[n]) for n in names},
                                  'alignment': alignments})
        except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
            message = str(error).replace(str(Path(data_root)), '<data_root>')
            anomalies.append({'object_id': oid, 'reason': message})
            object_report.append({'object_id': oid, 'accepted': False, 'reason': message})
    coverage = {str(k): sum(len(o['affordances']) >= k for o in objects) for k in (1, 2, 3, 4)}
    relation_ready = coverage['2'] >= min_multi_objects and not leakage
    summary = {'dataset': manifest['dataset'], 'split': split, 'setting': setting,
               'manifest_object_count': len(grouped), 'selected_object_count': len(selected),
               'accepted_objects': len(objects), 'rejected_objects': len(anomalies),
               'objects_with_at_least_k_affordances': coverage,
               'mask_column_histogram': dict(Counter(str(r['mask_column_count']) for r in column_report)),
               'leakage_count': len(leakage), 'min_multi_objects': min_multi_objects,
               'relation_ready': relation_ready,
               'decision': 'GO' if relation_ready else 'NO_GO',
               'scope': 'Only supplied manifest records were checked; supply ALL partitions.',
               'identity_policy': 'Externally verified mapping; geometry does not prove identity.'}
    output_dir = Path(output_dir)
    for filename, content in [('summary.json', summary), ('objects.json', object_report),
                              ('anomalies.json', anomalies), ('split_leakage.json', leakage),
                              ('mask_columns.json', column_report), ('mask_overlap.json', overlaps),
                              ('cooccurrence.json', {'vocabulary': vocabulary, 'counts': cooccurrence.tolist()})]:
        write_json(output_dir / filename, content)
    if index_path is not None:
        if not objects:
            raise ValueError(f'No accepted objects; inspect audit report in {output_dir}')
        index = {'version': INDEX_VERSION, 'dataset': manifest['dataset'], 'split': split,
                 'setting': setting, 'affordance_vocabulary': vocabulary,
                 'category_vocabulary': categories, 'audit': summary,
                 'alignment_policy': {'max_distance': max_distance, 'units': 'source_coordinates',
                                      'canonical_count': canonical_count, 'method': 'exact_or_inverse_distance_3nn'},
                 'objects': objects}
        save_index(index_path, index)
    return summary
