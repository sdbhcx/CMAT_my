"""Prepare disjoint object and cue holdouts from the official TRAIN index only."""
import argparse
from collections import defaultdict
import copy
import hashlib
from pathlib import Path

import yaml

from data.object_affordance_index import load_index, save_index, sha256_file, write_json


def holdout(values, seed, fraction=.1):
    if len(values) < 2:
        raise ValueError('At least two entries are needed for a disjoint holdout')
    ordered = sorted(values, key=lambda value: hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest())
    count = max(1, min(len(values)-1, round(len(values)*fraction)))
    return ordered[count:], ordered[:count]


def prepare(config_path):
    config = yaml.safe_load(Path(config_path).read_text())
    source = Path(config['data']['index_paths']['train'])
    index = load_index(source)
    if index['split'] != 'train':
        raise ValueError('Holdout must be derived from the official training partition')
    categories = defaultdict(list)
    objects = {o['object_id']: o for o in index['objects'] if len(o['affordances']) >= 2}
    for oid, obj in objects.items():
        categories[obj['category']].append(oid)
    selections = {'train': [], 'val': []}
    counts = {}
    for category, ids in sorted(categories.items()):
        train_ids, val_ids = holdout(ids, 42)
        selections['train'].extend(train_ids); selections['val'].extend(val_ids)
        counts[category] = {'train': len(train_ids), 'val': len(val_ids)}
    used = {a['image_pool_key'] for obj in objects.values() for a in obj['affordances']}
    pools = {'train': {}, 'val': {}}
    for key in sorted(used):
        pool = index['image_pools'][key]
        train_images, val_images = holdout(sorted(set(pool['images'])), 42)
        for split, images in (('train', train_images), ('val', val_images)):
            pools[split][key] = {**pool, 'images': images, 'split': split, 'source_split': 'train'}
    image_sets = {split: {image for pool in pools[split].values() for image in pool['images']} for split in pools}
    if image_sets['train'] & image_sets['val']:
        raise ValueError('A source image appears in multiple pools and crossed the holdout')
    paths = {}
    for split in ('train', 'val'):
        derived = copy.deepcopy(index)
        derived['split'] = split
        derived['objects'] = [objects[oid] for oid in sorted(selections[split])]
        derived['image_pools'] = pools[split]
        derived['audit'] = {'decision': 'GO', 'relation_ready': True,
                            'source_index_sha256': sha256_file(source), 'source_split': 'train',
                            'holdout_seed': 42, 'objects': len(derived['objects']),
                            'image_count': len(image_sets[split]), 'leakage_count': 0}
        paths[split] = str(source.with_name(f'{split}_holdout_seed42.json'))
        save_index(paths[split], derived)
    if set(selections['train']) & set(selections['val']):
        raise AssertionError('Object leakage')
    config['data'].update(index_paths=paths, validation_split='val', min_train_affordances=2)
    config['training'].update(epochs=10, batch_size=2, batch_size_objects=2)
    config['model'].update(selector_temperature=.2, selector_temperature_fixed=True, freeze_encoder_wrappers=True)
    config['paths'].update(checkpoint_dir='work/stage_a/controlled/checkpoints', log_dir='work/stage_a/controlled/logs')
    result = {'seed': 42, 'source_index': str(source), 'source_sha256': sha256_file(source),
              'official_test_used': False, 'per_category': counts,
              'objects': {s: len(ids) for s, ids in selections.items()},
              'images': {s: len(images) for s, images in image_sets.items()}, 'configs': {}}
    for model in ('fbd_afford', 'las'):
        cfg = copy.deepcopy(config)
        cfg['name'] = f'{model}_holdout_t02_10ep_seed42'
        cfg['model']['name'] = model
        cfg['data']['object_las_compat'] = model == 'las'
        if model == 'las':
            cfg['loss']['focal_weight'] = 1.0
        path = Path('configs') / f'piadv2_{model}_holdout_10ep.yaml'
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        result['configs'][model] = str(path)
    write_json('work/stage_a/controlled/split_report.json', result)
    print(yaml.safe_dump(result, sort_keys=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/piadv2_fbd_unseen_obj_stage_a.yaml')
    prepare(parser.parse_args().config)
