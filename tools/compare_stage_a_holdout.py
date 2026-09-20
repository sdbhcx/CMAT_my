"""Evaluate saved Stage A models on validation only, without training."""
import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.indexed_affordance_dataset import ObjectAffordanceDataset
from data.multi_affordance_dataset import multi_affordance_collate
from data.multi_affordance_collate import flatten_object_batch
from models import create_model
from utils.metrics import compute_metrics


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


@torch.no_grad()
def run(args):
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    seed(42)
    model = create_model(config).cuda().eval()
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    del checkpoint['model_state_dict']
    data = config['data']
    dataset = ObjectAffordanceDataset(
        data['index_paths']['val'], config['paths']['data_root'],
        num_points=data['num_points'], max_affordances=4,
        image_size=data['image_size'], training=False, augment=False,
        seed=config.get('seed', 42))
    assert dataset.index['split'] == 'val'
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4,
                        collate_fn=multi_affordance_collate)
    fbd = config['model']['name'] == 'fbd_afford'
    predictions = {key: [] for key in ('original', 'swapped', 'alternate_rng')}
    targets, aids, cats, alphas, records, pair_gt, pair_pred = [], [], [], [], [], [], []
    point_sets, basis_sets, visual_images, visual_indices = [], [], [], []
    seen_categories = set()
    input_hash = hashlib.sha256()
    prompt_offset = 0
    for step, batch in enumerate(loader):
        for key in ('points', 'images', 'masks', 'valid', 'affordance_ids', 'category_ids'):
            input_hash.update(batch[key].numpy().tobytes())
        point_sets.append(batch['points'].numpy())
        for b, category in enumerate(batch['category_ids'].tolist()):
            if category not in seen_categories:
                seen_categories.add(category)
                for a in batch['valid'][b].nonzero().flatten().tolist():
                    visual_images.append(batch['images'][b, a].numpy())
                    visual_indices.append(prompt_offset + int(batch['valid'][:b].sum()) + a)
        prompt_offset += int(batch['valid'].sum())
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        valid = batch['valid']
        original = None
        for mode in predictions:
            altered = dict(batch)
            if mode == 'swapped':
                altered['images'] = batch['images'].clone()
                for b in range(len(valid)):
                    ids = valid[b].nonzero().flatten()
                    altered['images'][b, ids] = batch['images'][b, ids.roll(1)]
            seed(10000 + step + (100000 if mode == 'alternate_rng' else 0))
            output = model(altered if fbd else flatten_object_batch(altered))
            pred = output['segmentation_logits'].sigmoid()
            flat = pred[valid] if fbd else pred.squeeze(-1)
            assert torch.isfinite(flat).all()
            predictions[mode].append(flat.cpu().numpy())
            if mode == 'original':
                original = torch.zeros_like(batch['masks'])
                original[valid] = flat
                if fbd:
                    alphas.append(output['alpha'][valid].cpu().numpy())
                    basis_sets.append(output['basis_maps'].cpu().numpy())
        targets.append(batch['masks'][valid].cpu().numpy())
        aids.extend(batch['affordance_ids'][valid].cpu().tolist())
        cats.extend(batch['category_ids'][:, None].expand_as(valid)[valid].cpu().tolist())
        for b, oid in enumerate(batch['object_ids']):
            pairs = list(itertools.combinations(valid[b].nonzero().flatten().tolist(), 2))
            pd = [float((original[b, i]-original[b, j]).abs().mean()) for i, j in pairs]
            gd = [float((batch['masks'][b, i]-batch['masks'][b, j]).abs().mean()) for i, j in pairs]
            pair_pred.extend(pd)
            pair_gt.extend(gd)
            start = sum(len(t) for t in targets[:-1]) + int(valid[:b].sum())
            records.append({'object_id': oid, 'start': start, 'end': start+int(valid[b].sum()), 'category_id': int(batch['category_ids'][b]), 'prediction_pair_mae': float(np.mean(pd)) if pd else None,
                            'gt_pair_mae': float(np.mean(gd)) if gd else None})
        if (step+1) % 32 == 0:
            print(f'{config["model"]["name"]}: {step+1}/{len(loader)}', flush=True)
    arrays = {key: np.concatenate(value) for key, value in predictions.items()}
    gt = np.concatenate(targets)
    result = {'checkpoint': args.checkpoint, 'checkpoint_epoch': int(checkpoint['epoch'])+1,
              'objects': len(records), 'valid_prompts': len(gt),
              'parameters': sum(p.numel() for p in model.parameters()),
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'metrics': {key: compute_metrics(value, gt) for key, value in arrays.items()},
              'prediction_pair_mae': float(np.mean(pair_pred)), 'gt_pair_mae': float(np.mean(pair_gt)),
              'near_identical_objects': sum(r['prediction_pair_mae'] is not None and r['prediction_pair_mae'] < 1e-6 for r in records),
              'rng_prediction_mae': float(np.abs(arrays['original']-arrays['alternate_rng']).mean()),
              'cue_prediction_mae': float(np.abs(arrays['original']-arrays['swapped']).mean()),
              'per_object': records, 'input_sha256': input_hash.hexdigest(),
              'affordance_vocabulary': dataset.index['affordance_vocabulary'],
              'category_vocabulary': dataset.index['category_vocabulary']}
    result['low_differentiation_objects'] = sum(r['gt_pair_mae'] is not None and r['gt_pair_mae'] > .02 and r['prediction_pair_mae'] < .25*r['gt_pair_mae'] for r in records)
    result['distinct_gt_objects'] = sum(r['gt_pair_mae'] is not None and r['gt_pair_mae'] > .02 for r in records)
    for label, ids, vocabulary in [('affordance', aids, dataset.index['affordance_vocabulary']),
                                    ('category', cats, dataset.index['category_vocabulary'])]:
        result['per_'+label] = {str(vocabulary[i]): {'prompts': ids.count(i),
            **compute_metrics(arrays['original'][np.array(ids)==i], gt[np.array(ids)==i])}
            for i in sorted(set(ids))}
    if alphas:
        alpha = np.concatenate(alphas)
        result['selector'] = {'argmax_counts': np.bincount(alpha.argmax(-1), minlength=alpha.shape[-1]).tolist(),
                              'mean_weights': alpha.mean(0).tolist(),
                              'entropy': float((-(alpha*np.log(np.maximum(alpha, 1e-9))).sum(-1)/np.log(alpha.shape[-1])).mean())}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, default=float))
    extra = {'points': np.concatenate(point_sets), 'visual_images': np.stack(visual_images),
             'visual_indices': np.array(visual_indices)}
    if basis_sets:
        extra.update(basis_maps=np.concatenate(basis_sets), alpha=np.concatenate(alphas))
    np.savez_compressed(destination.with_suffix('.npz'), **arrays, targets=gt, affordance_ids=aids, category_ids=cats, **extra)
    print(json.dumps({k: v for k, v in result.items() if not k.startswith('per_')}, default=float), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
