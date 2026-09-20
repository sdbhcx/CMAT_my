"""Deterministic cue intervention and real-object Stage A memorization experiment.

Uses training objects only. Evaluation mode intentionally disables dropout while
autograd remains enabled during fitting; frozen backbones are still real encoders.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from data.indexed_affordance_dataset import ObjectAffordanceDataset
from data.multi_affordance_dataset import multi_affordance_collate
from data.object_affordance_index import write_json
from models import create_model, get_loss_function


def pair_distance(values, valid):
    pairs = [float((values[b, i] - values[b, j]).abs().mean())
             for b in range(len(valid))
             for i, j in itertools.combinations(valid[b].nonzero().flatten().tolist(), 2)]
    return float(np.mean(pairs)) if pairs else 0.0


def intervene(batch, mode):
    images = batch['images'].clone()
    permutation = torch.arange(images.shape[1], device=images.device).repeat(len(images), 1)
    for b in range(len(images)):
        ids = batch['valid'][b].nonzero().flatten()
        if mode == 'permute':
            permutation[b, ids] = ids.roll(1)
            images[b, ids] = batch['images'][b, ids.roll(1)]
        elif mode == 'same':
            images[b, ids] = batch['images'][b, ids[0]]
        elif mode == 'zero':
            images[b, ids] = 0  # Normalized mean-color image, not a missing token.
        else:
            raise ValueError(mode)
    return {**batch, 'images': images}, permutation


def gradients(model):
    groups = defaultdict(list)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            group = '.'.join(name.split('.')[:2]) if name.startswith('functional_basis_head.') else name.split('.')[0]
            groups[group].append(parameter.grad.detach().norm())
    result = {key: float(torch.stack(vals).norm()) for key, vals in groups.items()}
    if not result or not all(np.isfinite(v) for v in result.values()):
        raise RuntimeError('Missing or nonfinite gradients')
    return result


def fit_reference(predictions, masks, valid, criterion):
    """Soft-target diagnostic reference, not an attainable loss lower bound."""
    targets = masks.clamp(0, 1)
    union = (targets * valid[..., None]).amax(1)
    reference_outputs = {'segmentation_logits': torch.logit(targets.clamp(1e-6, 1-1e-6)),
                         'basis_maps': union[..., None]}
    reference_loss = criterion(reference_outputs, {'masks': targets, 'valid': valid})[0]
    # Best query-independent prediction for pointwise absolute error is a median.
    medians = torch.stack([targets[b, valid[b]].median(0).values for b in range(len(valid))])
    return {'soft_target_reference_loss': float(reference_loss),
            'note': 'Predicting soft GT exactly does not give zero focal/dice loss; this reference is not a loss lower bound.',
            'prediction_gt_mae': float((predictions-targets).abs()[valid].mean()),
            'best_shared_prediction_gt_mae': float((medians[:, None]-targets).abs()[valid].mean())}


def forward(model, batch):
    # FPS can consume random numbers even in eval mode. Hold its seed constant
    # across interventions so any prediction change is attributable to the cue.
    torch.manual_seed(42)
    return model(batch)


@torch.no_grad()
def diagnose(model, batch, criterion):
    model.eval()
    out = forward(model, batch)
    valid = batch['valid']
    pred = out['segmentation_logits'].sigmoid()
    alpha = out['alpha'][valid]
    desc = F.normalize(out['basis_descriptors'], dim=-1)
    cosine = desc @ desc.transpose(-1, -2)
    offdiag = ~torch.eye(cosine.shape[-1], dtype=torch.bool, device=cosine.device)
    result = {
        'loss': float(criterion(out, batch)[0]),
        'loss_components': {k: float(v) for k, v in criterion(out, batch)[1].items()},
        'prediction_pair_mae': pair_distance(pred, valid),
        'gt_pair_mae': pair_distance(batch['masks'], valid),
        'alpha_pair_mae': pair_distance(out['alpha'], valid),
        'query_pair_mae': pair_distance(out['query_tokens'], valid),
        'alpha_entropy_normalized': float((-(alpha * alpha.clamp_min(1e-9).log()).sum(-1) / np.log(alpha.shape[-1])).mean()) if alpha.shape[-1] > 1 else 0.0,
        'basis_descriptor_offdiag_cosine': float(cosine[:, offdiag].mean()) if alpha.shape[-1] > 1 else None,
        'basis_map_pair_mae': pair_distance(out['basis_maps'].transpose(1, 2), torch.ones(out['basis_maps'].shape[::2], dtype=torch.bool, device=valid.device)),
        'selector_temperature_effective': float(model.functional_basis_head.basis_selector.temperature.clamp_min(model.functional_basis_head.basis_selector.min_temperature)),
        'selected_basis_counts': torch.bincount(alpha.argmax(-1), minlength=alpha.shape[-1]).cpu().tolist(),
        'per_object': [{'prediction_pair_mae': pair_distance(pred[i:i+1], valid[i:i+1]),
                        'gt_pair_mae': pair_distance(batch['masks'][i:i+1], valid[i:i+1])}
                       for i in range(len(valid))],
    }
    for mode in ('permute', 'same', 'zero'):
        changed, permutation = intervene(batch, mode)
        altered = forward(model, changed)
        difference = (altered['segmentation_logits'].sigmoid() - pred).abs()[valid]
        result[mode] = {'prediction_mae_change': float(difference.mean()),
                        'loss': float(criterion(altered, batch)[0]),
                        'basis_max_change': float((altered['basis_logits'] - out['basis_logits']).abs().max())}
        torch.testing.assert_close(altered['basis_logits'], out['basis_logits'])
        if mode == 'permute':
            expected = out['segmentation_logits'].gather(1, permutation[:, :, None].expand_as(out['segmentation_logits']))
            torch.testing.assert_close(altered['segmentation_logits'], expected, atol=1e-5, rtol=1e-5)
            result[mode]['equivariance_max_error'] = float((altered['segmentation_logits'] - expected).abs().max())
        if mode == 'same':
            result[mode]['prediction_pair_mae'] = pair_distance(altered['segmentation_logits'].sigmoid(), valid)
    return result, out


def run(args):
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.set_num_threads(4)
    config = yaml.safe_load(Path(args.config).read_text())
    if config['loss']['stage'] != 'A' or config['model']['name'] != 'fbd_afford':
        raise ValueError('Requires Stage A functional basis configuration')
    root = Path(args.output_dir); root.mkdir(parents=True, exist_ok=True)
    dataset = ObjectAffordanceDataset(config['data']['index_paths']['train'], config['paths']['data_root'],
                                      training=True, augment=False, min_affordances=2, seed=42)
    by_category = defaultdict(list)
    for i, (position, _) in enumerate(dataset.items):
        by_category[dataset.index['objects'][position]['category']].append(i)
    samples, manifest = [], []
    # Round robin across categories; select distinguishable masks for this
    # diagnostic subset, not a representative benchmark or a test-set score.
    candidates = itertools.zip_longest(*(by_category[k] for k in sorted(by_category)))
    for row in candidates:
        for i in row:
            if i is None:
                continue
            sample = dataset[i]
            distance = pair_distance(sample['masks'][None], sample['valid'][None])
            if distance < .02:
                continue
            samples.append(sample)
            manifest.append({'dataset_position': i, 'object_id': sample['object_id'],
                             'category': dataset.index['category_vocabulary'][sample['category_id']],
                             'affordance_ids': sample['affordance_ids'].tolist(), 'gt_pair_mae': distance,
                             'image_tensor_sha256': hashlib.sha256(sample['images'].numpy().tobytes()).hexdigest()})
            if len(samples) == args.objects:
                break
        if len(samples) == args.objects:
            break
    if len(samples) != args.objects:
        raise RuntimeError('Insufficient distinguishable training objects')
    write_json(root / 'subset.json', {'seed': 42, 'split': 'train', 'augment': False, 'objects': manifest})
    batch = multi_affordance_collate(samples)
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    if args.fresh_basis_count is not None:
        config['model']['num_basis'] = args.fresh_basis_count
    model = create_model(config).cuda()
    state = torch.load(args.checkpoint, map_location='cpu')['model_state_dict']
    if args.fresh_basis_count is None:
        model.load_state_dict(state, strict=True)
    else:
        # Same pretrained trunk/projections, independently initialized heads.
        # Reset seed explicitly so constructor RNG consumption cannot confound it.
        from models.relational_affordance_model import RelationalAffordanceHead
        torch.manual_seed(42)
        cfg = config['model']
        model.functional_basis_head = RelationalAffordanceHead(
            model.unified_dim, model.unified_dim, cfg.get('basis_hidden_dim', 256),
            cfg.get('selector_hidden_dim', 256), args.fresh_basis_count,
            cfg.get('selector_temperature', .07)).cuda()
        common = {k: v for k, v in state.items() if not k.startswith('functional_basis_head.')}
        incompatible = model.load_state_dict(common, strict=False)
        if incompatible.unexpected_keys or any(not k.startswith('functional_basis_head.') for k in incompatible.missing_keys):
            raise RuntimeError('Unexpected trunk checkpoint mismatch')
    selector = model.functional_basis_head.basis_selector
    if args.temperature is not None:
        with torch.no_grad():
            selector.temperature.fill_(args.temperature)
    selector.temperature.requires_grad_(not args.fixed_temperature)
    if any(p.requires_grad for encoder in (model.point_encoder, model.prompt_encoder) for p in encoder.parameters()):
        raise RuntimeError('Diagnostic requires frozen backbones')
    criterion = get_loss_function(config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0)
    initial, _ = diagnose(model, batch, criterion)
    write_json(root / 'initial.json', initial)
    trace = []
    for step in range(args.steps):
        model.eval()  # Disable stochastic layers, retain autograd.
        start = (step * 4) % args.objects
        ids = torch.arange(start, min(start + 4, args.objects), device='cuda')
        minibatch = {k: v[ids] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss, _ = criterion(forward(model, minibatch), minibatch)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        norms = gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        trace.append({'step': step + 1, 'loss': float(loss.detach()), 'gradient_norms_before_clip': norms})
        if (step + 1) % 100 == 0:
            print(json.dumps(trace[-1]), flush=True)
            write_json(root / 'trace.json', trace)
    final, out = diagnose(model, batch, criterion)
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'config': config, 'steps': args.steps, 'source_checkpoint': args.checkpoint}, root / 'overfit.pth')
    np.savez_compressed(root / 'predictions.npz', points=batch['points'].cpu().numpy(),
                        masks=batch['masks'].cpu().numpy(), valid=batch['valid'].cpu().numpy(),
                        predictions=out['segmentation_logits'].sigmoid().cpu().numpy(),
                        alpha=out['alpha'].cpu().numpy(), basis_maps=out['basis_maps'].cpu().numpy())
    report = {'scope': 'fixed training subset only; not generalization performance',
              'intervention': {'temperature_override': args.temperature, 'fixed_temperature': args.fixed_temperature,
                               'fresh_basis_count': args.fresh_basis_count},
              'objects': args.objects, 'steps': args.steps, 'learning_rate': args.lr,
              'source_checkpoint': args.checkpoint, 'initial': initial, 'final': final,
              'relative_loss_reduction': 1 - final['loss'] / initial['loss'],
              'all_gradient_steps_finite': len(trace),
              'fit_reference': fit_reference(out['segmentation_logits'].sigmoid(), batch['masks'], batch['valid'], criterion),
              'gate_note': 'Conservative heuristic, not proof of convergence or a theoretical loss threshold.',
              'decision_rule': 'loss reduction >= 50%, prediction/GT pair MAE >= 0.25, shuffled loss increase >= 0.01'}
    report['ready_for_long_training'] = (report['relative_loss_reduction'] >= .5 and
        final['prediction_pair_mae'] >= .25 * final['gt_pair_mae'] and
        final['permute']['loss'] - final['loss'] >= .01)
    write_json(root / 'trace.json', trace); write_json(root / 'report.json', report)
    plot_results(batch, out, dataset.index, trace, root)
    print(json.dumps(report, indent=2))


def plot_results(batch, outputs, index, trace, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot([r['step'] for r in trace], [r['loss'] for r in trace], alpha=.5, linewidth=.6, label='Minibatch loss')
    means = [np.mean([r['loss'] for r in trace[i:i+4]]) for i in range(0, len(trace), 4)]
    ax.plot(range(1, len(trace)+1, 4), means, label='Four-batch mean')
    ax.set(xlabel='Optimizer step', ylabel='Stage A loss', title=f'Fixed {len(batch["points"])}-object training subset (not test performance)')
    ax.legend(); fig.tight_layout(); fig.savefig(root / 'loss_curve.png', dpi=130); plt.close(fig)
    row = int(batch['valid'].sum(-1).argmax())
    count = int(batch['valid'][row].sum())
    fig = plt.figure(figsize=(4*count, 9))
    xyz = batch['points'][row].cpu().numpy()
    for a in range(count):
        name = index['affordance_vocabulary'][int(batch['affordance_ids'][row, a])]
        cue = batch['images'][row, a].cpu().permute(1, 2, 0).numpy()
        cue = np.clip(cue * [.229, .224, .225] + [.485, .456, .406], 0, 1)
        ax = fig.add_subplot(3, count, a+1); ax.imshow(cue); ax.set_title(name); ax.axis('off')
        for r, values, label in ((1, batch['masks'][row, a], 'GT'),
                                 (2, outputs['segmentation_logits'][row, a].sigmoid(), 'Prediction')):
            ax = fig.add_subplot(3, count, r*count+a+1, projection='3d')
            ax.scatter(*xyz.T, c=values.cpu().numpy(), vmin=0, vmax=1, cmap='viridis', s=3)
            ax.set_title(label); ax.set_axis_off(); ax.set_box_aspect((1, 1, 1))
    fig.tight_layout(); fig.savefig(root / 'predictions.png', dpi=130); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--objects', type=int, default=16)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--fixed-temperature', action='store_true')
    parser.add_argument('--fresh-basis-count', type=int, help='Reinitialize only head for K ablation; common trunk restored')
    args = parser.parse_args()
    if args.objects < 4 or args.objects % 4 or args.steps < 1 or args.lr <= 0:
        parser.error('objects must be a positive multiple of 4; steps and lr must be positive')
    if args.temperature is not None and args.temperature < .02:
        parser.error('temperature must be >= selector floor 0.02')
    if args.fresh_basis_count is not None and args.fresh_basis_count < 1:
        parser.error('fresh-basis-count must be positive')
    run(args)
