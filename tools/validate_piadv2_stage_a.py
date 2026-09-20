"""Exercise the real indexed Stage-A trainer, checkpoint restore and data plots."""
from __future__ import annotations

import argparse
import copy
from itertools import islice
import random
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from data.indexed_affordance_dataset import ObjectAffordanceDataset
from data.multi_affordance_dataset import multi_affordance_collate
from data.object_affordance_index import write_json
from models import create_model, get_loss_function
from train import UnifiedTrainer


class LimitedLoader:
    def __init__(self, loader, limit):
        self.loader, self.limit = loader, limit
        self.dataset = loader.dataset

    def __len__(self):
        return min(len(self.loader), self.limit)

    def __iter__(self):
        return islice(self.loader, self.limit)


class CheckedTrainer(UnifiedTrainer):
    def __init__(self, config):
        self.trace, self.gradient_norms = [], []
        super().__init__(config)
        self.optimizer.register_step_pre_hook(self._check_gradients)

    def _check_gradients(self, optimizer, args, kwargs):
        gradients = [p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
        if not gradients:
            raise RuntimeError('No gradients reached the optimizer')
        norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in gradients]))
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite gradients')
        self.gradient_norms.append(float(norm))

    def _compute_batch_loss(self, outputs, batch):
        total, components = super()._compute_batch_loss(outputs, batch)
        if not torch.isfinite(total) or not all(torch.isfinite(v).all() for v in components.values()):
            raise RuntimeError('Nonfinite loss')
        self.trace.append({'phase': 'train' if self.model.training else 'validation',
                           'valid_affordances': int(batch['valid'].sum()),
                           'objects': len(batch['points']),
                           **{key: float(value.detach()) for key, value in components.items()}})
        return total, components

    def train_epoch(self):
        self.train_stats = super().train_epoch()
        return self.train_stats

    def validate(self):
        self.val_stats = super().validate()
        return self.val_stats


def data_samples(config):
    data = config['data']
    dataset = ObjectAffordanceDataset(data['index_paths']['train'], config['paths']['data_root'],
                                     num_points=data['num_points'], max_affordances=4,
                                     image_size=data['image_size'], training=True, augment=False,
                                     min_affordances=2, seed=config.get('seed', 42))
    samples = []
    for count in (2, 3, 4):
        i = next(i for i, (pos, _) in enumerate(dataset.items)
                 if min(len(dataset.index['objects'][pos]['affordances']), 4) == count)
        sample = dataset[i]
        # Re-fetch checks point, mask and cue determinism on actual sources.
        repeat = dataset[i]
        for key in ('points', 'masks', 'images', 'sample_indices'):
            torch.testing.assert_close(sample[key], repeat[key])
        samples.append(sample)
    batch = multi_affordance_collate(samples)
    assert batch['valid'].sum().item() == 9
    assert torch.all(batch['affordance_ids'][~batch['valid']] == -1)
    assert torch.all(batch['images'][~batch['valid']] == 0)
    assert torch.all(batch['masks'][~batch['valid']] == 0)
    return batch, dataset.index


def plot_sample(batch, outputs, index, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    row = 2  # Four-affordance real object selected by data_samples.
    valid = batch['valid'][row].cpu().numpy()
    count = int(valid.sum())
    points = batch['points'][row].cpu().numpy()
    masks = batch['masks'][row].cpu().numpy()
    pred = outputs['segmentation_logits'][row].sigmoid().cpu().numpy()
    images = batch['images'][row].cpu().permute(0, 2, 3, 1).numpy()
    images = np.clip(images * np.array([.229, .224, .225]) + np.array([.485, .456, .406]), 0, 1)
    figure = plt.figure(figsize=(4 * count, 11))
    category = index['category_vocabulary'][int(batch['category_ids'][row])]
    for a in range(count):
        name = index['affordance_vocabulary'][int(batch['affordance_ids'][row, a])]
        ax = figure.add_subplot(3, count, a + 1)
        ax.imshow(images[a]); ax.set_title(f'{category} / {name}: cue'); ax.axis('off')
        for r, values, title in ((1, masks[a], 'Ground truth'), (2, pred[a], 'Prediction')):
            ax = figure.add_subplot(3, count, r * count + a + 1, projection='3d')
            ax.scatter(*points.T, c=values, vmin=0, vmax=1, cmap='viridis', s=3)
            ax.set_title(title); ax.set_axis_off(); ax.set_box_aspect((1, 1, 1))
    figure.suptitle('PIADv2 Unseen_obj: one point cloud, four functional masks', fontsize=16)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def run(config_path, output_dir, short=False):
    config = yaml.safe_load(Path(config_path).read_text())
    config = copy.deepcopy(config)
    if config['model']['name'] != 'fbd_afford' or config['loss']['stage'] != 'A':
        raise ValueError('This validator requires fbd_afford Stage A')
    config['training']['epochs'] = 1
    if short:
        config['name'] += '_short'
        config['validation_scope'] = {'train_batches': 32, 'test_batches': 8}
    else:
        config['validation_scope'] = {'train_batches': 'full selected dataset epoch', 'test_batches': 'all'}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get('seed', 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    qa_batch, index = data_samples(config)
    started = time.monotonic()
    trainer = CheckedTrainer(config)
    if any(p.requires_grad for p in trainer.model.point_encoder.parameters()):
        raise RuntimeError('Point encoder must be frozen for Stage A validation')
    if any(p.requires_grad for p in trainer.model.prompt_encoder.parameters()):
        raise RuntimeError('Prompt encoder must be frozen for Stage A validation')
    if short:
        trainer.train_loader = LimitedLoader(trainer.train_loader, 32)
        trainer.val_loader = LimitedLoader(trainer.val_loader, 8)
    trainer.train()
    checkpoint_path = Path(trainer.exp_dir) / 'checkpoint.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    restored = create_model(config).to(trainer.device)
    restored.load_state_dict(checkpoint['model_state_dict'], strict=True)
    trainer.model.eval(); restored.eval()
    qa_batch = {k: v.to(trainer.device) if isinstance(v, torch.Tensor) else v for k, v in qa_batch.items()}
    with torch.no_grad():
        torch.manual_seed(seed)
        original = trainer.model(qa_batch)
        torch.manual_seed(seed)
        reloaded = restored(qa_batch)
    torch.testing.assert_close(original['segmentation_logits'], reloaded['segmentation_logits'], rtol=1e-5, atol=1e-6)
    restored_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad],
                                          lr=float(config['training']['head_lr']))
    restored_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    restored.train()
    resumed_loss, _ = get_loss_function(config)(restored(qa_batch), qa_batch)
    resumed_loss.backward()
    if not torch.isfinite(resumed_loss):
        raise RuntimeError('Restored model produced a nonfinite loss')
    restored_optimizer.step()
    plot_sample(qa_batch, reloaded, index, output_dir / 'real_object_gt_predictions.png')
    train_rows = [r for r in trainer.trace if r['phase'] == 'train']
    report = {'config': str(config_path), 'scope': config['validation_scope'],
              'elapsed_seconds': time.monotonic() - started,
              'selected_train_objects': len(trainer.train_loader.dataset),
              'train_batches': len(train_rows), 'train_object_visits': sum(r['objects'] for r in train_rows),
              'validation_batches': sum(r['phase'] == 'validation' for r in trainer.trace),
              'train_loss': trainer.train_stats[0], 'validation_loss': trainer.val_stats[0],
              'validation_metrics': {key: float(value) for key, value in trainer.val_stats[3].items()},
              'first_10_train_loss_mean': float(np.mean([r['total_loss'] for r in train_rows[:10]])),
              'last_10_train_loss_mean': float(np.mean([r['total_loss'] for r in train_rows[-10:]])),
              'gradient_steps_finite': len(trainer.gradient_norms),
              'maximum_gradient_norm_after_clip': max(trainer.gradient_norms),
              'checkpoint': str(checkpoint_path), 'checkpoint_predictions_match': True,
              'optimizer_restore_step_passed': True, 'point_encoder_frozen': True, 'prompt_encoder_frozen': True,
              'qa_batch_shapes': {k: list(qa_batch[k].shape) for k in ('points', 'images', 'masks', 'valid')}}
    write_json(output_dir / 'report.json', report)
    write_json(output_dir / 'loss_trace.json', trainer.trace)
    trainer.writer.close()
    import json
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--short', action='store_true')
    args = parser.parse_args()
    run(args.config, args.output_dir, args.short)
