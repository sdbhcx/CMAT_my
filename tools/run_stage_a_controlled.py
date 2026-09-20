"""Seeded official-trainer entry point with fixed-temperature checks and status."""
import argparse
import json
from pathlib import Path
import random
import shutil
import traceback

import numpy as np
import torch
import yaml

from data.object_affordance_index import write_json
from train import UnifiedTrainer


class ControlledTrainer(UnifiedTrainer):
    def __init__(self, config, status_path):
        self.status_path = status_path
        self.history = []
        self.validation_diagnostics = []
        super().__init__(config)
        for encoder in (self.model.point_encoder, self.model.prompt_encoder):
            if any(p.requires_grad for p in encoder.parameters()):
                raise RuntimeError('Expected fully frozen backbone wrappers')
        if self.model_name == 'fbd_afford':
            parameter = self.model.functional_basis_head.basis_selector.temperature
            if parameter.requires_grad or abs(parameter.item()-.2) > 1e-6:
                raise RuntimeError('Selector temperature must be fixed at 0.2')
            if any(parameter is p for group in self.optimizer.param_groups for p in group['params']):
                raise RuntimeError('Frozen temperature appeared in optimizer')
        self.optimizer.register_step_pre_hook(self.check_gradients)
        self.save_status('running')

    def check_gradients(self, optimizer, args, kwargs):
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for group in optimizer.param_groups for p in group['params']):
            raise RuntimeError('Nonfinite gradient')

    def save_status(self, status, **extra):
        write_json(self.status_path, {'status': status, 'experiment_dir': self.exp_dir,
                   'model': self.model_name, 'epochs_requested': self.config['training']['epochs'],
                   'train_objects': len(self.train_loader.dataset), 'validation_samples': len(self.val_loader.dataset),
                   'history': self.history, **extra})

    def _compute_batch_loss(self, outputs, batch):
        result = super()._compute_batch_loss(outputs, batch)
        if not torch.isfinite(result[0]):
            raise RuntimeError('Nonfinite loss')
        if not self.model.training and self.model_name == 'fbd_afford':
            alpha = outputs['alpha'].detach()[batch['valid']]
            self.validation_diagnostics.append(alpha.cpu())
        return result

    def train_epoch(self):
        self.train_stats = super().train_epoch()
        return self.train_stats

    def validate(self):
        self.validation_diagnostics = []
        result = super().validate()
        row = {'epoch': self.epoch+1, 'train_loss': float(self.train_stats[0]),
               'learning_rates': [group['lr'] for group in self.optimizer.param_groups],
               'validation_loss': float(result[0]), 'metrics': {k: float(v) for k,v in result[3].items()}}
        if self.validation_diagnostics:
            alpha = torch.cat(self.validation_diagnostics)
            row['basis_argmax_counts'] = torch.bincount(alpha.argmax(-1), minlength=alpha.shape[-1]).tolist()
            row['selector_entropy'] = float((-(alpha*alpha.clamp_min(1e-9).log()).sum(-1)/np.log(alpha.shape[-1])).mean())
            row['temperature'] = float(self.model.functional_basis_head.basis_selector.temperature.detach())
        self.history.append(row)
        self.save_status('running')
        return result


def restore_training(trainer, path, restart_lr=None):
    checkpoint = torch.load(path, map_location='cpu')
    for key in ('model', 'data', 'seed', 'dataset_type', 'setting_type'):
        if checkpoint['config'].get(key) != trainer.config.get(key):
            raise ValueError(f'Resume configuration mismatch: {key}')
    start = int(checkpoint['epoch']) + 1
    remaining = trainer.config['training']['epochs'] - start
    if remaining <= 0:
        raise ValueError('Requested total epochs must exceed the completed checkpoint epoch')
    trainer.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if trainer.scheduler and checkpoint.get('scheduler_state_dict'):
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if restart_lr is not None:
        for group in trainer.optimizer.param_groups:
            group['lr'] = group['initial_lr'] = restart_lr
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=remaining, eta_min=1e-6)
    trainer.epoch = start
    trainer.best_val_aiou = checkpoint['best_val_aiou']
    trainer.best_val_loss = checkpoint['best_val_loss']
    # Carry the old best checkpoint forward if later epochs do not beat it.
    best = Path(path).with_name('checkpoint_best.pth')
    if not best.exists():
        raise FileNotFoundError(best)
    shutil.copy2(best, Path(trainer.exp_dir) / best.name)
    previous = trainer.config.get('previous_status')
    if previous:
        prior = json.loads(Path(previous).read_text())
        trainer.history = [row for row in prior['history'] if row['epoch'] <= start]
    # Old checkpoints did not save RNG state. Use a documented restart seed,
    # rather than pretending this is bitwise-identical uninterrupted training.
    seed = int(trainer.config.get('seed', 42)) + start
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if trainer.train_loader.generator is not None:
        trainer.train_loader.generator.manual_seed(seed)
    info = {'checkpoint': str(path), 'completed_epochs': start,
            'next_epoch': start+1, 'remaining_epochs': remaining,
            'restart_lr': restart_lr, 'optimizer_state_entries': len(trainer.optimizer.state),
            'rng_state_restored': False, 'restart_seed': seed}
    write_json(Path(trainer.exp_dir) / 'resume.json', info)
    print('RESUME ' + json.dumps(info), flush=True)
    trainer.save_status('running', resume=info)


def run(args):
    config = yaml.safe_load(Path(args.config).read_text())
    seed = config.get('seed', 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(4)
    trainer = None
    try:
        trainer = ControlledTrainer(config, args.status)
        if args.resume:
            restore_training(trainer, args.resume, args.restart_lr)
        if args.preflight:
            # Use the largest supported A so memory checks do not depend on the first shuffled batch.
            dataset = trainer.train_loader.dataset
            positions = sorted(range(len(dataset.items)), key=lambda i: len(dataset.index['objects'][dataset.items[i][0]]['affordances']), reverse=True)
            batch = trainer.train_loader.collate_fn([dataset[i] for i in positions[:config['training']['batch_size_objects']]])
            batch = {k: v.to(trainer.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
            trainer.model.train()
            loss, _ = trainer._compute_batch_loss(trainer.model(batch), batch)
            loss.backward(); trainer.optimizer.step(); trainer.optimizer.zero_grad(set_to_none=True)
            trainer.save_status('preflight_passed', loss=float(loss.detach()), peak_cuda_bytes=torch.cuda.max_memory_allocated())
        else:
            trainer.train()
            trainer.save_status('completed')
    except Exception:
        if trainer is not None:
            trainer.save_status('failed', error=traceback.format_exc())
        else:
            write_json(args.status, {'status': 'failed', 'error': traceback.format_exc()})
        raise
    finally:
        if trainer is not None:
            trainer.writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--status', required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume')
    parser.add_argument('--restart-lr', type=float)
    args = parser.parse_args()
    if args.restart_lr is not None and (not args.resume or args.restart_lr <= 1e-6):
        parser.error('--restart-lr requires --resume and must exceed 1e-6')
    run(args)
