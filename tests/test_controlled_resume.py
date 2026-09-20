import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from tools.run_stage_a_controlled import restore_training


class ControlledResumeTests(unittest.TestCase):
    def test_restores_adam_and_starts_remaining_twenty_epoch_schedule(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(source.parameters(), lr=1e-4)
            source(torch.ones(1, 2)).sum().backward(); optimizer.step()
            config = {'model': {'name': 'test'}, 'data': {'validation_split': 'val'},
                      'seed': 42, 'training': {'epochs': 10}}
            checkpoint = {'epoch': 9, 'model_state_dict': source.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(), 'config': config,
                          'best_val_aiou': .15, 'best_val_loss': .4}
            path = root / 'checkpoint.pth'
            torch.save(checkpoint, path); torch.save(checkpoint, root / 'checkpoint_best.pth')
            target = torch.nn.Linear(2, 1)
            target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-4)
            destination = root / 'new'; destination.mkdir()
            new_config = copy.deepcopy(config); new_config['training']['epochs'] = 30
            trainer = SimpleNamespace(config=new_config, model=target, optimizer=target_optimizer,
                scheduler=None, exp_dir=str(destination), train_loader=SimpleNamespace(generator=torch.Generator()),
                history=[], save_status=lambda *args, **kwargs: None)
            restore_training(trainer, str(path), restart_lr=5e-5)
            self.assertEqual(list(range(trainer.epoch, 30)), list(range(10, 30)))
            self.assertEqual(trainer.scheduler.T_max, 20)
            self.assertAlmostEqual(trainer.optimizer.param_groups[0]['lr'], 5e-5)
            for old, new in zip(source.parameters(), target.parameters()):
                torch.testing.assert_close(old, new)
                torch.testing.assert_close(optimizer.state[old]['exp_avg'], trainer.optimizer.state[new]['exp_avg'])
                self.assertEqual(trainer.optimizer.state[new]['step'].item(), 1)
            self.assertTrue((destination / 'checkpoint_best.pth').exists())
            for _ in range(20):
                trainer.optimizer.step(); trainer.scheduler.step()
            self.assertAlmostEqual(trainer.optimizer.param_groups[0]['lr'], 1e-6)
            trainer.config['data']['validation_split'] = 'test'
            with self.assertRaisesRegex(ValueError, 'data'):
                restore_training(trainer, str(path), restart_lr=5e-5)


if __name__ == '__main__':
    unittest.main()
