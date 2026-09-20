"""Check LAS objective/gradient parity through the padded FBD adapter."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from losses.functional_basis_loss import FunctionalBasisLoss
from models.las_model import LASLoss
from models.model_factory import get_loss_function
from tools.run_stage_a_controlled import restore_training


class FBDLASSegmentationTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(17)
        logits = torch.randn(2, 3, 11, requires_grad=True)
        basis = torch.randn(2, 11, 4, requires_grad=True)
        batch = {'masks': torch.rand(2, 3, 11),
                 'valid': torch.tensor([[True, False, False], [True, True, True]])}
        return logits, basis, batch

    def test_matches_las_soft_binary_losses_and_gradients(self):
        for binary in (False, True):
            with self.subTest(binary=binary):
                logits, basis, batch = self.fixture()
                if binary:
                    batch['masks'] = (batch['masks'] > .5).float()
                out = {'segmentation_logits': logits, 'basis_maps': basis.sigmoid()}
                criterion = FunctionalBasisLoss(segmentation_loss='las', union_weight=0,
                                               focal_weight=.7, dice_weight=1.3)
                total, parts = criterion(out, batch)
                flat = logits.detach()[batch['valid']].unsqueeze(-1).requires_grad_()
                ref, reference = LASLoss(focal_weight=.7, dice_weight=1.3)(
                    flat, batch['masks'][batch['valid']].unsqueeze(-1))
                torch.testing.assert_close(total, ref, rtol=0, atol=0)
                torch.testing.assert_close(parts['focal'], reference['focal_loss'], rtol=0, atol=0)
                torch.testing.assert_close(parts['dice'], reference['dice_loss'], rtol=0, atol=0)
                total.backward(); ref.backward()
                torch.testing.assert_close(logits.grad[batch['valid']], flat.grad.squeeze(-1), rtol=0, atol=0)
                self.assertEqual(logits.grad[~batch['valid']].abs().sum().item(), 0)

    def test_union_preserved_and_padding_does_not_change_loss(self):
        logits, basis, batch = self.fixture()
        out = {'segmentation_logits': logits, 'basis_maps': basis.sigmoid()}
        criterion = FunctionalBasisLoss(segmentation_loss='las')
        total, parts = criterion(out, batch)
        _, old = FunctionalBasisLoss()(out, batch)
        torch.testing.assert_close(parts['union'], old['union'], rtol=0, atol=0)
        torch.testing.assert_close(total, parts['segmentation'] + .2*parts['union'])
        altered = {**batch, 'masks': batch['masks'].clone()}
        altered['masks'][~batch['valid']] = 1
        altered_logits = logits.detach().clone()
        altered_logits[~batch['valid']] = 100
        changed, _ = criterion({**out, 'segmentation_logits': altered_logits}, altered)
        torch.testing.assert_close(total, changed, rtol=0, atol=0)
        total.backward()
        self.assertTrue(torch.isfinite(basis.grad).all())
        self.assertGreater(basis.grad.abs().sum().item(), 0)

    def test_factory_selects_las_and_preserves_legacy_weight_behavior(self):
        logits, basis, batch = self.fixture()
        out = {'segmentation_logits': logits, 'basis_maps': basis.sigmoid()}
        config = {'model': {'name': 'fbd_afford'}, 'loss': {'focal_weight': .5}}
        old = get_loss_function(config)
        self.assertEqual(old.segmentation_loss, 'fbd')
        torch.testing.assert_close(old(out,batch)[0], FunctionalBasisLoss()(out,batch)[0], rtol=0, atol=0)
        config['loss'].update(segmentation_loss='las', focal_weight=1., dice_weight=1.)
        selected = get_loss_function(config)
        self.assertIsInstance(selected.las_segmentation, LASLoss)
        self.assertEqual(selected.union_weight, .2)

    def test_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, 'segmentation_loss'):
            FunctionalBasisLoss(segmentation_loss='typo')

    def test_resume_rejects_changed_loss_before_loading_weights(self):
        config = {'model': {}, 'data': {}, 'loss': {'segmentation_loss': 'fbd'}}
        changed = copy.deepcopy(config)
        changed['loss']['segmentation_loss'] = 'las'
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder)/'checkpoint.pth'
            torch.save({'config': config}, checkpoint)
            with self.assertRaisesRegex(ValueError, 'loss'):
                restore_training(SimpleNamespace(config=changed), checkpoint)


if __name__ == '__main__':
    unittest.main()
