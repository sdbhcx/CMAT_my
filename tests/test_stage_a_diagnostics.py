import unittest

import torch

from tools.diagnose_piadv2_stage_a import diagnose, fit_reference, gradients, intervene, pair_distance
from losses.functional_basis_loss import FunctionalBasisLoss
from models.relational_affordance_model import RelationalAffordanceHead


class StageADiagnosticsTests(unittest.TestCase):
    def test_single_basis_diagnostics_are_finite_and_query_independent(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.functional_basis_head = RelationalAffordanceHead(3, 3, 4, 4, num_basis=1)

            def forward(self, batch):
                return self.functional_basis_head(batch['points'], batch['images'].mean((-1, -2)), batch['valid'])

        batch = {'points': torch.randn(2, 5, 3), 'images': torch.randn(2, 2, 3, 2, 2),
                 'masks': torch.rand(2, 2, 5), 'valid': torch.ones(2, 2, dtype=torch.bool)}
        result, outputs = diagnose(TinyModel(), batch, FunctionalBasisLoss())
        self.assertEqual(result['alpha_entropy_normalized'], 0.)
        self.assertIsNone(result['basis_descriptor_offdiag_cosine'])
        self.assertEqual(result['prediction_pair_mae'], 0.)
        self.assertEqual(result['permute']['prediction_mae_change'], 0.)
        torch.testing.assert_close(outputs['alpha'], torch.ones(2, 2, 1))

    def test_intervention_moves_only_valid_cues_and_preserves_targets(self):
        batch = {'images': torch.arange(8.).reshape(2, 4, 1, 1, 1),
                 'valid': torch.tensor([[True, True, False, False], [True, True, True, False]]),
                 'masks': torch.randn(2, 4, 5)}
        original = batch['images'].clone()
        swapped, permutation = intervene(batch, 'permute')
        self.assertEqual(permutation.tolist(), [[1, 0, 2, 3], [2, 0, 1, 3]])
        torch.testing.assert_close(swapped['images'].flatten(), torch.tensor([1., 0., 2., 3., 6., 4., 5., 7.]))
        torch.testing.assert_close(batch['images'], original)
        self.assertIs(swapped['masks'], batch['masks'])
        for mode in ('same', 'zero'):
            changed, _ = intervene(batch, mode)
            torch.testing.assert_close(changed['images'][~batch['valid']], original[~batch['valid']])
            self.assertEqual(pair_distance(changed['images'], batch['valid']), 0.)

    def test_pair_distance_excludes_padding(self):
        values = torch.tensor([[[0., 0.], [1., 1.], [float('nan'), float('nan')]]])
        self.assertEqual(pair_distance(values, torch.tensor([[True, True, False]])), 1.)

    def test_soft_target_reference_is_not_zero_loss(self):
        masks = torch.tensor([[[.2, .4], [.8, .6], [1., 1.]]])
        valid = torch.tensor([[True, True, False]])
        result = fit_reference(masks, masks, valid, FunctionalBasisLoss())
        self.assertGreater(result['soft_target_reference_loss'], 0.)
        self.assertEqual(result['prediction_gt_mae'], 0.)
        self.assertAlmostEqual(result['best_shared_prediction_gt_mae'], .2, places=6)

    def test_gradient_check_rejects_nonfinite_and_missing(self):
        model = torch.nn.Linear(2, 1)
        with self.assertRaises(RuntimeError):
            gradients(model)
        model(torch.ones(1, 2)).sum().backward()
        self.assertTrue(all(value > 0 for value in gradients(model).values()))
        model.weight.grad[0, 0] = float('nan')
        with self.assertRaises(RuntimeError):
            gradients(model)


if __name__ == '__main__':
    unittest.main()
