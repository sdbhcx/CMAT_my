import unittest

import torch

from models.functional_basis_generator import pool_basis_descriptors
from models.relational_affordance_model import (
    RelationalAffordanceHead,
    pool_prompt_tokens,
)


class FunctionalBasisHeadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.head = RelationalAffordanceHead(
            point_dim=12,
            query_dim=10,
            basis_hidden_dim=16,
            selector_hidden_dim=14,
            num_basis=4,
        )
        self.point_features = torch.randn(2, 9, 12, requires_grad=True)
        self.query_tokens = torch.randn(2, 3, 10, requires_grad=True)

    def test_output_contract_and_backward(self):
        valid = torch.tensor([[True, True, False], [True, False, False]])
        outputs = self.head(self.point_features, self.query_tokens, valid=valid)

        self.assertEqual(outputs["segmentation_logits"].shape, (2, 3, 9))
        self.assertEqual(outputs["basis_logits"].shape, (2, 9, 4))
        self.assertEqual(outputs["basis_maps"].shape, (2, 9, 4))
        self.assertEqual(outputs["basis_descriptors"].shape, (2, 4, 12))
        self.assertEqual(outputs["alpha"].shape, (2, 3, 4))
        self.assertTrue(torch.isfinite(outputs["segmentation_logits"]).all())

        self.assertTrue(torch.equal(outputs["alpha"][0, 2], torch.zeros(4)))
        self.assertTrue(
            torch.equal(outputs["segmentation_logits"][0, 2], torch.zeros(9))
        )
        valid_alpha_sums = outputs["alpha"].sum(dim=-1)[valid]
        self.assertTrue(torch.allclose(valid_alpha_sums, torch.ones_like(valid_alpha_sums)))

        outputs["segmentation_logits"][valid].sum().backward()
        self.assertIsNotNone(self.point_features.grad)
        self.assertIsNotNone(self.query_tokens.grad)
        self.assertTrue(torch.isfinite(self.point_features.grad).all())
        self.assertTrue(torch.isfinite(self.query_tokens.grad).all())

    def test_basis_is_query_independent_and_cues_are_permutation_equivariant(self):
        outputs = self.head(self.point_features, self.query_tokens)
        permutation = torch.tensor([2, 0, 1])
        permuted = self.head(self.point_features, self.query_tokens[:, permutation])

        self.assertTrue(torch.equal(outputs["basis_logits"], permuted["basis_logits"]))
        self.assertTrue(torch.equal(outputs["basis_maps"], permuted["basis_maps"]))
        self.assertTrue(
            torch.allclose(outputs["alpha"][:, permutation], permuted["alpha"])
        )
        self.assertTrue(
            torch.allclose(
                outputs["segmentation_logits"][:, permutation],
                permuted["segmentation_logits"],
            )
        )

    def test_zero_basis_weights_are_finite(self):
        point_features = torch.randn(2, 5, 3)
        basis_maps = torch.zeros(2, 5, 4)
        descriptors = pool_basis_descriptors(point_features, basis_maps)

        self.assertTrue(torch.isfinite(descriptors).all())
        self.assertTrue(torch.equal(descriptors, torch.zeros_like(descriptors)))

    def test_masked_prompt_pooling(self):
        features = torch.tensor(
            [[[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]]]
        )
        mask = torch.tensor([[[True, True, False]]])
        pooled = pool_prompt_tokens(features, mask)
        self.assertTrue(torch.equal(pooled, torch.tensor([[[2.0, 4.0]]])))

        empty = pool_prompt_tokens(features, torch.zeros_like(mask))
        self.assertTrue(torch.equal(empty, torch.zeros_like(empty)))


if __name__ == "__main__":
    unittest.main()
