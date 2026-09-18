import unittest

import torch

from losses.functional_basis_loss import FunctionalBasisLoss


class FunctionalBasisLossTest(unittest.TestCase):
    def test_loss_masks_padding_and_backpropagates(self):
        logits = torch.randn(2, 2, 5, requires_grad=True)
        basis_logits = torch.randn(2, 5, 3, requires_grad=True)
        outputs = {
            "segmentation_logits": logits,
            "basis_maps": torch.sigmoid(basis_logits),
        }
        batch = {
            "masks": torch.randint(0, 2, (2, 2, 5)).float(),
            "valid": torch.tensor([[True, False], [True, True]]),
        }

        criterion = FunctionalBasisLoss()
        total, losses = criterion(outputs, batch)
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(
            set(losses),
            {
                "total_loss",
                "segmentation",
                "focal",
                "dice",
                "union",
                "union_bce",
                "union_dice",
            },
        )
        total.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isfinite(basis_logits.grad).all())
        self.assertTrue(torch.equal(logits.grad[0, 1], torch.zeros(5)))


if __name__ == "__main__":
    unittest.main()
