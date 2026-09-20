import unittest
import torch
from models.functional_basis_model import FunctionalBasisAffordanceModel
from models.relational_affordance_model import RelationalAffordanceHead


class TinyModel(FunctionalBasisAffordanceModel):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.point_projection = torch.nn.Linear(3, 6)
        self.prompt_projection = torch.nn.Linear(3, 6)
        self.functional_basis_head = RelationalAffordanceHead(6, 6, 8, 8, 4)
        self.point_calls, self.image_count = 0, 0

    def encode_points(self, points):
        self.point_calls += 1
        return self.point_projection(points)

    def encode_prompts(self, images):
        self.image_count = len(images)
        tokens = self.prompt_projection(images.mean((-1, -2)))[:, None]
        return tokens, torch.ones(tokens.shape[:2], dtype=torch.bool, device=images.device)


class ValidCueTest(unittest.TestCase):
    def test_only_valid_images_encoded_and_padding_cannot_contaminate(self):
        model = TinyModel()
        batch = {'points': torch.randn(2, 9, 3), 'images': torch.randn(2, 3, 3, 4, 4),
                 'valid': torch.tensor([[True, True, False], [True, False, False]])}
        output = model(batch)
        self.assertEqual(model.point_calls, 1)
        self.assertEqual(model.image_count, 3)
        batch['images'][~batch['valid']] = float('nan')
        padded = model(batch)
        torch.testing.assert_close(output['segmentation_logits'], padded['segmentation_logits'])
        torch.testing.assert_close(output['basis_logits'], padded['basis_logits'])
        padded['segmentation_logits'][batch['valid']].sum().backward()
        self.assertTrue(torch.isfinite(model.prompt_projection.weight.grad).all())


if __name__ == '__main__':
    unittest.main()
