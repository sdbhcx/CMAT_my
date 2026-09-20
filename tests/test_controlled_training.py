import unittest

import torch

from models.las_model import LASModel, DenseSegmentationHead
from tools.prepare_stage_a_controlled import holdout


class ControlledTrainingTests(unittest.TestCase):
    def test_holdout_is_disjoint_complete_and_order_independent(self):
        values = [str(i) for i in range(100)]
        train, val = holdout(values, 42)
        self.assertEqual((len(train), len(val)), (90, 10))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train) | set(val), set(values))
        self.assertEqual((train, val), holdout(list(reversed(values)), 42))
        with self.assertRaises(ValueError):
            holdout(['only'], 42)

    def test_freeze_mode_keeps_encoders_eval_and_head_trainable(self):
        model = LASModel.__new__(LASModel)
        torch.nn.Module.__init__(model)
        model.model_config = {'freeze_encoder_wrappers': True}
        model.point_encoder = torch.nn.Dropout()
        model.prompt_encoder = torch.nn.Dropout()
        model.segmentation_head = DenseSegmentationHead(input_dim=8, hidden_dim=4, num_layers=2, dropout=.1)
        model.train()
        self.assertFalse(model.point_encoder.training)
        self.assertFalse(model.prompt_encoder.training)
        self.assertTrue(model.segmentation_head.training)
        model.eval()
        self.assertFalse(model.segmentation_head.training)


if __name__ == '__main__':
    unittest.main()
