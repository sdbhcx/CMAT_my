"""Object padding and lossless LAS compatibility collation."""
import torch
from .multi_affordance_dataset import multi_affordance_collate


def flatten_object_batch(batch):
    """Repeat the SAME sampled points for each valid cue; never resample."""
    valid = batch['valid'].bool()
    object_positions = torch.arange(valid.shape[0], device=valid.device)[:, None].expand_as(valid)[valid]
    return {'points': batch['points'][object_positions], 'image': batch['images'][valid],
            'gt_mask': batch['masks'][valid].unsqueeze(-1),
            'affordance_id': batch['affordance_ids'][valid],
            'category_id': batch['category_ids'][object_positions],
            'object_ids': [batch['object_ids'][i] for i in object_positions.tolist()]}


def las_object_collate(samples):
    return flatten_object_batch(multi_affordance_collate(samples))
