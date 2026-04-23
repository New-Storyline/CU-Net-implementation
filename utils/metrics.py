

import torch


def L2_depth_loss(pred_depth, gd_depth):
    """
    L2^2 loss calculated only on valid pixels (where gd_depth > 0)
    """
    valid_mask = gd_depth > 0
    diff = pred_depth[valid_mask] - gd_depth[valid_mask]
    return torch.mean(diff ** 2)