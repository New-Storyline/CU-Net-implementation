
import torch

def L2_depth_loss(pred_depth, gd_depth):
    """
    L2^2 loss calculated only on valid pixels (where gd_depth > 0)
    """
    valid_mask = gd_depth > 0
    diff = pred_depth[valid_mask] - gd_depth[valid_mask]
    return torch.mean(diff ** 2)

def RMSE_depth_metric(pred_depth, gd_depth):
    """
    RMSE loss calculated only on valid pixels (where gd_depth > 0)
    """
    mse = L2_depth_loss(pred_depth, gd_depth)
    return torch.sqrt(mse)