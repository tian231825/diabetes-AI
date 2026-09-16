import torch

def calculate_metrics(pred, target, mask=None):
    """
    Calculate common regression metrics (MSE, MAE) considering masks.
    """
    if mask is not None:
        mask = mask.float()
        diff = pred - target
        mse = ((diff ** 2) * mask).sum() / mask.sum()
        mae = (torch.abs(diff) * mask).sum() / mask.sum()
    else:
        mse = torch.nn.functional.mse_loss(pred, target)
        mae = torch.nn.functional.l1_loss(pred, target)
    
    return {
        'mse': mse.item(),
        'mae': mae.item(),
        'rmse': torch.sqrt(mse).item()
    }