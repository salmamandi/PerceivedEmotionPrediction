import torch

def compute_loss(pred_v, pred_a, gt_v, gt_a, delta):

    mse_v = torch.mean((pred_v - gt_v) ** 2)
    mse_a = torch.mean((pred_a - gt_a) ** 2)

    transition_loss = torch.mean(delta ** 2)

    loss = mse_v + mse_a + 0.01 * transition_loss

    return loss

def compute_loss_task(pred, gt, delta):
    
    mse = torch.mean((pred - gt) ** 2)
    transition_loss = torch.mean(delta ** 2)
    return mse + 0.01 * transition_loss