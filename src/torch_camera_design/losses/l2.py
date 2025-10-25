from __future__ import annotations

import torch

__all__ = ["l2_loss"]

def l2_loss(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Compute L2 (MSE) loss between ``pred`` and ``target``.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted tensor.
    target : torch.Tensor
        Target tensor with the same shape as ``pred``.
    reduction : {"none", "mean", "sum"}
        Reduction method. Default is "mean".

    Returns
    -------
    torch.Tensor
        Loss tensor following the specified ``reduction``.
    """
    diff = pred - target
    loss = diff.pow(2)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


