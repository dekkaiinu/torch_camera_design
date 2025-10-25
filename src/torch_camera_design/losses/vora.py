from __future__ import annotations

import torch

__all__ = ["vora_value", "vora_loss", "vora_value_general"]


def _orthonormal_basis(x: torch.Tensor) -> torch.Tensor:
    """Return an orthonormal basis spanning ``col(x)`` using QR decomposition.

    ``x``: shape (n, k)
    Returns: ``Q`` with shape (n, r), r = rank(x)
    """
    if x.numel() == 0:
        raise ValueError("input is empty")
    q, r = torch.linalg.qr(x, mode="reduced")
    # Handle potential rank-deficiency by trimming near-zero diagonals
    diag = torch.abs(torch.diagonal(r))
    tol = torch.finfo(x.dtype).eps * max(x.shape)
    rnk = int((diag > tol).sum().item())
    return q[:, :rnk]


def _subspace_projector(q: torch.Tensor) -> torch.Tensor:
    """Projector matrix onto span(q), where q has orthonormal columns."""
    return q @ q.mT


def vora_value(sensors: torch.Tensor, cmfs: torch.Tensor) -> torch.Tensor:
    """Compute Vora-Value: similarity of two subspaces in [0, 1].

    Defined as the average of squared cosines of principal angles between the
    subspaces spanned by ``sensors`` and ``cmfs``. Implemented via projectors:

    VV = (1/m) * trace(P_sensors @ P_cmfs), where m = min(rank(sensors), rank(cmfs)).
    """
    if sensors.ndim != 2 or cmfs.ndim != 2:
        raise ValueError("sensors and cmfs must be 2D tensors")
    if sensors.size(0) != cmfs.size(0):
        raise ValueError("sensors and cmfs must share the first dimension (wavelength samples)")

    q_s = _orthonormal_basis(sensors)
    q_c = _orthonormal_basis(cmfs)
    p_s = _subspace_projector(q_s)
    p_c = _subspace_projector(q_c)
    m = min(q_s.size(1), q_c.size(1))
    # trace(Ps Pc) equals sum of squared cosines of principal angles
    val = torch.trace(p_s @ p_c) / float(m)
    # Clamp to [0, 1] for numerical safety
    return torch.clamp(val, 0.0, 1.0)


def vora_value_general(Q: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """一般射影版 Vora-Value．

    提示クラスの ``compute_vora_value`` 相当の定義（P= X (X^T X)^{-1} X^T）を
    双方に適用して Vora-Value を計算する．
    """
    if Q.ndim != 2 or X.ndim != 2:
        raise ValueError("Q, X は 2D Tensor である必要があります")
    if Q.size(0) != X.size(0):
        raise ValueError("Q と X は同じサンプル数（第1次元）である必要があります")
    XtX_inv = torch.linalg.inv(X.T @ X + 1e-6 * torch.eye(X.size(1), device=X.device, dtype=X.dtype))
    PtX = X @ XtX_inv @ X.T
    QtQ_inv = torch.linalg.inv(Q.T @ Q + 1e-6 * torch.eye(Q.size(1), device=Q.device, dtype=Q.dtype))
    PtQ = Q @ QtQ_inv @ Q.T
    m = min(X.size(1), Q.size(1))
    val = torch.trace(PtQ @ PtX) / float(m)
    return torch.clamp(val, 0.0, 1.0)


def vora_loss(sensors: torch.Tensor, cmfs: torch.Tensor) -> torch.Tensor:
    """Loss counterpart of Vora-Value: 1 - VV, in [0, 1]."""
    return 1.0 - vora_value(sensors, cmfs)


