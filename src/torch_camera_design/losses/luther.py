from __future__ import annotations

import torch

__all__ = [
    "luther_loss",
    "estimate_luther_mapping",
    "luther_mapping_loss",
    "luther_regression_loss",
]


def _projection_matrix(basis: torch.Tensor) -> torch.Tensor:
    """Return the (orthogonal) projection matrix onto ``span(basis)``.

    Parameters
    ----------
    basis : torch.Tensor, shape (n_wavelengths, k)
        Column space defines the subspace. ``k`` is typically 3.
    """
    pseudo_inverse = torch.linalg.pinv(basis)
    return basis @ pseudo_inverse


def estimate_luther_mapping(cmfs: torch.Tensor, sensors: torch.Tensor) -> torch.Tensor:
    """Least-squares mapping ``A`` s.t. ``cmfs @ A ≈ sensors``.

    Parameters
    ----------
    cmfs : torch.Tensor, shape (n, 3)
        Color matching functions sampled at ``n`` wavelengths.
    sensors : torch.Tensor, shape (n, m)
        Sensor sensitivities (``m`` channels, often 3).

    Returns
    -------
    torch.Tensor, shape (3, m)
        The minimizing linear map ``A``.
    """
    return torch.linalg.pinv(cmfs) @ sensors


def luther_loss(
    sensors: torch.Tensor,
    cmfs: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Deviation from the Luther condition as subspace distance.

    Computes the Frobenius norm of the residual between the sensor set and its
    projection onto the CMF subspace: ``||(I - P_cmfs) @ sensors||_F``. If
    ``normalize=True``, divides by ``||sensors||_F`` to obtain a scale-free loss
    in ``[0, +inf)`` with ``0`` meaning perfect Luther.

    Parameters
    ----------
    sensors : torch.Tensor, shape (n, m)
        Sensor sensitivities sampled at the same ``n`` wavelengths.
    cmfs : torch.Tensor, shape (n, 3)
        CIE color matching functions (or other reference basis).
    normalize : bool, default True
        If True, divide by ``||sensors||_F``.

    Returns
    -------
    torch.Tensor, scalar
        Luther loss value.
    """
    if sensors.ndim != 2 or cmfs.ndim != 2:
        raise ValueError("sensors and cmfs must be 2D tensors")
    if sensors.size(0) != cmfs.size(0):
        raise ValueError("sensors and cmfs must share the first dimension (wavelength samples)")

    projector_cmfs = _projection_matrix(cmfs)
    identity = torch.eye(projector_cmfs.size(0), dtype=sensors.dtype, device=sensors.device)
    residual = (identity - projector_cmfs) @ sensors
    num = torch.linalg.norm(residual, ord="fro")
    if not normalize:
        return num
    denom = torch.linalg.norm(sensors, ord="fro")
    # Avoid division by zero for degenerate input
    return num / (denom + torch.finfo(sensors.dtype).eps)


def luther_mapping_loss(Q: torch.Tensor, M: torch.Tensor, V: torch.Tensor, *, normalize: bool = False) -> torch.Tensor:
    """Luther loss (mapping 版): ||Q M − V||_F．

    提示コードの ``compute_luther_loss`` と同等の定義．``normalize=True`` のとき
    は ``||V||_F`` で割ってスケール不変にする．
    """
    if Q.ndim != 2 or M.ndim != 2 or V.ndim != 2:
        raise ValueError("Q, M, V は 2D Tensor である必要があります")
    if Q.size(1) != M.size(0):
        raise ValueError("Q @ M が定義できません（次元不一致）")
    if Q.size(0) != V.size(0) or M.size(1) != V.size(1):
        raise ValueError("QM と V の形状が一致しません")
    diff = Q @ M - V
    num = torch.linalg.norm(diff, ord="fro")
    if not normalize:
        return num
    denom = torch.linalg.norm(V, ord="fro")
    return num / (denom + torch.finfo(V.dtype).eps)


def luther_regression_loss(Q: torch.Tensor, X: torch.Tensor, *, normalize: bool = False) -> torch.Tensor:
    """回帰版 Luther: M=pinv(Q)X による最小二乗誤差 ||Q M − X||_F．

    これは ``(P_Q − I)X`` の Frobenius ノルムと等価（``P_Q=Q pinv(Q)``）．
    """
    if Q.ndim != 2 or X.ndim != 2:
        raise ValueError("Q, X は 2D Tensor である必要があります")
    if Q.size(0) != X.size(0):
        raise ValueError("Q と X は同じサンプル数（第1次元）である必要があります")
    M_hat = torch.linalg.pinv(Q) @ X
    return luther_mapping_loss(Q, M_hat, X, normalize=normalize)


