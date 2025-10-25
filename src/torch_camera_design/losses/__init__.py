"""Loss functions for camera design and evaluation.

This package provides:
- Luther loss: deviation from the Luther condition (linear mapping to CMFs)
- Vora loss/value: subspace similarity between sensor set and CMFs
- L2 loss: basic mean-squared-error utility
"""

from __future__ import annotations

from .l2 import l2_loss
from .luther import luther_loss
from .vora import vora_loss, vora_value

__all__ = [
    "l2_loss",
    "luther_loss",
    "vora_loss",
    "vora_value",
]


