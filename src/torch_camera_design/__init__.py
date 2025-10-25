"""Top-level package for torch_camera_design.

Expose a light, lazy API surface so importing the package does not eagerly
pull heavy dependencies. Subpackages and frequently used functions are
available via lazy attribute access.
"""

from __future__ import annotations

from ._version import __version__

from typing import TYPE_CHECKING
import importlib

__all__ = [
    "__version__",
    "losses",
    "evaluation",
    # Frequently used symbols (resolved lazily)
    "l2_loss",
    "luther_loss",
    "luther_mapping_loss",
    "luther_regression_loss",
    "vora_loss",
    "vora_value",
    "vora_value_general",
]

if TYPE_CHECKING:  # For IDEs/type-checkers only (no runtime import cost)
    from . import losses as losses  # noqa: F401
    from . import evaluation as evaluation  # noqa: F401
    from .losses import (  # noqa: F401
        l2_loss,
        luther_loss,
        luther_mapping_loss,
        luther_regression_loss,
        vora_loss,
        vora_value,
        vora_value_general,
    )

_LAZY_ATTRS = {
    # subpackages
    "losses": "torch_camera_design.losses",
    "evaluation": "torch_camera_design.evaluation",
    # selected symbols
    "l2_loss": "torch_camera_design.losses",
    "luther_loss": "torch_camera_design.losses",
    "luther_mapping_loss": "torch_camera_design.losses",
    "luther_regression_loss": "torch_camera_design.losses",
    "vora_loss": "torch_camera_design.losses",
    "vora_value": "torch_camera_design.losses",
    "vora_value_general": "torch_camera_design.losses",
}


def __getattr__(name: str):
    module_path = _LAZY_ATTRS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'torch_camera_design' has no attribute '{name}'")
    module = importlib.import_module(module_path)
    # If requesting a subpackage, return module directly
    if name in ("losses", "evaluation"):
        return module
    return getattr(module, name)


def __dir__():  # pragma: no cover - convenience only
    return sorted(list(globals().keys()) + list(_LAZY_ATTRS.keys()))
