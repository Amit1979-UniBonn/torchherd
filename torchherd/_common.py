"""Shared helpers used by the livestock PyTorch modules."""

from __future__ import annotations

from typing import Union

import torch

# Default precision matches Java ``Double`` semantics.
DEFAULT_DTYPE = torch.float64

Number = Union[float, int, torch.Tensor]


def as_tensor(value: Number, *, dtype: torch.dtype = DEFAULT_DTYPE,
              device: torch.device | None = None) -> torch.Tensor:
    """Coerce ``value`` to a tensor with the default precision.

    Scalars become 0-d tensors; existing tensors are cast/moved when needed.
    """
    if isinstance(value, torch.Tensor):
        if value.dtype != dtype:
            value = value.to(dtype)
        if device is not None and value.device != device:
            value = value.to(device)
        return value
    return torch.as_tensor(value, dtype=dtype, device=device)


def make_parameter(value: Number, *, dtype: torch.dtype = DEFAULT_DTYPE) -> torch.nn.Parameter:
    """Wrap ``value`` in an ``nn.Parameter`` of the requested dtype."""
    return torch.nn.Parameter(as_tensor(value, dtype=dtype).clone())


def module_device(module: torch.nn.Module) -> torch.device:
    """Return the device of the first parameter or buffer of ``module``.

    Falls back to CPU when the module is parameter-less.
    """
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    return torch.device("cpu")


def soft_eq_string(left: str, right: str) -> bool:
    """Case-insensitive string comparison mirroring Java's ``equalsIgnoreCase``."""
    return left.lower() == right.lower()
