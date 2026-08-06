"""PyTorch translation of ``FeedEnergyContent.java``."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ._common import DEFAULT_DTYPE, make_parameter, as_tensor


class FeedEnergyContent(nn.Module):
    """Sum of ``feed_production[k] * feed_energy_content[k]`` across 5 compartments.

    Mirrors the Java ``process()`` exactly, including the null-coalescing on
    every compartment except the first (compartment 1 is mandatory in the
    Java code path).  Inputs that are ``None`` are skipped.
    """

    def __init__(
        self,
        c_energy_contents: Optional[torch.Tensor] = None,
        *,
        dtype: torch.dtype = DEFAULT_DTYPE,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if c_energy_contents is None:
            c_energy_contents = torch.full((5,), 0.25, dtype=dtype)
        c_energy_contents = as_tensor(c_energy_contents, dtype=dtype)
        if c_energy_contents.shape != (5,):
            raise ValueError("expected exactly 5 energy contents")
        if trainable:
            self.cFeedEnergyContent = nn.Parameter(c_energy_contents.clone())
        else:
            self.register_buffer("cFeedEnergyContent", c_energy_contents.clone())

    def forward(
        self,
        iFeedProduction1: torch.Tensor,
        iFeedProduction2: Optional[torch.Tensor] = None,
        iFeedProduction3: Optional[torch.Tensor] = None,
        iFeedProduction4: Optional[torch.Tensor] = None,
        iFeedProduction5: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        c = self.cFeedEnergyContent
        total = iFeedProduction1 * c[0]
        for i, p in enumerate((iFeedProduction2, iFeedProduction3, iFeedProduction4, iFeedProduction5), start=1):
            if p is not None:
                total = total + p * c[i]
        return total
