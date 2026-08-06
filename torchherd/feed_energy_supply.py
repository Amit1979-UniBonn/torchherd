"""PyTorch translation of ``FeedEnergySupply.java``."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ._common import DEFAULT_DTYPE, make_parameter, as_tensor


class FeedEnergySupply(nn.Module):
    """Forage / concentrate storage with daily degradation.

    Stateful module: ``sForageStorage``, ``sConcentrateStorage``, and the
    cumulative output totals (``ForageConsumed``, ``ConcentrateConsumed``,
    ``ForageDegraded``) are kept as non-trainable buffers and mutated
    out-of-place each call so autograd can still flow through a fresh forward
    pass.

    The :meth:`forward` method returns the updated outputs as a dict mirroring
    the Java output variables.  The Java in-place state mutation is recreated
    by reassigning the buffers via :func:`register_buffer` style attribute
    writes — but using ``copy_`` would break differentiability through the
    state.  Instead, the state lives in :class:`torch.Tensor` attributes and
    is replaced wholesale, retaining the graph.
    """

    def __init__(
        self,
        *,
        cDegradationFactor: float = 0.001,
        cForageInitial: float = 0.0,
        cForageMax: float = 0.0,
        cConcentrateInitial: float = 0.0,
        dtype: torch.dtype = DEFAULT_DTYPE,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if trainable:
            self.cDegradationFactor = make_parameter(cDegradationFactor, dtype=dtype)
        else:
            self.register_buffer("cDegradationFactor", as_tensor(cDegradationFactor, dtype=dtype))
        # Initial/maximum are simulation constants, not optimisation targets.
        self.register_buffer("cForageInitial", as_tensor(cForageInitial, dtype=dtype))
        self.register_buffer("cForageMax", as_tensor(cForageMax, dtype=dtype))
        self.register_buffer("cConcentrateInitial", as_tensor(cConcentrateInitial, dtype=dtype))

        self._dtype = dtype
        self.reset_state()

    def reset_state(self) -> None:
        """Reinitialise per-Java ``init()``."""
        # State variables live as plain attributes so they can be replaced
        # without losing autograd connectivity.
        self.sForageStorage = self.cForageInitial.clone()
        self.sConcentrateStorage = self.cConcentrateInitial.clone()
        self.ForageConsumed = torch.zeros((), dtype=self._dtype, device=self.cForageInitial.device)
        self.ConcentrateConsumed = torch.zeros((), dtype=self._dtype, device=self.cForageInitial.device)
        self.ForageDegraded = torch.zeros((), dtype=self._dtype, device=self.cForageInitial.device)

    def forward(
        self,
        iForageEnergyConsumption: torch.Tensor,
        iConcentrateEnergyConsumption: torch.Tensor,
        *,
        iForageProductionEnergyContent: Optional[torch.Tensor] = None,
        iConcentrateProductionEnergyContent: Optional[torch.Tensor] = None,
        iDoInitialize: bool = False,
        iDoEmptyForageStore: bool = False,
    ) -> dict:
        if iDoEmptyForageStore:
            self.ForageDegraded = self.ForageDegraded + self.sForageStorage
            self.sForageStorage = torch.zeros_like(self.sForageStorage)
        if iDoInitialize:
            self.reset_state()

        if iForageProductionEnergyContent is not None:
            self.sForageStorage = (
                self.sForageStorage + iForageProductionEnergyContent - iForageEnergyConsumption
            )

        one_minus_deg = 1 - self.cDegradationFactor
        self.ForageDegraded = self.ForageDegraded + self.sForageStorage * one_minus_deg
        self.sForageStorage = self.sForageStorage * one_minus_deg

        # cForageMax > 0 acts as a cap.
        if float(self.cForageMax) > 0:
            # Use torch.where so autograd flows through the cap correctly.
            over = self.sForageStorage - self.cForageMax
            mask = (self.cForageMax < self.sForageStorage)
            self.ForageDegraded = self.ForageDegraded + torch.where(
                mask, over, torch.zeros_like(over)
            )
            self.sForageStorage = torch.where(mask, self.cForageMax, self.sForageStorage)

        self.ForageConsumed = self.ForageConsumed + iForageEnergyConsumption

        if iConcentrateProductionEnergyContent is not None:
            self.sConcentrateStorage = (
                self.sConcentrateStorage
                + iConcentrateProductionEnergyContent
                - iConcentrateEnergyConsumption
            )
        self.ConcentrateConsumed = self.ConcentrateConsumed + iConcentrateEnergyConsumption

        return {
            "sForageStorage": self.sForageStorage,
            "sConcentrateStorage": self.sConcentrateStorage,
            "ForageConsumed": self.ForageConsumed,
            "ConcentrateConsumed": self.ConcentrateConsumed,
            "ForageDegraded": self.ForageDegraded,
        }
