"""PyTorch translation of ``FodderConsumption.java``."""

from __future__ import annotations

import torch
from torch import nn

from ._common import DEFAULT_DTYPE, make_parameter, as_tensor


class FodderConsumption(nn.Module):
    """Daily fodder consumption + storage update.

    Per the Java implementation, ``rFodder`` is computed as the weighted sum
    of per-class **weights** (not animal counts) times a fraction
    ``cXxxFodderFraction``.  We preserve that exact interpretation.
    """

    def __init__(
        self,
        *,
        cJuvenileFemaleFodderFraction: float = 0.025,
        cSubAdultFemaleFodderFraction: float = 0.025,
        cAdultFemaleFodderFraction: float = 0.025,
        cJuvenileMaleFodderFraction: float = 0.025,
        cSubAdultMaleFodderFraction: float = 0.025,
        cAdultMaleFodderFraction: float = 0.025,
        cDegradationFactor: float = 0.001,
        dtype: torch.dtype = DEFAULT_DTYPE,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        register = (lambda name, v: setattr(self, name, make_parameter(v, dtype=dtype))) \
            if trainable else \
            (lambda name, v: self.register_buffer(name, as_tensor(v, dtype=dtype)))

        register("cJuvenileFemaleFodderFraction", cJuvenileFemaleFodderFraction)
        register("cSubAdultFemaleFodderFraction", cSubAdultFemaleFodderFraction)
        register("cAdultFemaleFodderFraction", cAdultFemaleFodderFraction)
        register("cJuvenileMaleFodderFraction", cJuvenileMaleFodderFraction)
        register("cSubAdultMaleFodderFraction", cSubAdultMaleFodderFraction)
        register("cAdultMaleFodderFraction", cAdultMaleFodderFraction)
        register("cDegradationFactor", cDegradationFactor)

        self._dtype = dtype
        self.reset_state()

    def reset_state(self) -> None:
        device = self.cDegradationFactor.device
        self.sFodderStorage = torch.zeros((), dtype=self._dtype, device=device)
        self.FodderConsumed = torch.zeros((), dtype=self._dtype, device=device)
        self.FodderDegraded = torch.zeros((), dtype=self._dtype, device=device)

    def forward(
        self,
        iJuvenileFemaleWeight: torch.Tensor,
        iSubAdultFemaleWeight: torch.Tensor,
        iAdultFemaleWeight: torch.Tensor,
        iJuvenileMaleWeight: torch.Tensor,
        iSubAdultMaleWeight: torch.Tensor,
        iAdultMaleWeight: torch.Tensor,
        iFodderProduction: torch.Tensor,
        *,
        iDoInitialize: bool = False,
    ) -> dict:
        if iDoInitialize:
            self.reset_state()

        rFodder = (
            iJuvenileFemaleWeight * self.cJuvenileFemaleFodderFraction
            + iJuvenileMaleWeight * self.cJuvenileMaleFodderFraction
            + iSubAdultFemaleWeight * self.cSubAdultFemaleFodderFraction
            + iSubAdultMaleWeight * self.cSubAdultMaleFodderFraction
            + iAdultFemaleWeight * self.cAdultFemaleFodderFraction
            + iAdultMaleWeight * self.cAdultMaleFodderFraction
        )

        self.sFodderStorage = iFodderProduction - rFodder + self.sFodderStorage
        one_minus_deg = 1 - self.cDegradationFactor
        self.FodderDegraded = self.FodderDegraded + self.sFodderStorage * one_minus_deg
        self.sFodderStorage = self.sFodderStorage * one_minus_deg
        self.FodderConsumed = self.FodderConsumed + rFodder

        return {
            "rFodder": rFodder,
            "sFodderStorage": self.sFodderStorage,
            "FodderConsumed": self.FodderConsumed,
            "FodderDegraded": self.FodderDegraded,
        }
