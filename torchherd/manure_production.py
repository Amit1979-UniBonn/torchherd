"""PyTorch translation of ``ManureProduction.java``."""

from __future__ import annotations

import torch
from torch import nn

from ._common import DEFAULT_DTYPE, make_parameter, as_tensor


class ManureProduction(nn.Module):
    """Daily manure rate + storage / fertiliser / dropping flows."""

    def __init__(
        self,
        *,
        cJuvenileFemaleManureAmount: float = 500.0,
        cSubAdultFemaleManureAmount: float = 1000.0,
        cAdultFemaleManureAmount: float = 1100.0,
        cJuvenileMaleManureAmount: float = 500.0,
        cSubAdultMaleManureAmount: float = 1000.0,
        cAdultMaleManureAmount: float = 1100.0,
        cDegradationFactor: float = 0.001,
        dtype: torch.dtype = DEFAULT_DTYPE,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        register = (lambda n, v: setattr(self, n, make_parameter(v, dtype=dtype))) \
            if trainable else \
            (lambda n, v: self.register_buffer(n, as_tensor(v, dtype=dtype)))

        register("cJuvenileFemaleManureAmount", cJuvenileFemaleManureAmount)
        register("cSubAdultFemaleManureAmount", cSubAdultFemaleManureAmount)
        register("cAdultFemaleManureAmount", cAdultFemaleManureAmount)
        register("cJuvenileMaleManureAmount", cJuvenileMaleManureAmount)
        register("cSubAdultMaleManureAmount", cSubAdultMaleManureAmount)
        register("cAdultMaleManureAmount", cAdultMaleManureAmount)
        register("cDegradationFactor", cDegradationFactor)

        self._dtype = dtype
        self.reset_state()

    def reset_state(self) -> None:
        device = self.cDegradationFactor.device
        self.sManureStorage = torch.zeros((), dtype=self._dtype, device=device)
        self.ManureProduced = torch.zeros((), dtype=self._dtype, device=device)

    def forward(
        self,
        iJuvenileFemale: torch.Tensor,
        iSubAdultFemale: torch.Tensor,
        iAdultFemale: torch.Tensor,
        iJuvenileMale: torch.Tensor,
        iSubAdultMale: torch.Tensor,
        iAdultMale: torch.Tensor,
        iFertilizeFraction: torch.Tensor,
        iDroppingFraction: torch.Tensor,
        iMaintenanceNutritionIndex: torch.Tensor,
        *,
        iDoInitialize: bool = False,
    ) -> dict:
        if iDoInitialize:
            self.reset_state()

        rManure = iMaintenanceNutritionIndex * (
            iJuvenileFemale * self.cJuvenileFemaleManureAmount
            + iJuvenileMale * self.cJuvenileMaleManureAmount
            + iSubAdultFemale * self.cSubAdultFemaleManureAmount
            + iSubAdultMale * self.cSubAdultMaleManureAmount
            + iAdultFemale * self.cAdultFemaleManureAmount
            + iAdultMale * self.cAdultMaleManureAmount
        )

        DroppingAmount = rManure * iDroppingFraction
        self.ManureProduced = self.ManureProduced + rManure
        DegradedAmount = self.sManureStorage * self.cDegradationFactor
        self.sManureStorage = (
            self.sManureStorage * (1 - self.cDegradationFactor) + rManure - DroppingAmount
        )
        FertilizerAmount = self.sManureStorage * iFertilizeFraction
        self.sManureStorage = self.sManureStorage - FertilizerAmount

        return {
            "rManure": rManure,
            "sManureStorage": self.sManureStorage,
            "ManureProduced": self.ManureProduced,
            "FertilizerAmount": FertilizerAmount,
            "DroppingAmount": DroppingAmount,
            "DegradedAmount": DegradedAmount,
        }
