"""PyTorch translation of ``MilkProduction.java``."""

from __future__ import annotations

import torch
from torch import nn

from ._common import DEFAULT_DTYPE, make_parameter, as_tensor


class MilkProduction(nn.Module):
    """Milk produced / consumed / exported per day."""

    def __init__(
        self,
        *,
        cMilkOfftake: float = 1500.0,
        cParturition: float = 0.5,
        cMilkingDays: int = 90,
        cJuvenileConsumptionOfftake: float = 800.0,
        dtype: torch.dtype = DEFAULT_DTYPE,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        register = (lambda n, v: setattr(self, n, make_parameter(v, dtype=dtype))) \
            if trainable else \
            (lambda n, v: self.register_buffer(n, as_tensor(v, dtype=dtype)))

        register("cMilkOfftake", cMilkOfftake)
        register("cParturition", cParturition)
        register("cJuvenileConsumptionOfftake", cJuvenileConsumptionOfftake)
        # MilkingDays is integer in Java; keep as int buffer (not trainable).
        self.register_buffer("cMilkingDays", as_tensor(cMilkingDays, dtype=dtype))

    def forward(
        self,
        iJuvenileMale: torch.Tensor,
        iJuvenileFemale: torch.Tensor,
        iAdultFemale: torch.Tensor,
    ) -> dict:
        MilkProduced = (
            iAdultFemale * self.cParturition * self.cMilkOfftake * self.cMilkingDays / 365
        )
        MilkConsumed = (iJuvenileFemale + iJuvenileMale) * self.cJuvenileConsumptionOfftake
        MilkExported = MilkProduced - MilkConsumed
        return {
            "MilkProduced": MilkProduced,
            "MilkConsumed": MilkConsumed,
            "MilkExported": MilkExported,
        }
