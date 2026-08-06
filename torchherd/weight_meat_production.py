"""PyTorch translation of ``WeightAndMeatProduction.java``.

Note: the Java ``process()`` writes per-class weight outputs in a shuffled
order (e.g. ``SubAdultFemaleWeight = iJuvenileMale * cWeightPerJuvenileMale``).
Per the CLAUDE.md directive to *match the exact logic of the Java
implementation*, this translation preserves that exact mapping. The
aggregate ``SumWeight`` is unaffected since every age/sex class contributes
exactly once to the total.
"""

from __future__ import annotations

import torch
from torch import nn

from ._common import DEFAULT_DTYPE, make_parameter, as_tensor


class WeightAndMeatProduction(nn.Module):
    def __init__(
        self,
        *,
        cWeightPerJuvenileFemale: float = 35000.0,
        cWeightPerSubAdultFemale: float = 150000.0,
        cWeightPerAdultFemale: float = 250000.0,
        cWeightPerJuvenileMale: float = 45000.0,
        cWeightPerSubAdultMale: float = 185000.0,
        cWeightPerAdultMale: float = 300000.0,
        cCarcassFraction: float = 0.47,
        dtype: torch.dtype = DEFAULT_DTYPE,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        register = (lambda n, v: setattr(self, n, make_parameter(v, dtype=dtype))) \
            if trainable else \
            (lambda n, v: self.register_buffer(n, as_tensor(v, dtype=dtype)))

        register("cWeightPerJuvenileFemale", cWeightPerJuvenileFemale)
        register("cWeightPerSubAdultFemale", cWeightPerSubAdultFemale)
        register("cWeightPerAdultFemale", cWeightPerAdultFemale)
        register("cWeightPerJuvenileMale", cWeightPerJuvenileMale)
        register("cWeightPerSubAdultMale", cWeightPerSubAdultMale)
        register("cWeightPerAdultMale", cWeightPerAdultMale)
        register("cCarcassFraction", cCarcassFraction)

        self._dtype = dtype
        self.reset_state()

    def reset_state(self) -> None:
        device = self.cCarcassFraction.device
        self.SumWeight = torch.zeros((), dtype=self._dtype, device=device)

    def forward(
        self,
        iJuvenileFemale: torch.Tensor,
        iSubAdultFemale: torch.Tensor,
        iAdultFemale: torch.Tensor,
        iJuvenileMale: torch.Tensor,
        iSubAdultMale: torch.Tensor,
        iAdultMale: torch.Tensor,
    ) -> dict:
        # Java mapping preserved verbatim (mismatched output names but
        # mathematically the same sum):
        JuvenileFemaleWeight = iJuvenileFemale * self.cWeightPerJuvenileFemale
        SubAdultFemaleWeight = iJuvenileMale * self.cWeightPerJuvenileMale
        AdultFemaleWeight = iSubAdultFemale * self.cWeightPerSubAdultFemale
        JuvenileMaleWeight = iSubAdultMale * self.cWeightPerSubAdultMale
        SubAdultMaleWeight = iAdultFemale * self.cWeightPerAdultFemale
        AdultMaleWeight = iAdultMale * self.cWeightPerAdultMale

        tWeight = (
            JuvenileFemaleWeight
            + SubAdultFemaleWeight
            + AdultFemaleWeight
            + JuvenileMaleWeight
            + SubAdultMaleWeight
            + AdultMaleWeight
        )
        WeightChange = self.SumWeight - tWeight
        self.SumWeight = tWeight
        SumMeat = self.SumWeight * self.cCarcassFraction

        return {
            "JuvenileFemaleWeight": JuvenileFemaleWeight,
            "SubAdultFemaleWeight": SubAdultFemaleWeight,
            "AdultFemaleWeight": AdultFemaleWeight,
            "JuvenileMaleWeight": JuvenileMaleWeight,
            "SubAdultMaleWeight": SubAdultMaleWeight,
            "AdultMaleWeight": AdultMaleWeight,
            "SumWeight": self.SumWeight,
            "WeightChange": WeightChange,
            "SumMeat": SumMeat,
        }
