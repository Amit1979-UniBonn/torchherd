"""PyTorch translations of ``MethaneEmission.java`` and ``MethaneEmissionLite.java``."""

from __future__ import annotations

import torch
from torch import nn

from ._common import DEFAULT_DTYPE, make_parameter, as_tensor


class MethaneEmission(nn.Module):
    """Enteric fermentation emission per IPCC 2006 Vol 4 Ch 10 (eq. 10.21).

    Mirrors the Java implementation: weight-shares of the herd weight a
    per-class emission factor multiplied by the per-weight share of energy
    consumption — the bookkeeping is mathematically equivalent to summing
    ``Y_m * specific_emission * (energy_share_per_class)``.
    """

    def __init__(
        self,
        *,
        cSpecificMethanEmission: float = 1.8e-5,
        cJuvenileFemaleMethanConversionFraction: float = 0.045,
        cSubAdultFemaleMethanConversionFraction: float = 0.055,
        cAdultFemaleMethanConversionFraction: float = 0.065,
        cJuvenileMaleMethanConversionFraction: float = 0.045,
        cSubAdultMaleMethanConversionFraction: float = 0.055,
        cAdultMaleMethanConversionFraction: float = 0.065,
        cJuvenileFemaleWeight: float = 35000.0,
        cSubAdultFemaleWeight: float = 150000.0,
        cAdultFemaleWeight: float = 250000.0,
        cJuvenileMaleWeight: float = 45000.0,
        cSubAdultMaleWeight: float = 185000.0,
        cAdultMaleWeight: float = 300000.0,
        dtype: torch.dtype = DEFAULT_DTYPE,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        register = (lambda n, v: setattr(self, n, make_parameter(v, dtype=dtype))) \
            if trainable else \
            (lambda n, v: self.register_buffer(n, as_tensor(v, dtype=dtype)))

        register("cSpecificMethanEmission", cSpecificMethanEmission)
        register("cJuvenileFemaleMethanConversionFraction", cJuvenileFemaleMethanConversionFraction)
        register("cSubAdultFemaleMethanConversionFraction", cSubAdultFemaleMethanConversionFraction)
        register("cAdultFemaleMethanConversionFraction", cAdultFemaleMethanConversionFraction)
        register("cJuvenileMaleMethanConversionFraction", cJuvenileMaleMethanConversionFraction)
        register("cSubAdultMaleMethanConversionFraction", cSubAdultMaleMethanConversionFraction)
        register("cAdultMaleMethanConversionFraction", cAdultMaleMethanConversionFraction)
        register("cJuvenileFemaleWeight", cJuvenileFemaleWeight)
        register("cSubAdultFemaleWeight", cSubAdultFemaleWeight)
        register("cAdultFemaleWeight", cAdultFemaleWeight)
        register("cJuvenileMaleWeight", cJuvenileMaleWeight)
        register("cSubAdultMaleWeight", cSubAdultMaleWeight)
        register("cAdultMaleWeight", cAdultMaleWeight)

    def forward(
        self,
        iJuvenileFemale: torch.Tensor,
        iSubAdultFemale: torch.Tensor,
        iAdultFemale: torch.Tensor,
        iJuvenileMale: torch.Tensor,
        iSubAdultMale: torch.Tensor,
        iAdultMale: torch.Tensor,
        iEnergyConsumption: torch.Tensor,
    ) -> dict:
        tTotalWeight = (
            iAdultFemale * self.cAdultFemaleWeight
            + iAdultMale * self.cAdultMaleWeight
            + iSubAdultFemale * self.cSubAdultFemaleWeight
            + iSubAdultMale * self.cSubAdultMaleWeight
            + iJuvenileFemale * self.cJuvenileFemaleWeight
            + iJuvenileMale * self.cJuvenileMaleWeight
        )
        tEnergyPerWeight = iEnergyConsumption / tTotalWeight

        spec = self.cSpecificMethanEmission
        ch4 = (
            self.cAdultFemaleMethanConversionFraction * spec * self.cAdultFemaleWeight * iAdultFemale * tEnergyPerWeight
            + self.cJuvenileFemaleMethanConversionFraction * spec * self.cJuvenileFemaleWeight * iJuvenileFemale * tEnergyPerWeight
            + self.cSubAdultFemaleMethanConversionFraction * spec * self.cSubAdultFemaleWeight * iSubAdultFemale * tEnergyPerWeight
            + self.cAdultMaleMethanConversionFraction * spec * self.cAdultMaleWeight * iAdultMale * tEnergyPerWeight
            + self.cJuvenileMaleMethanConversionFraction * spec * self.cJuvenileMaleWeight * iJuvenileMale * tEnergyPerWeight
            + self.cSubAdultMaleMethanConversionFraction * spec * self.cSubAdultMaleWeight * iSubAdultMale * tEnergyPerWeight
        )
        return {"CH4emission": ch4}


class MethaneEmissionLite(nn.Module):
    """Ndungu et al. (2019) lite model: ``CH4 = 20.7 * (E / 18100) / 1000``.

    Returns kg CH4 / day from energy consumption in joules (the Java code
    converts 18.1 MJ/kg dry matter -> 18100 J/g, then 20.7 g/day -> kg/day).
    """

    def __init__(self, *, dtype: torch.dtype = DEFAULT_DTYPE) -> None:
        super().__init__()
        self._dtype = dtype
        self.register_buffer("_marker", torch.zeros((), dtype=dtype))

    def forward(self, iEnergyConsumption: torch.Tensor) -> dict:
        E = as_tensor(iEnergyConsumption, dtype=self._dtype)
        tDryMatterConsumed = E / 18100  # g
        ch4 = 20.7 * tDryMatterConsumed / 1000  # kg
        return {"CH4emission": ch4}
