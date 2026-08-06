"""PyTorch translation of ``LivestockEnergyDemand.java``.

The Java class branches on two strings (``cAnimal``, ``cActivity``).  In
keeping with the CLAUDE.md mandate, those are turned into index-based
lookups:

* ``animal``    in ``{"Cattle", "Sheep"}``    -> ``{0, 1}``
* ``activity``  in ``{"Stall", "Pasture", "Grazing"}`` -> ``{0, 1, 2}``

A 2x3 lookup table holds the IPCC ``Ca`` activity coefficients.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from ._common import DEFAULT_DTYPE, make_parameter, as_tensor, soft_eq_string


# IPCC Cfi maintenance coefficients [J / (kg * day)]
_CFI_COW_NONLACTATING = 322000.0
_CFI_COW_LACTATING = 386000.0
_CFI_BULL = 370000.0
_CFI_LAMB = 236000.0
_CFI_SHEEP = 217000.0

# Activity coefficients indexed by (animal, activity).
_CA_TABLE = torch.tensor(
    [
        # Stall, Pasture, Grazing
        [0.0, 0.17, 0.36],   # Cattle
        [0.0067, 0.107, 0.0],  # Sheep
    ],
    dtype=DEFAULT_DTYPE,
)

_ANIMAL_INDEX = {"cattle": 0, "sheep": 1}
_ACTIVITY_INDEX = {"stall": 0, "pasture": 1, "grazing": 2}


def _resolve_index(value, mapping: Mapping[str, int]) -> int:
    if isinstance(value, str):
        try:
            return mapping[value.lower()]
        except KeyError as exc:
            raise ValueError(f"unknown category {value!r}; expected one of {list(mapping)}") from exc
    return int(value)


class LivestockEnergyDemand(nn.Module):
    """Maintenance + growth + lactation energy demand per IPCC Vol 4 Ch 10."""

    def __init__(
        self,
        *,
        cJuvenileFemaleWeight: float = 35000.0,
        cSubAdultFemaleWeight: float = 150000.0,
        cAdultFemaleWeight: float = 250000.0,
        cJuvenileMaleWeight: float = 45000.0,
        cSubAdultMaleWeight: float = 185000.0,
        cAdultMaleWeight: float = 300000.0,
        cMeanWinterTemperature: float = 20.0,
        cMilkEnergyContent: float = 4600.0,
        cAnimal: str | int = "Cattle",
        cActivity: str | int = "Stall",
        dtype: torch.dtype = DEFAULT_DTYPE,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        register = (lambda n, v: setattr(self, n, make_parameter(v, dtype=dtype))) \
            if trainable else \
            (lambda n, v: self.register_buffer(n, as_tensor(v, dtype=dtype)))

        register("cJuvenileFemaleWeight", cJuvenileFemaleWeight)
        register("cSubAdultFemaleWeight", cSubAdultFemaleWeight)
        register("cAdultFemaleWeight", cAdultFemaleWeight)
        register("cJuvenileMaleWeight", cJuvenileMaleWeight)
        register("cSubAdultMaleWeight", cSubAdultMaleWeight)
        register("cAdultMaleWeight", cAdultMaleWeight)
        register("cMeanWinterTemperature", cMeanWinterTemperature)
        register("cMilkEnergyContent", cMilkEnergyContent)

        self.cAnimal = _resolve_index(cAnimal, _ANIMAL_INDEX)
        self.cActivity = _resolve_index(cActivity, _ACTIVITY_INDEX)

        self.register_buffer("_ca_table", _CA_TABLE.to(dtype=dtype).clone())
        self._dtype = dtype

    # -- helpers ----------------------------------------------------------------
    def _maintenance(self, ca: torch.Tensor, cfi: float, weight_kg: torch.Tensor,
                     temp_addon: torch.Tensor) -> torch.Tensor:
        return (ca + 1) * (temp_addon + cfi) * torch.pow(weight_kg, 0.75)

    # --------------------------------------------------------------------------
    def forward(
        self,
        iJuvenileFemale: torch.Tensor,
        iSubAdultFemale: torch.Tensor,
        iAdultFemale: torch.Tensor,
        iJuvenileMale: torch.Tensor,
        iSubAdultMale: torch.Tensor,
        iAdultMale: torch.Tensor,
        iMilkAmount: torch.Tensor,
    ) -> dict:
        ca = self._ca_table[self.cAnimal, self.cActivity]

        temp_addon = 4800 * (20 - self.cMeanWinterTemperature)

        jf_kg = self.cJuvenileFemaleWeight / 1000
        sf_kg = self.cSubAdultFemaleWeight / 1000
        af_kg = self.cAdultFemaleWeight / 1000
        jm_kg = self.cJuvenileMaleWeight / 1000
        sm_kg = self.cSubAdultMaleWeight / 1000
        am_kg = self.cAdultMaleWeight / 1000

        if self.cAnimal == _ANIMAL_INDEX["cattle"]:
            tJF_M = self._maintenance(ca, _CFI_COW_NONLACTATING, jf_kg, temp_addon)
            tSF_M = self._maintenance(ca, _CFI_COW_NONLACTATING, sf_kg, temp_addon)
            tAF_M = self._maintenance(ca, _CFI_COW_LACTATING, af_kg, temp_addon)
            tJM_M = self._maintenance(ca, _CFI_BULL, jm_kg, temp_addon)
            tSM_M = self._maintenance(ca, _CFI_BULL, sm_kg, temp_addon)
            tAM_M = self._maintenance(ca, _CFI_BULL, am_kg, temp_addon)

            # 4% fat in milk -> 1.47 + 0.4 * 4 (Java hard-codes the fat term).
            MilkEnergyDemand = iMilkAmount * (1.47 + 0.4 * 4) * 1000

            tMWMale = am_kg
            tMWFemale = af_kg

            tJF_G = 22.02 * torch.pow(jf_kg / 0.8 / tMWFemale, 0.75) * (0.9 ** 1.097) * 1_000_000
            tSF_G = 22.02 * torch.pow(sf_kg / 0.8 / tMWFemale, 0.75) * (0.4 ** 1.097) * 1_000_000
            tAF_G = 22.02 * torch.pow(af_kg / 0.8 / tMWFemale, 0.75) * (0.0 ** 1.097) * 1_000_000
            tJM_G = 22.02 * torch.pow(jm_kg / 1.0 / tMWMale, 0.75) * (0.9 ** 1.097) * 1_000_000
            tSM_G = 22.02 * torch.pow(sm_kg / 1.2 / tMWMale, 0.75) * (0.7 ** 1.097) * 1_000_000
            tAM_G = 22.02 * torch.pow(am_kg / 1.2 / tMWMale, 0.75) * (0.0 ** 1.097) * 1_000_000

        elif self.cAnimal == _ANIMAL_INDEX["sheep"]:
            tJF_M = self._maintenance(ca, _CFI_LAMB, jf_kg, temp_addon)
            tSF_M = self._maintenance(ca, _CFI_SHEEP, sf_kg, temp_addon)
            tAF_M = self._maintenance(ca, _CFI_SHEEP, af_kg, temp_addon)
            tJM_M = self._maintenance(ca, _CFI_LAMB, jm_kg, temp_addon)
            tSM_M = self._maintenance(ca, _CFI_SHEEP, sm_kg, temp_addon)
            tAM_M = self._maintenance(ca, _CFI_SHEEP, am_kg, temp_addon)

            # Sheep: 4.6 MJ/kg = 4600 J/g.
            MilkEnergyDemand = iMilkAmount * 4600

            tJF_G = 0.4 * (2.1 + 0.5 * 0.45 * (self.cJuvenileFemaleWeight + self.cAdultFemaleWeight) / 1000) * 1_000_000
            tSF_G = 0.2 * (2.1 + 0.5 * 0.45 * (self.cJuvenileFemaleWeight + self.cAdultFemaleWeight) / 1000) * 1_000_000
            tAF_G = 0.0 * (2.1 + 0.5 * 0.45 * (self.cJuvenileFemaleWeight + self.cAdultFemaleWeight) / 1000) * 1_000_000
            tJM_G = 0.5 * (2.5 + 0.5 * 0.35 * (self.cJuvenileMaleWeight + self.cAdultMaleWeight) / 1000) * 1_000_000
            tSM_G = 0.35 * (2.5 + 0.5 * 0.35 * (self.cJuvenileMaleWeight + self.cAdultMaleWeight) / 1000) * 1_000_000
            tAM_G = 0.0 * (2.5 + 0.5 * 0.35 * (self.cJuvenileMaleWeight + self.cAdultMaleWeight) / 1000) * 1_000_000
        else:
            raise ValueError(f"unsupported animal index {self.cAnimal}")

        JuvenileFemaleMaintenanceEnergyDemand = iJuvenileFemale * tJF_M
        SubAdultFemaleMaintenanceEnergyDemand = iSubAdultFemale * tSF_M
        AdultFemaleMaintenanceEnergyDemand = iAdultFemale * tAF_M
        JuvenileMaleMaintenanceEnergyDemand = iJuvenileMale * tJM_M
        SubAdultMaleMaintenanceEnergyDemand = iSubAdultMale * tSM_M
        AdultMaleMaintenanceEnergyDemand = iAdultMale * tAM_M

        JuvenileFemaleTotalEnergyDemand = JuvenileFemaleMaintenanceEnergyDemand + iJuvenileFemale * tJF_G
        SubAdultFemaleTotalEnergyDemand = SubAdultFemaleMaintenanceEnergyDemand + iSubAdultFemale * tSF_G
        AdultFemaleTotalEnergyDemand = AdultFemaleMaintenanceEnergyDemand + iAdultFemale * tAF_G
        JuvenileMaleTotalEnergyDemand = JuvenileMaleMaintenanceEnergyDemand + iJuvenileMale * tJM_G
        SubAdultMaleTotalEnergyDemand = SubAdultMaleMaintenanceEnergyDemand + iSubAdultMale * tSM_G
        AdultMaleTotalEnergyDemand = AdultMaleMaintenanceEnergyDemand + iAdultMale * tAM_G

        MaintenanceEnergyDemand = (
            JuvenileFemaleMaintenanceEnergyDemand
            + SubAdultFemaleMaintenanceEnergyDemand
            + AdultFemaleMaintenanceEnergyDemand
            + JuvenileMaleMaintenanceEnergyDemand
            + SubAdultMaleMaintenanceEnergyDemand
            + AdultMaleMaintenanceEnergyDemand
        )

        TotalEnergyDemand = (
            JuvenileFemaleTotalEnergyDemand
            + SubAdultFemaleTotalEnergyDemand
            + AdultFemaleTotalEnergyDemand
            + JuvenileMaleTotalEnergyDemand
            + SubAdultMaleTotalEnergyDemand
            + AdultMaleTotalEnergyDemand
            + MilkEnergyDemand
        )

        return {
            "JuvenileFemaleMaintenanceEnergyDemand": JuvenileFemaleMaintenanceEnergyDemand,
            "SubAdultFemaleMaintenanceEnergyDemand": SubAdultFemaleMaintenanceEnergyDemand,
            "AdultFemaleMaintenanceEnergyDemand": AdultFemaleMaintenanceEnergyDemand,
            "JuvenileMaleMaintenanceEnergyDemand": JuvenileMaleMaintenanceEnergyDemand,
            "SubAdultMaleMaintenanceEnergyDemand": SubAdultMaleMaintenanceEnergyDemand,
            "AdultMaleMaintenanceEnergyDemand": AdultMaleMaintenanceEnergyDemand,
            "JuvenileFemaleTotalEnergyDemand": JuvenileFemaleTotalEnergyDemand,
            "SubAdultFemaleTotalEnergyDemand": SubAdultFemaleTotalEnergyDemand,
            "AdultFemaleTotalEnergyDemand": AdultFemaleTotalEnergyDemand,
            "JuvenileMaleTotalEnergyDemand": JuvenileMaleTotalEnergyDemand,
            "SubAdultMaleTotalEnergyDemand": SubAdultMaleTotalEnergyDemand,
            "AdultMaleTotalEnergyDemand": AdultMaleTotalEnergyDemand,
            "MilkEnergyDemand": MilkEnergyDemand,
            "MaintenanceEnergyDemand": MaintenanceEnergyDemand,
            "TotalEnergyDemand": TotalEnergyDemand,
        }
