"""PyTorch translation of ``LivestockEnergyStress.java``.

The Java ``process()`` has four mutually-exclusive branches selecting how to
allocate forage / concentrate supply against maintenance / total demand.
Branching on tensor inputs is done with ``torch.where`` so the computational
graph stays intact — the selected branch's gradient flows back to its
contributing inputs, exactly as in the original ``if`` cascade.
"""

from __future__ import annotations

import torch
from torch import nn

from ._common import DEFAULT_DTYPE, as_tensor


class LivestockEnergyStress(nn.Module):
    """Compute energy consumption + nutrition stress indices."""

    def __init__(self, *, dtype: torch.dtype = DEFAULT_DTYPE) -> None:
        super().__init__()
        self._dtype = dtype
        # No trainable parameters in the Java class.  Buffers store an
        # arbitrary tensor so :func:`module_device` resolves to a sensible
        # device.
        self.register_buffer("_marker", torch.zeros((), dtype=dtype))

    def forward(
        self,
        iForageEnergyAvailable: torch.Tensor,
        iConcentrateEnergyAvailable: torch.Tensor,
        iMaintenanceEnergyDemand: torch.Tensor,
        iTotalEnergyDemand: torch.Tensor,
    ) -> dict:
        forage = as_tensor(iForageEnergyAvailable, dtype=self._dtype)
        conc = as_tensor(iConcentrateEnergyAvailable, dtype=self._dtype)
        maint = as_tensor(iMaintenanceEnergyDemand, dtype=self._dtype)
        total = as_tensor(iTotalEnergyDemand, dtype=self._dtype)

        supply_total = forage + conc

        # Java cascade (mirrored exactly):
        # case A:  total <= forage
        # case B:  total <= forage + concentrate
        # case C:  maintenance <= forage + concentrate < total
        # case D:  otherwise (maintenance > supply)
        case_a = total <= forage
        case_b = (~case_a) & (total <= supply_total)
        case_c = (~case_a) & (~case_b) & (maint <= supply_total)
        case_d = ~(case_a | case_b | case_c)

        zero = torch.zeros_like(forage)
        one = torch.ones_like(forage)

        # Forage consumed -----------------------------------------------------
        ForageEnergyConsumption = torch.where(case_a, total, forage)
        # Concentrate consumed ------------------------------------------------
        ConcentrateEnergyConsumption = torch.where(
            case_a, zero,
            torch.where(case_b, total - forage, conc),
        )
        # Energy consumption --------------------------------------------------
        EnergyConsumption = torch.where(
            case_a | case_b, total, supply_total,
        )
        # Production nutrition index -----------------------------------------
        # Avoid division by zero in unused branches by using a safe denom.
        denom_prod = total - maint
        safe_denom_prod = torch.where(denom_prod != 0, denom_prod, torch.ones_like(denom_prod))
        prod_c = (EnergyConsumption - maint) / safe_denom_prod
        ProductionNutritionIndex = torch.where(
            case_a | case_b, one,
            torch.where(case_c, prod_c, zero),
        )
        # Maintenance nutrition index ----------------------------------------
        safe_maint = torch.where(maint != 0, maint, torch.ones_like(maint))
        maint_d = EnergyConsumption / safe_maint
        MaintenanceNutritionIndex = torch.where(case_d, maint_d, one)

        # Forage storage factor ----------------------------------------------
        safe_total = torch.where(total != 0, total, torch.ones_like(total))
        ForageStorageFactor = (forage - ForageEnergyConsumption) / safe_total

        return {
            "EnergyConsumption": EnergyConsumption,
            "ForageEnergyConsumption": ForageEnergyConsumption,
            "ConcentrateEnergyConsumption": ConcentrateEnergyConsumption,
            "ForageStorageFactor": ForageStorageFactor,
            "ProductionNutritionIndex": ProductionNutritionIndex,
            "MaintenanceNutritionIndex": MaintenanceNutritionIndex,
        }
