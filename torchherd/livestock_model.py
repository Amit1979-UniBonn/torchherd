"""Top-level :class:`LivestockModel` composing every translated sub-module.

Provides a single ``step()`` method that wires the sub-modules together in
the order they would naturally execute over a daily timestep:

  1. ``LivestockEnergyDemand``  - compute the energy demand from current pop.
  2. ``FeedEnergyContent``      - convert feed mass produced today to joules.
  3. ``LivestockEnergyStress``  - allocate supply against demand, derive
                                  production/maintenance nutrition indices.
  4. ``FeedEnergySupply``       - update storage with today's intake/outflow.
  5. ``PopulationGrowth``       - advance state by one timestep.
  6. ``WeightAndMeatProduction`` - aggregate weights/meat from new population.
  7. ``MilkProduction``         - milk produced / consumed / exported.
  8. ``FodderConsumption``      - fodder rate + storage update (uses weights).
  9. ``ManureProduction``       - manure rate + storage flows.
  10. ``MethaneEmission``       - daily methane emission.
  11. ``SellOutManagement``     - optional sell-out to cover cash deficit.

The Java simulation engine wires components via shared variables; we mirror
that by passing tensors between modules.  Each sub-module remains usable on
its own — :class:`LivestockModel` is a convenience composition.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ._common import DEFAULT_DTYPE, as_tensor
from .feed_energy_content import FeedEnergyContent
from .feed_energy_supply import FeedEnergySupply
from .fodder_consumption import FodderConsumption
from .livestock_energy_demand import LivestockEnergyDemand
from .livestock_energy_stress import LivestockEnergyStress
from .manure_production import ManureProduction
from .methane_emission import MethaneEmission, MethaneEmissionLite
from .milk_production import MilkProduction
from .population_growth import PopulationGrowth
from .sell_out_management import SellOutManagement
from .weight_meat_production import WeightAndMeatProduction


class LivestockModel(nn.Module):
    """Composed differentiable livestock model."""

    def __init__(
        self,
        *,
        animal: str = "Cattle",
        activity: str = "Stall",
        dtype: torch.dtype = DEFAULT_DTYPE,
        trainable: bool = True,
        use_lite_methane: bool = False,
        n: int = 365,
        forage_energy_contents: Optional[torch.Tensor] = None,
        concentrate_energy_contents: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self._dtype = dtype

        self.energy_demand = LivestockEnergyDemand(
            cAnimal=animal, cActivity=activity, dtype=dtype, trainable=trainable,
        )
        self.forage_content = FeedEnergyContent(
            c_energy_contents=forage_energy_contents,
            dtype=dtype, trainable=trainable,
        )
        self.concentrate_content = FeedEnergyContent(
            c_energy_contents=concentrate_energy_contents,
            dtype=dtype, trainable=trainable,
        )
        self.energy_stress = LivestockEnergyStress(dtype=dtype)
        self.feed_supply = FeedEnergySupply(dtype=dtype, trainable=trainable)
        self.population = PopulationGrowth(dtype=dtype, trainable=trainable, n=n)
        self.weight_meat = WeightAndMeatProduction(dtype=dtype, trainable=trainable)
        self.milk = MilkProduction(dtype=dtype, trainable=trainable)
        self.fodder = FodderConsumption(dtype=dtype, trainable=trainable)
        self.manure = ManureProduction(dtype=dtype, trainable=trainable)
        if use_lite_methane:
            self.methane: nn.Module = MethaneEmissionLite(dtype=dtype)
        else:
            self.methane = MethaneEmission(dtype=dtype, trainable=trainable)
        self.use_lite_methane = use_lite_methane
        self.sell_out = SellOutManagement(dtype=dtype, trainable=trainable)

    # --------------------------------------------------------------------------
    def reset_state(self) -> None:
        self.feed_supply.reset_state()
        self.population.reset_state()
        self.weight_meat.reset_state()
        self.fodder.reset_state()
        self.manure.reset_state()

    # --------------------------------------------------------------------------
    def step(
        self,
        *,
        iMilkAmount: torch.Tensor,
        iForageProduction: torch.Tensor,
        iConcentrateProduction: torch.Tensor,
        iFodderProduction: torch.Tensor,
        iFertilizeFraction: torch.Tensor,
        iDroppingFraction: torch.Tensor,
        iJuvenileFemaleIntake: Optional[torch.Tensor] = None,
        iSubAdultFemaleIntake: Optional[torch.Tensor] = None,
        iAdultFemaleIntake: Optional[torch.Tensor] = None,
        iJuvenileMaleIntake: Optional[torch.Tensor] = None,
        iSubAdultMaleIntake: Optional[torch.Tensor] = None,
        iAdultMaleIntake: Optional[torch.Tensor] = None,
        iJuvenileFemaleOfftake: Optional[torch.Tensor] = None,
        iSubAdultFemaleOfftake: Optional[torch.Tensor] = None,
        iAdultFemaleOfftake: Optional[torch.Tensor] = None,
        iJuvenileMaleOfftake: Optional[torch.Tensor] = None,
        iSubAdultMaleOfftake: Optional[torch.Tensor] = None,
        iAdultMaleOfftake: Optional[torch.Tensor] = None,
        iCashAvail: Optional[torch.Tensor] = None,
    ) -> dict:
        """Advance the model by a single (daily) timestep."""

        zero = torch.zeros((), dtype=self._dtype, device=self.population.sJuvenileFemale.device)

        def default(x):
            return x if x is not None else zero

        # -- 1. energy demand from current population ---------------------------
        pop = self.population
        demand_out = self.energy_demand(
            iJuvenileFemale=pop.sJuvenileFemale,
            iSubAdultFemale=pop.sSubAdultFemale,
            iAdultFemale=pop.sAdultFemale,
            iJuvenileMale=pop.sJuvenileMale,
            iSubAdultMale=pop.sSubAdultMale,
            iAdultMale=pop.sAdultMale,
            iMilkAmount=iMilkAmount,
        )

        # -- 2. convert mass production to energy ------------------------------
        # ``iForageProduction``/``iConcentrateProduction`` are 1-or-5 tensor.
        forage_energy = self.forage_content(*self._split5(iForageProduction))
        conc_energy = self.concentrate_content(*self._split5(iConcentrateProduction))

        # -- 3. allocate supply -------------------------------------------------
        # Use the *current* storage as the "available" energy, plus today's
        # incoming production.
        forage_avail = self.feed_supply.sForageStorage + forage_energy
        conc_avail = self.feed_supply.sConcentrateStorage + conc_energy
        stress_out = self.energy_stress(
            iForageEnergyAvailable=forage_avail,
            iConcentrateEnergyAvailable=conc_avail,
            iMaintenanceEnergyDemand=demand_out["MaintenanceEnergyDemand"],
            iTotalEnergyDemand=demand_out["TotalEnergyDemand"],
        )

        # -- 4. apply today's intake to storage --------------------------------
        supply_out = self.feed_supply(
            iForageEnergyConsumption=stress_out["ForageEnergyConsumption"],
            iConcentrateEnergyConsumption=stress_out["ConcentrateEnergyConsumption"],
            iForageProductionEnergyContent=forage_energy,
            iConcentrateProductionEnergyContent=conc_energy,
        )

        # -- 5. population dynamics --------------------------------------------
        pop_out = self.population(
            iProductionNutritionIndex=stress_out["ProductionNutritionIndex"],
            iMaintenanceNutritionIndex=stress_out["MaintenanceNutritionIndex"],
            iJuvenileFemaleIntakeNumber=default(iJuvenileFemaleIntake),
            iSubAdultFemaleIntakeNumber=default(iSubAdultFemaleIntake),
            iAdultFemaleIntakeNumber=default(iAdultFemaleIntake),
            iJuvenileMaleIntakeNumber=default(iJuvenileMaleIntake),
            iSubAdultMaleIntakeNumber=default(iSubAdultMaleIntake),
            iAdultMaleIntakeNumber=default(iAdultMaleIntake),
            iJuvenileFemaleOfftakeNumber=default(iJuvenileFemaleOfftake),
            iSubAdultFemaleOfftakeNumber=default(iSubAdultFemaleOfftake),
            iAdultFemaleOfftakeNumber=default(iAdultFemaleOfftake),
            iJuvenileMaleOfftakeNumber=default(iJuvenileMaleOfftake),
            iSubAdultMaleOfftakeNumber=default(iSubAdultMaleOfftake),
            iAdultMaleOfftakeNumber=default(iAdultMaleOfftake),
        )

        # -- 6/7/8 aggregate herd-derived outputs ------------------------------
        weight_out = self.weight_meat(
            iJuvenileFemale=pop_out["sJuvenileFemale"],
            iSubAdultFemale=pop_out["sSubAdultFemale"],
            iAdultFemale=pop_out["sAdultFemale"],
            iJuvenileMale=pop_out["sJuvenileMale"],
            iSubAdultMale=pop_out["sSubAdultMale"],
            iAdultMale=pop_out["sAdultMale"],
        )
        milk_out = self.milk(
            iJuvenileMale=pop_out["sJuvenileMale"],
            iJuvenileFemale=pop_out["sJuvenileFemale"],
            iAdultFemale=pop_out["sAdultFemale"],
        )
        fodder_out = self.fodder(
            iJuvenileFemaleWeight=weight_out["JuvenileFemaleWeight"],
            iSubAdultFemaleWeight=weight_out["SubAdultFemaleWeight"],
            iAdultFemaleWeight=weight_out["AdultFemaleWeight"],
            iJuvenileMaleWeight=weight_out["JuvenileMaleWeight"],
            iSubAdultMaleWeight=weight_out["SubAdultMaleWeight"],
            iAdultMaleWeight=weight_out["AdultMaleWeight"],
            iFodderProduction=iFodderProduction,
        )

        # -- 9/10 manure & methane ---------------------------------------------
        manure_out = self.manure(
            iJuvenileFemale=pop_out["sJuvenileFemale"],
            iSubAdultFemale=pop_out["sSubAdultFemale"],
            iAdultFemale=pop_out["sAdultFemale"],
            iJuvenileMale=pop_out["sJuvenileMale"],
            iSubAdultMale=pop_out["sSubAdultMale"],
            iAdultMale=pop_out["sAdultMale"],
            iFertilizeFraction=iFertilizeFraction,
            iDroppingFraction=iDroppingFraction,
            iMaintenanceNutritionIndex=stress_out["MaintenanceNutritionIndex"],
        )
        if self.use_lite_methane:
            methane_out = self.methane(iEnergyConsumption=stress_out["EnergyConsumption"])
        else:
            methane_out = self.methane(
                iJuvenileFemale=pop_out["sJuvenileFemale"],
                iSubAdultFemale=pop_out["sSubAdultFemale"],
                iAdultFemale=pop_out["sAdultFemale"],
                iJuvenileMale=pop_out["sJuvenileMale"],
                iSubAdultMale=pop_out["sSubAdultMale"],
                iAdultMale=pop_out["sAdultMale"],
                iEnergyConsumption=stress_out["EnergyConsumption"],
            )

        # -- 11 sell-out (optional) --------------------------------------------
        if iCashAvail is not None:
            sell_out = self.sell_out(
                iNumberGroup1=pop_out["sAdultMale"],
                iNumberGroup2=pop_out["sSubAdultMale"],
                iNumberGroup3=pop_out["sJuvenileMale"],
                iNumberGroup4=pop_out["sJuvenileFemale"],
                iNumberGroup5=pop_out["sSubAdultFemale"],
                iNumberGroup6=pop_out["sAdultFemale"],
                iCashAvail=iCashAvail,
            )
        else:
            sell_out = None

        return {
            "demand": demand_out,
            "stress": stress_out,
            "supply": supply_out,
            "population": pop_out,
            "weight": weight_out,
            "milk": milk_out,
            "fodder": fodder_out,
            "manure": manure_out,
            "methane": methane_out,
            "sell_out": sell_out,
        }

    @staticmethod
    def _split5(x: torch.Tensor):
        """Split a 1-d 5-vector into 5 0-d tensors, padding with ``None``."""
        x = as_tensor(x)
        if x.dim() == 0:
            return (x, None, None, None, None)
        if x.shape[-1] == 5:
            parts = list(x.unbind(-1))
            return tuple(parts)
        # Pad with None to length 5.
        n = x.shape[-1]
        parts = list(x.unbind(-1))
        parts.extend([None] * (5 - n))
        return tuple(parts)

    def forward(self, **kwargs):
        return self.step(**kwargs)
