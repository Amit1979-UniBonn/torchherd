"""PyTorch translation of ``PopulationGrowth.java``.

Wraps :func:`functions.calculate_population_rates` with the per-class
``nn.Parameter``/buffer plumbing the Java class declared.

The Java module mutates population states in place each ``process()`` call.
The PyTorch version mirrors this: state tensors are stored as attributes
(not buffers) and reassigned out-of-place each forward, which keeps the
autograd graph alive across consecutive calls when the caller wants to
back-propagate through multiple time steps.
"""

from __future__ import annotations

import torch
from torch import nn

from ._common import DEFAULT_DTYPE, make_parameter, as_tensor
from .functions import Fraction, calculate_population_rates


class PopulationGrowth(nn.Module):
    def __init__(
        self,
        *,
        # Maxima
        cJuvenileFemaleMaxNumber: float = 10000.0,
        cSubAdultFemaleMaxNumber: float = 10000.0,
        cAdultFemaleMaxNumber: float = 10000.0,
        cJuvenileMaleMaxNumber: float = 10000.0,
        cSubAdultMaleMaxNumber: float = 10000.0,
        cAdultMaleMaxNumber: float = 10000.0,
        # Initial
        cJuvenileFemaleInitial: float = 10000.0,
        cSubAdultFemaleInitial: float = 10000.0,
        cAdultFemaleInitial: float = 10000.0,
        cJuvenileMaleInitial: float = 10000.0,
        cSubAdultMaleInitial: float = 10000.0,
        cAdultMaleInitial: float = 10000.0,
        # Durations (years)
        cJuvenileFemaleDuration: float = 1.0,
        cSubAdultFemaleDuration: float = 3.0,
        cAdultFemaleDuration: float = 11.0,
        cJuvenileMaleDuration: float = 1.0,
        cSubAdultMaleDuration: float = 3.0,
        cAdultMaleDuration: float = 6.0,
        # Reproduction
        cProlification: float = 1.0,
        cParturition: float = 0.5,
        cFemaleAtBirth: float = 0.5,
        # Mortality (yearly)
        cJuvenileFemaleMortalityFraction: float = 0.13,
        cSubAdultFemaleMortalityFraction: float = 0.05,
        cAdultFemaleMortalityFraction: float = 0.03,
        cJuvenileMaleMortalityFraction: float = 0.13,
        cSubAdultMaleMortalityFraction: float = 0.05,
        cAdultMaleMortalityFraction: float = 0.03,
        # Offtake (daily, multiplied by 365 in process)
        cJuvenileFemaleOfftakeFraction: float = 0.0,
        cSubAdultFemaleOfftakeFraction: float = 0.0,
        cAdultFemaleOfftakeFraction: float = 0.0,
        cJuvenileMaleOfftakeFraction: float = 0.0,
        cSubAdultMaleOfftakeFraction: float = 0.0,
        cAdultMaleOfftakeFraction: float = 0.0,
        cMaintenanceNutritionMortalityFactor: float = 2.0,
        dtype: torch.dtype = DEFAULT_DTYPE,
        trainable: bool = True,
        n: int = 365,
    ) -> None:
        super().__init__()
        register = (lambda name, v: setattr(self, name, make_parameter(v, dtype=dtype))) \
            if trainable else \
            (lambda name, v: self.register_buffer(name, as_tensor(v, dtype=dtype)))

        for name, value in dict(
            cJuvenileFemaleMaxNumber=cJuvenileFemaleMaxNumber,
            cSubAdultFemaleMaxNumber=cSubAdultFemaleMaxNumber,
            cAdultFemaleMaxNumber=cAdultFemaleMaxNumber,
            cJuvenileMaleMaxNumber=cJuvenileMaleMaxNumber,
            cSubAdultMaleMaxNumber=cSubAdultMaleMaxNumber,
            cAdultMaleMaxNumber=cAdultMaleMaxNumber,
            cJuvenileFemaleDuration=cJuvenileFemaleDuration,
            cSubAdultFemaleDuration=cSubAdultFemaleDuration,
            cAdultFemaleDuration=cAdultFemaleDuration,
            cJuvenileMaleDuration=cJuvenileMaleDuration,
            cSubAdultMaleDuration=cSubAdultMaleDuration,
            cAdultMaleDuration=cAdultMaleDuration,
            cProlification=cProlification,
            cParturition=cParturition,
            cFemaleAtBirth=cFemaleAtBirth,
            cJuvenileFemaleMortalityFraction=cJuvenileFemaleMortalityFraction,
            cSubAdultFemaleMortalityFraction=cSubAdultFemaleMortalityFraction,
            cAdultFemaleMortalityFraction=cAdultFemaleMortalityFraction,
            cJuvenileMaleMortalityFraction=cJuvenileMaleMortalityFraction,
            cSubAdultMaleMortalityFraction=cSubAdultMaleMortalityFraction,
            cAdultMaleMortalityFraction=cAdultMaleMortalityFraction,
            cJuvenileFemaleOfftakeFraction=cJuvenileFemaleOfftakeFraction,
            cSubAdultFemaleOfftakeFraction=cSubAdultFemaleOfftakeFraction,
            cAdultFemaleOfftakeFraction=cAdultFemaleOfftakeFraction,
            cJuvenileMaleOfftakeFraction=cJuvenileMaleOfftakeFraction,
            cSubAdultMaleOfftakeFraction=cSubAdultMaleOfftakeFraction,
            cAdultMaleOfftakeFraction=cAdultMaleOfftakeFraction,
            cMaintenanceNutritionMortalityFactor=cMaintenanceNutritionMortalityFactor,
        ).items():
            register(name, value)

        # Initials are scenario configuration, not optimisation targets.
        for name, value in dict(
            cJuvenileFemaleInitial=cJuvenileFemaleInitial,
            cSubAdultFemaleInitial=cSubAdultFemaleInitial,
            cAdultFemaleInitial=cAdultFemaleInitial,
            cJuvenileMaleInitial=cJuvenileMaleInitial,
            cSubAdultMaleInitial=cSubAdultMaleInitial,
            cAdultMaleInitial=cAdultMaleInitial,
        ).items():
            self.register_buffer(name, as_tensor(value, dtype=dtype))

        self._dtype = dtype
        self.n = n
        self.reset_state()

    # -- state ------------------------------------------------------------------
    def reset_state(self) -> None:
        self.sJuvenileFemale = self.cJuvenileFemaleInitial.clone()
        self.sSubAdultFemale = self.cSubAdultFemaleInitial.clone()
        self.sAdultFemale = self.cAdultFemaleInitial.clone()
        self.sJuvenileMale = self.cJuvenileMaleInitial.clone()
        self.sSubAdultMale = self.cSubAdultMaleInitial.clone()
        self.sAdultMale = self.cAdultMaleInitial.clone()
        self.sTotalAnimals = (
            self.sJuvenileFemale + self.sSubAdultFemale + self.sAdultFemale
            + self.sJuvenileMale + self.sSubAdultMale + self.sAdultMale
        )

    # --------------------------------------------------------------------------
    def forward(
        self,
        *,
        iProductionNutritionIndex: torch.Tensor,
        iMaintenanceNutritionIndex: torch.Tensor,
        iJuvenileFemaleIntakeNumber: torch.Tensor,
        iSubAdultFemaleIntakeNumber: torch.Tensor,
        iAdultFemaleIntakeNumber: torch.Tensor,
        iJuvenileMaleIntakeNumber: torch.Tensor,
        iSubAdultMaleIntakeNumber: torch.Tensor,
        iAdultMaleIntakeNumber: torch.Tensor,
        iJuvenileFemaleOfftakeNumber: torch.Tensor,
        iSubAdultFemaleOfftakeNumber: torch.Tensor,
        iAdultFemaleOfftakeNumber: torch.Tensor,
        iJuvenileMaleOfftakeNumber: torch.Tensor,
        iSubAdultMaleOfftakeNumber: torch.Tensor,
        iAdultMaleOfftakeNumber: torch.Tensor,
        iDoInitialize: bool = False,
    ) -> dict:
        if iDoInitialize:
            self.reset_state()

        state = [
            self.sJuvenileFemale, self.sSubAdultFemale, self.sAdultFemale,
            self.sJuvenileMale, self.sSubAdultMale, self.sAdultMale,
        ]
        pdeath = [
            self.cJuvenileFemaleMortalityFraction, self.cSubAdultFemaleMortalityFraction,
            self.cAdultFemaleMortalityFraction, self.cJuvenileMaleMortalityFraction,
            self.cSubAdultMaleMortalityFraction, self.cAdultMaleMortalityFraction,
        ]
        # Java multiplies by 365 — yearly probability passed downstream.
        pofftake = [
            self.cJuvenileFemaleOfftakeFraction * 365, self.cSubAdultFemaleOfftakeFraction * 365,
            self.cAdultFemaleOfftakeFraction * 365, self.cJuvenileMaleOfftakeFraction * 365,
            self.cSubAdultMaleOfftakeFraction * 365, self.cAdultMaleOfftakeFraction * 365,
        ]
        duration = [
            self.cJuvenileFemaleDuration, self.cSubAdultFemaleDuration, self.cAdultFemaleDuration,
            self.cJuvenileMaleDuration, self.cSubAdultMaleDuration, self.cAdultMaleDuration,
        ]
        intake = [
            iJuvenileFemaleIntakeNumber, iSubAdultFemaleIntakeNumber, iAdultFemaleIntakeNumber,
            iJuvenileMaleIntakeNumber, iSubAdultMaleIntakeNumber, iAdultMaleIntakeNumber,
        ]
        offtake = [
            iJuvenileFemaleOfftakeNumber, iSubAdultFemaleOfftakeNumber, iAdultFemaleOfftakeNumber,
            iJuvenileMaleOfftakeNumber, iSubAdultMaleOfftakeNumber, iAdultMaleOfftakeNumber,
        ]
        a_max = [
            self.cJuvenileFemaleMaxNumber, self.cSubAdultFemaleMaxNumber, self.cAdultFemaleMaxNumber,
            self.cJuvenileMaleMaxNumber, self.cSubAdultMaleMaxNumber, self.cAdultMaleMaxNumber,
        ]

        # Java: intake[i] -= min(state[i], offtake[i]) — adjust per-class intake
        # so that user-supplied offtake reduces net inflow.
        intake = [
            intake_i - torch.minimum(state_i, off_i)
            for intake_i, state_i, off_i in zip(intake, state, offtake)
        ]

        rates = calculate_population_rates(
            state=state,
            pdeath=pdeath,
            pofftake=pofftake,
            duration=duration,
            intake=intake,
            production_nutrition_index=iProductionNutritionIndex,
            maintenance_nutrition_index=iMaintenanceNutritionIndex,
            maintenance_nutrition_mortality_factor=self.cMaintenanceNutritionMortalityFactor,
            a_max=a_max,
            prol=self.cProlification,
            parturition=self.cParturition,
            female_proportion=self.cFemaleAtBirth,
            n=self.n,
            dtype=self._dtype,
        )
        dRate = rates["dRate"]

        # State update (out-of-place).
        self.sJuvenileFemale = self.sJuvenileFemale + dRate[Fraction.FJ]
        self.sSubAdultFemale = self.sSubAdultFemale + dRate[Fraction.FS]
        self.sAdultFemale = self.sAdultFemale + dRate[Fraction.FA]
        self.sJuvenileMale = self.sJuvenileMale + dRate[Fraction.MJ]
        self.sSubAdultMale = self.sSubAdultMale + dRate[Fraction.MS]
        self.sAdultMale = self.sAdultMale + dRate[Fraction.MA]
        self.sTotalAnimals = (
            self.sJuvenileFemale + self.sSubAdultFemale + self.sAdultFemale
            + self.sJuvenileMale + self.sSubAdultMale + self.sAdultMale
        )

        Balance = rates["Balance"]
        dBirth = rates["dBirth"]
        dDeath = rates["dDeath"]
        dIntake = rates["dIntake"]
        dOfftake = rates["dOfftake"]

        return {
            "rJuvenileFemaleRate": dRate[Fraction.FJ],
            "rSubAdultFemaleRate": dRate[Fraction.FS],
            "rAdultFemaleRate": dRate[Fraction.FA],
            "rJuvenileMaleRate": dRate[Fraction.MJ],
            "rSubAdultMaleRate": dRate[Fraction.MS],
            "rAdultMaleRate": dRate[Fraction.MA],
            "rJuvenileFemaleBirthRate": dBirth[Fraction.FJ],
            "rJuvenileMaleBirthRate": dBirth[Fraction.MJ],
            "rJuvenileFemaleMortalityRate": dDeath[Fraction.FJ],
            "rSubAdultFemaleMortalityRate": dDeath[Fraction.FS],
            "rAdultFemaleMortalityRate": dDeath[Fraction.FA],
            "rJuvenileMaleMortalityRate": dDeath[Fraction.MJ],
            "rSubAdultMaleMortalityRate": dDeath[Fraction.MS],
            "rAdultMaleMortalityRate": dDeath[Fraction.MA],
            "rJuvenileFemaleIntakeRate": dIntake[Fraction.FJ],
            "rSubAdultFemaleIntakeRate": dIntake[Fraction.FS],
            "rAdultFemaleIntakeRate": dIntake[Fraction.FA],
            "rJuvenileMaleIntakeRate": dIntake[Fraction.MJ],
            "rSubAdultMaleIntakeRate": dIntake[Fraction.MS],
            "rAdultMaleIntakeRate": dIntake[Fraction.MA],
            "rJuvenileFemaleOfftakeRate": dOfftake[Fraction.FJ],
            "rSubAdultFemaleOfftakeRate": dOfftake[Fraction.FS],
            "rAdultFemaleOfftakeRate": dOfftake[Fraction.FA],
            "rJuvenileMaleOfftakeRate": dOfftake[Fraction.MJ],
            "rSubAdultMaleOfftakeRate": dOfftake[Fraction.MS],
            "rAdultMaleOfftakeRate": dOfftake[Fraction.MA],
            "sJuvenileFemale": self.sJuvenileFemale,
            "sSubAdultFemale": self.sSubAdultFemale,
            "sAdultFemale": self.sAdultFemale,
            "sJuvenileMale": self.sJuvenileMale,
            "sSubAdultMale": self.sSubAdultMale,
            "sAdultMale": self.sAdultMale,
            "sTotalAnimals": self.sTotalAnimals,
            "LivestockBalance": Balance[0],
            "LivestockBalanceFemale": Balance[1],
            "LivestockBalanceMale": Balance[2],
        }
