"""Differentiable PyTorch translation of the SIMPLACE livestock model.

This package mirrors the Java components in ``net.simplace.sim.components.experimental.livestock``
as ``torch.nn.Module`` instances.  The intent is to reproduce the **exact**
arithmetic of the original Java code, while keeping the computational graph
intact so the model can be optimised end-to-end with autograd.

Sub-modules
-----------
* :mod:`functions` -- pure-function port of ``LivestockFunctions.java``.
* :mod:`feed_energy_content` -- :class:`FeedEnergyContent`.
* :mod:`feed_energy_supply` -- :class:`FeedEnergySupply`.
* :mod:`fodder_consumption` -- :class:`FodderConsumption`.
* :mod:`livestock_energy_demand` -- :class:`LivestockEnergyDemand`.
* :mod:`livestock_energy_stress` -- :class:`LivestockEnergyStress`.
* :mod:`manure_production` -- :class:`ManureProduction`.
* :mod:`methane_emission` -- :class:`MethaneEmission` / :class:`MethaneEmissionLite`.
* :mod:`milk_production` -- :class:`MilkProduction`.
* :mod:`population_growth` -- :class:`PopulationGrowth`.
* :mod:`sell_out_management` -- :class:`SellOutManagement`.
* :mod:`weight_meat_production` -- :class:`WeightAndMeatProduction`.
* :mod:`livestock_model` -- top-level :class:`LivestockModel` composing the above.
"""

from .functions import (
    Fraction,
    HfromP,
    PfromH,
    PRescaled,
    surv,
    gamma,
    calculate_population_rates,
)
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
from .livestock_model import LivestockModel
from .simulation import DEFAULT_FORCING, run_simulation

__version__ = "0.0.1"

__all__ = [
    "__version__",
    "DEFAULT_FORCING",
    "run_simulation",
    "Fraction",
    "HfromP",
    "PfromH",
    "PRescaled",
    "surv",
    "gamma",
    "calculate_population_rates",
    "FeedEnergyContent",
    "FeedEnergySupply",
    "FodderConsumption",
    "LivestockEnergyDemand",
    "LivestockEnergyStress",
    "ManureProduction",
    "MethaneEmission",
    "MethaneEmissionLite",
    "MilkProduction",
    "PopulationGrowth",
    "SellOutManagement",
    "WeightAndMeatProduction",
    "LivestockModel",
]
