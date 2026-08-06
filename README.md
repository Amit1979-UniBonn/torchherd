# torchherd

[![image](https://img.shields.io/pypi/v/torchherd.svg)](https://pypi.python.org/pypi/torchherd)
[![image](https://img.shields.io/conda/vn/conda-forge/torchherd.svg)](https://anaconda.org/conda-forge/torchherd)

**TorchHerd is an open-source, differentiable PyTorch framework for livestock
population modelling.**

TorchHerd re-implements a mechanistic, age- and sex-structured herd model as a
set of composable [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
components. Because every step is expressed with differentiable tensor
operations, the whole model can be run forward as a simulator **and**
back-propagated through — so herd parameters (mortality, offtake, feed energy
content, methane conversion factors, …) can be *calibrated with gradient
descent* against observations, or embedded inside a larger learning pipeline.

- Free software: MIT License
- Documentation: <https://Amit1979-UniBonn.github.io/torchherd>

## Features

- **End-to-end differentiable.** Every component is an `nn.Module`; run a
  multi-day simulation and call `.backward()` to get gradients w.r.t. any
  parameter.
- **Age/sex-structured demographics.** Six classes (juvenile / sub-adult /
  adult × female / male) with births, mortality, ageing transitions, intake
  and offtake, following the DynMod projection method (Lesnoff, 2008).
- **Energy-balance driven.** Feed energy supply is allocated against
  maintenance and production demand (IPCC 2006, Vol. 4, Ch. 10); the resulting
  nutrition stress feeds back into growth and mortality.
- **Whole-farm outputs.** Milk, meat/carcass weight, fodder use, manure and
  fertiliser flows, and enteric methane emission (full and "lite" variants).
- **Cash-driven management.** Optional priority-based destocking
  (`SellOutManagement`) to cover a cash deficit.
- **Cattle and sheep** presets out of the box, with `Stall` / `Pasture` /
  `Grazing` activity levels.

## Installation

```bash
pip install torchherd            # from PyPI (once released)
pip install git+https://github.com/Amit1979-UniBonn/torchherd   # from source
```

TorchHerd requires Python ≥ 3.8 and PyTorch ≥ 1.12.

## Quickstart

```python
import torchherd

# Run a 1-year cattle simulation under a constant daily forcing.
records = torchherd.run_simulation(animal="Cattle", activity="Stall", days=365)

final = records[-1]
print(final["total_animals"], final["ch4_emission"])
```

Or drive the composed model directly for full control of each timestep:

```python
import torch
from torchherd import LivestockModel

model = LivestockModel(animal="Cattle", activity="Stall")
model.reset_state()

out = model.step(
    iMilkAmount=torch.tensor(1000.0, dtype=torch.float64),
    iForageProduction=torch.tensor(2.0e10, dtype=torch.float64),
    iConcentrateProduction=torch.tensor(5.0e9, dtype=torch.float64),
    iFodderProduction=torch.tensor(5.0e6, dtype=torch.float64),
    iFertilizeFraction=torch.tensor(0.1, dtype=torch.float64),
    iDroppingFraction=torch.tensor(0.2, dtype=torch.float64),
)
print(out["population"]["sTotalAnimals"])
```

A command-line entry point is also installed:

```bash
torchherd simulate --animal Cattle --activity Stall --days 365 --output herd.csv
```

## Model components

| Module | Class | Role |
| ------ | ----- | ---- |
| `population_growth` | `PopulationGrowth` | Age/sex-structured births, deaths, ageing, intake/offtake |
| `livestock_energy_demand` | `LivestockEnergyDemand` | Maintenance + growth + lactation energy demand |
| `feed_energy_content` | `FeedEnergyContent` | Feed mass → energy content |
| `feed_energy_supply` | `FeedEnergySupply` | Forage / concentrate storage with degradation |
| `livestock_energy_stress` | `LivestockEnergyStress` | Allocate supply vs demand → nutrition indices |
| `weight_meat_production` | `WeightAndMeatProduction` | Herd live weight and carcass meat |
| `milk_production` | `MilkProduction` | Milk produced / consumed / exported |
| `fodder_consumption` | `FodderConsumption` | Fodder consumption and storage |
| `manure_production` | `ManureProduction` | Manure, dropping and fertiliser flows |
| `methane_emission` | `MethaneEmission`, `MethaneEmissionLite` | Enteric methane emission |
| `sell_out_management` | `SellOutManagement` | Cash-driven destocking |
| `livestock_model` | `LivestockModel` | Composition wiring all of the above |

See the [example notebook](docs/examples/intro.ipynb) for a narrated,
plotted walkthrough.

## Scientific background

The dynamics port and coefficients derive from:

- M. Lesnoff (2008). *DynMod: A tool for demographic projections of tropical
  livestock populations under Microsoft Excel.* CIRAD / ILRI.
- IPCC (2006). *2006 IPCC Guidelines for National Greenhouse Gas Inventories*,
  Vol. 4, Ch. 10 (Emissions from livestock and manure management).

## Contributing

Contributions are welcome — see [`docs/contributing.md`](docs/contributing.md).
Run the test suite with `pytest`.

## License

Distributed under the terms of the MIT License. See [`LICENSE`](LICENSE).
