# Usage

## Installation

```bash
pip install torchherd
```

TorchHerd requires Python ≥ 3.8 and PyTorch ≥ 1.12. All computations use
`float64` by default to match the precision of the reference implementation.

## Quickstart: run a simulation

The fastest way to get a trajectory is [`run_simulation`][torchherd.simulation.run_simulation],
which drives the composed model forward under a constant daily forcing and
returns one plain-`float` record per day:

```python
import torchherd

records = torchherd.run_simulation(animal="Cattle", activity="Stall", days=365)

print(len(records))            # 365
print(records[-1]["total_animals"])
print(records[-1]["ch4_emission"])
```

Each record contains `day`, the six population classes, `total_animals`,
`milk_produced`, `milk_exported`, `ch4_emission`, `manure_storage` and
`sum_meat`. Because the records are ordinary dictionaries of floats, they turn
straight into a `pandas.DataFrame` or a matplotlib plot:

```python
import pandas as pd
df = pd.DataFrame(records).set_index("day")
df[["adult_female", "adult_male"]].plot()
```

Override any part of the daily forcing via `forcing=`:

```python
records = torchherd.run_simulation(
    animal="Sheep", activity="Grazing", days=730,
    forcing={"iForageProduction": 3.0e10, "iMilkAmount": 500.0},
)
```

## Driving the composed model step by step

For full control, instantiate [`LivestockModel`][torchherd.livestock_model.LivestockModel]
and call `step()` yourself. This is what `run_simulation` does internally, but
it lets you vary the forcing per day and read every intermediate output:

```python
import torch
from torchherd import LivestockModel

model = LivestockModel(animal="Cattle", activity="Stall", n=365)
model.reset_state()

def f(x):
    return torch.tensor(float(x), dtype=torch.float64)

for day in range(365):
    out = model.step(
        iMilkAmount=f(1000.0),
        iForageProduction=f(2.0e10),
        iConcentrateProduction=f(5.0e9),
        iFodderProduction=f(5.0e6),
        iFertilizeFraction=f(0.1),
        iDroppingFraction=f(0.2),
    )

print(out["population"]["sTotalAnimals"])
print(out["methane"]["CH4emission"])
```

`step()` returns a nested dict with the sections `demand`, `stress`, `supply`,
`population`, `weight`, `milk`, `fodder`, `manure`, `methane` and `sell_out`.
Pass `iCashAvail=` to enable the destocking manager.

## Using individual components

Every module is usable on its own — for example, the demographic core:

```python
import torch
from torchherd import PopulationGrowth

pop = PopulationGrowth(n=365, trainable=False)
pop.reset_state()

out = pop(
    iProductionNutritionIndex=torch.tensor(1.0, dtype=torch.float64),
    iMaintenanceNutritionIndex=torch.tensor(1.0, dtype=torch.float64),
    **{k: torch.tensor(0.0, dtype=torch.float64) for k in (
        "iJuvenileFemaleIntakeNumber", "iSubAdultFemaleIntakeNumber",
        "iAdultFemaleIntakeNumber", "iJuvenileMaleIntakeNumber",
        "iSubAdultMaleIntakeNumber", "iAdultMaleIntakeNumber",
        "iJuvenileFemaleOfftakeNumber", "iSubAdultFemaleOfftakeNumber",
        "iAdultFemaleOfftakeNumber", "iJuvenileMaleOfftakeNumber",
        "iSubAdultMaleOfftakeNumber", "iAdultMaleOfftakeNumber",
    )},
)
print(out["sTotalAnimals"])
```

## Calibrating parameters with autograd

Because the model is differentiable, you can fit parameters to a target with a
standard PyTorch optimiser. Here we nudge the adult-female mortality fraction so
the herd hits a target size after 30 days:

```python
import torch
from torchherd import LivestockModel

model = LivestockModel(animal="Cattle", activity="Stall", trainable=True)
target = torch.tensor(55000.0, dtype=torch.float64)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

forcing = dict(
    iMilkAmount=torch.tensor(1000.0, dtype=torch.float64),
    iForageProduction=torch.tensor(2.0e10, dtype=torch.float64),
    iConcentrateProduction=torch.tensor(5.0e9, dtype=torch.float64),
    iFodderProduction=torch.tensor(5.0e6, dtype=torch.float64),
    iFertilizeFraction=torch.tensor(0.1, dtype=torch.float64),
    iDroppingFraction=torch.tensor(0.2, dtype=torch.float64),
)

for epoch in range(20):
    model.reset_state()
    for _ in range(30):
        out = model.step(**forcing)
    loss = (out["population"]["sTotalAnimals"] - target) ** 2
    opt.zero_grad()
    loss.backward()
    opt.step()
```

## Command-line interface

Installing the package also installs a `torchherd` console command:

```bash
torchherd simulate --animal Cattle --activity Stall --days 365
torchherd simulate --animal Sheep --days 730 --output herd.csv
torchherd --version
```
