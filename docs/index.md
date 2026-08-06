# Welcome to torchherd

[![image](https://img.shields.io/pypi/v/torchherd.svg)](https://pypi.python.org/pypi/torchherd)

**TorchHerd is an open-source, differentiable PyTorch framework for livestock
population modelling.**

TorchHerd re-implements a mechanistic, age- and sex-structured herd model as a
set of composable [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
components. Because every step is expressed with differentiable tensor
operations, the whole model can be run forward as a simulator **and**
back-propagated through — so herd parameters can be *calibrated with gradient
descent* against observations, or embedded inside a larger learning pipeline.

- Free software: MIT License
- Documentation: <https://Amit1979-UniBonn.github.io/torchherd>

## Features

- **End-to-end differentiable** — every component is an `nn.Module`; run a
  multi-day simulation and call `.backward()` for gradients w.r.t. any
  parameter.
- **Age/sex-structured demographics** — six classes with births, mortality,
  ageing transitions, intake and offtake (DynMod method, Lesnoff 2008).
- **Energy-balance driven** — feed supply allocated against maintenance and
  production demand (IPCC 2006, Vol. 4, Ch. 10), with nutrition stress feeding
  back into growth and mortality.
- **Whole-farm outputs** — milk, meat, fodder, manure/fertiliser flows, and
  enteric methane emission.
- **Cash-driven management** — optional priority-based destocking.
- **Cattle and sheep** presets with `Stall` / `Pasture` / `Grazing` activity.

## Where to next

- [Installation](installation.md)
- [Usage](usage.md) — quickstart, driving the composed model, and calibration.
- [Example notebook](examples/intro.ipynb) — a narrated, plotted walkthrough.
- [API reference](torchherd.md)
