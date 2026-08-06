# FAQ

## What is TorchHerd?

A differentiable PyTorch re-implementation of a mechanistic, age- and
sex-structured livestock herd model. It runs as a forward simulator and, because
every step is differentiable, supports gradient-based calibration of its
parameters.

## Which animals are supported?

`Cattle` and `Sheep`, each with `Stall`, `Pasture` or `Grazing` activity levels.
These select the maintenance and activity coefficients used in the energy-demand
component.

## Why does everything use `float64`?

The model ports a reference implementation that relies on double-precision
arithmetic. `float64` avoids accumulation error over long simulations and keeps
results reproducible. You can pass `dtype=torch.float32` to the modules if you
prefer speed over precision.

## How do I calibrate parameters to data?

Build the model with `trainable=True`, run a simulation, define a loss against
your observations, and optimise with any `torch.optim` optimiser. See the
"Calibrating parameters with autograd" section of the [Usage](usage.md) page.

## Do I need a GPU?

No. The model is small and runs comfortably on CPU. It will still run on a GPU
if you move the modules and inputs to a CUDA device.

## Where do the coefficients come from?

From the DynMod demographic method (Lesnoff, 2008) and the IPCC 2006 Guidelines
(Vol. 4, Ch. 10). See the [home page](index.md) for full references.
