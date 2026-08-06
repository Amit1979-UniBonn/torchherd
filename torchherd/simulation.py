"""Convenience driver for running a :class:`~torchherd.livestock_model.LivestockModel`.

The individual ``nn.Module`` components (and :class:`LivestockModel` that
composes them) expose a single-timestep :meth:`step`.  Most users, however,
just want to push a herd forward over a number of days under a constant daily
forcing and collect the trajectory.  :func:`run_simulation` does exactly that
and returns plain-``float`` records that are trivial to turn into a CSV, a
``pandas.DataFrame``, or a matplotlib plot — without adding any runtime
dependency beyond :mod:`torch`.

Example
-------
>>> from torchherd import run_simulation
>>> records = run_simulation(animal="Cattle", activity="Stall", days=30)
>>> records[-1]["total_animals"]  # doctest: +SKIP
58575.0
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from ._common import DEFAULT_DTYPE
from .livestock_model import LivestockModel

#: Default constant daily forcing used when a field is not overridden.  Values
#: are given in the native units of :meth:`LivestockModel.step` (joules for the
#: feed production terms, grams for milk, fractions for the manure splits).
DEFAULT_FORCING: Dict[str, float] = {
    "iMilkAmount": 1000.0,
    "iForageProduction": 2.0e10,
    "iConcentrateProduction": 5.0e9,
    "iFodderProduction": 5.0e6,
    "iFertilizeFraction": 0.1,
    "iDroppingFraction": 0.2,
}


def _f(tensor: torch.Tensor) -> float:
    """Detach + convert to ``float`` (avoids the requires_grad scalar warning)."""
    return float(tensor.detach())


def run_simulation(
    model: Optional[LivestockModel] = None,
    *,
    days: int = 365,
    animal: str = "Cattle",
    activity: str = "Stall",
    forcing: Optional[Dict[str, float]] = None,
    dtype: torch.dtype = DEFAULT_DTYPE,
    trainable: bool = False,
    n: int = 365,
) -> List[Dict[str, float]]:
    """Run a herd forward under a constant daily forcing and collect the trajectory.

    Parameters
    ----------
    model:
        An existing :class:`LivestockModel` to drive.  When ``None`` (default) a
        fresh model is built from ``animal``/``activity``/``dtype``/``trainable``.
        The model's state is reset before the run starts.
    days:
        Number of daily timesteps to simulate.
    animal, activity:
        Passed to :class:`LivestockModel` when ``model is None``.  ``animal`` is
        one of ``{"Cattle", "Sheep"}``; ``activity`` one of
        ``{"Stall", "Pasture", "Grazing"}``.
    forcing:
        Overrides for :data:`DEFAULT_FORCING`.  Any subset of the keys may be
        provided; missing keys fall back to the defaults.
    dtype, trainable, n:
        Forwarded to :class:`LivestockModel` when it is constructed here.

    Returns
    -------
    list of dict
        One record per simulated day.  Each record is a flat mapping of
        ``str -> float`` with the day index and the headline population,
        production, and emission variables.  After the call, the live (autograd-
        connected) tensors remain available on ``model`` for backpropagation.
    """
    if model is None:
        model = LivestockModel(
            animal=animal, activity=activity, dtype=dtype, trainable=trainable, n=n,
        )
    model.reset_state()

    values = dict(DEFAULT_FORCING)
    if forcing:
        values.update(forcing)
    step_kwargs = {k: torch.as_tensor(v, dtype=dtype) for k, v in values.items()}

    records: List[Dict[str, float]] = []
    for day in range(1, days + 1):
        out = model.step(**step_kwargs)
        pop = out["population"]
        records.append(
            {
                "day": day,
                "total_animals": _f(pop["sTotalAnimals"]),
                "juvenile_female": _f(pop["sJuvenileFemale"]),
                "subadult_female": _f(pop["sSubAdultFemale"]),
                "adult_female": _f(pop["sAdultFemale"]),
                "juvenile_male": _f(pop["sJuvenileMale"]),
                "subadult_male": _f(pop["sSubAdultMale"]),
                "adult_male": _f(pop["sAdultMale"]),
                "milk_produced": _f(out["milk"]["MilkProduced"]),
                "milk_exported": _f(out["milk"]["MilkExported"]),
                "ch4_emission": _f(out["methane"]["CH4emission"]),
                "manure_storage": _f(out["manure"]["sManureStorage"]),
                "sum_meat": _f(out["weight"]["SumMeat"]),
            }
        )
    return records
