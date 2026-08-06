"""Functional port of ``LivestockFunctions.java``.

All routines operate on :class:`torch.Tensor` inputs and return tensors that
remain part of the autograd graph.  Hard ``if`` branches that depend only on
constants (e.g. the per-fraction branching in
:func:`calculate_population_rates`) are kept as ordinary Python control flow;
data-dependent branches (``if(p>0)``, ``if(h>0)``) are expressed with
:func:`torch.where`, which preserves gradients on the selected branch.

The reference implementation:

* M. Lesnoff, 2008, DynMod: A tool for demographic projections of tropical
  Livestock population under Microsoft Excel, User Manual, Version 1.
  CIRAD, ILRI.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, Sequence

import torch

from ._common import DEFAULT_DTYPE, as_tensor


class Fraction(IntEnum):
    """Population fractions mirroring ``LivestockFunctions.Fraction``."""

    FJ = 0  # Female Juvenile
    FS = 1  # Female Sub-Adult
    FA = 2  # Female Adult
    MJ = 3  # Male Juvenile
    MS = 4  # Male Sub-Adult
    MA = 5  # Male Adult


def HfromP(p: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """Hazard rate from a probability and its concurrent probability.

    Java source::

        if(p>0) h = -log(1 - (p + p2)) / (1 + p2 / p);
        else    h = 0;
    """
    p = as_tensor(p)
    p2 = as_tensor(p2)
    # Avoid 0/0 inside the non-selected branch by clamping the denominator
    # away from zero before applying ``torch.where``.
    safe_p = torch.where(p > 0, p, torch.ones_like(p))
    h = -torch.log1p(-(p + p2)) / (1 + p2 / safe_p)
    return torch.where(p > 0, h, torch.zeros_like(h))


def PfromH(h: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    """Probability from a hazard rate and its concurrent hazard rate.

    Java source::

        if(h>0) p = h / (h + h2) * (1 - exp(-(h + h2)));
        else    p = 0;
    """
    h = as_tensor(h)
    h2 = as_tensor(h2)
    total = h + h2
    safe_total = torch.where(total > 0, total, torch.ones_like(total))
    p = h / safe_total * (1 - torch.exp(-total))
    return torch.where(h > 0, p, torch.zeros_like(p))


def PRescaled(p: torch.Tensor, p2: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Rescale a probability to a different time unit.

    ``PfromH(HfromP(p, p2) / n, HfromP(p2, p) / n)``.
    """
    p = as_tensor(p)
    p2 = as_tensor(p2)
    n = as_tensor(n)
    return PfromH(HfromP(p, p2) / n, HfromP(p2, p) / n)


def surv(pd: torch.Tensor, po: torch.Tensor) -> torch.Tensor:
    """Survival probability ``1 - pd - po``."""
    return 1 - as_tensor(pd) - as_tensor(po)


def gamma(survive: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Transition probability from one age-class to the next.

    ``(s**(d-1) - s**d) / (1 - s**d)``.
    """
    s = as_tensor(survive)
    d = as_tensor(d)
    sd = torch.pow(s, d)
    sd_minus_1 = torch.pow(s, d - 1)
    return (sd_minus_1 - sd) / (1 - sd)


def _to_vec(values: Sequence, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(dtype=dtype, device=device)
    return torch.stack([as_tensor(v, dtype=dtype, device=device) for v in values])


def calculate_population_rates(
    state: Sequence,
    pdeath: Sequence,
    pofftake: Sequence,
    duration: Sequence,
    intake: Sequence,
    production_nutrition_index: torch.Tensor,
    maintenance_nutrition_index: torch.Tensor,
    maintenance_nutrition_mortality_factor: torch.Tensor,
    a_max: Sequence,
    prol: torch.Tensor,
    parturition: torch.Tensor,
    female_proportion: torch.Tensor,
    n: int,
    *,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: torch.device | None = None,
) -> Dict[str, torch.Tensor]:
    """PyTorch translation of ``calculatePopulationRates``.

    Returns
    -------
    dict
        Tensors keyed exactly as in the Java implementation:
        ``dRate``, ``dBirth``, ``dDeath``, ``dIntake``, ``dOfftake`` (length 6),
        and ``Balance`` (length 3: total / female / male).
    """
    if device is None:
        if isinstance(state, torch.Tensor):
            device = state.device
        else:
            for v in state:
                if isinstance(v, torch.Tensor):
                    device = v.device
                    break
    if device is None:
        device = torch.device("cpu")

    state_v = _to_vec(state, dtype=dtype, device=device)
    pdeath_v = _to_vec(pdeath, dtype=dtype, device=device)
    pofftake_v = _to_vec(pofftake, dtype=dtype, device=device)
    duration_v = _to_vec(duration, dtype=dtype, device=device)
    intake_v = _to_vec(intake, dtype=dtype, device=device)
    a_max_v = _to_vec(a_max, dtype=dtype, device=device)

    prod_idx = as_tensor(production_nutrition_index, dtype=dtype, device=device)
    maint_idx = as_tensor(maintenance_nutrition_index, dtype=dtype, device=device)
    mort_factor = as_tensor(maintenance_nutrition_mortality_factor, dtype=dtype, device=device)
    prol_t = as_tensor(prol, dtype=dtype, device=device)
    parturition_t = as_tensor(parturition, dtype=dtype, device=device)
    female_t = as_tensor(female_proportion, dtype=dtype, device=device)
    n_t = as_tensor(n, dtype=dtype, device=device)

    m = state_v.shape[0]

    d_birth = [None] * m
    d_death = [None] * m
    d_offtake = [None] * m
    d_intake = [None] * m
    d_move_from = [None] * m
    d_move_to = [None] * m
    d_surplus = [None] * m
    d_rate = [None] * m

    mortality_factor = (1 - maint_idx) * mort_factor + maint_idx

    for i in range(m):
        d_i = duration_v[i] * n_t
        n2 = torch.minimum(n_t, d_i)
        pd_i = PRescaled(pdeath_v[i], pofftake_v[i], n2)
        po_i = PRescaled(pofftake_v[i], pdeath_v[i], n2)
        surv_i = surv(pd_i, po_i)
        gamma_i = gamma(surv_i, d_i)

        d_intake[i] = intake_v[i]
        d_death[i] = pd_i * state_v[i] * mortality_factor
        d_offtake_i = po_i * state_v[i]
        d_move_from_i = (state_v[i] - d_death[i] - d_offtake_i) * gamma_i * prod_idx
        d_surplus[i] = torch.clamp(state_v[i] - a_max_v[i], min=0.0)
        d_offtake_i = d_offtake_i + d_surplus[i]

        if i == Fraction.FJ or i == Fraction.MJ:
            if i == Fraction.FJ:
                birth = (
                    prol_t * parturition_t / n_t * female_t * state_v[Fraction.FA] * surv_i
                )
            else:
                birth = (
                    prol_t * parturition_t / n_t * (1 - female_t)
                    * state_v[Fraction.FA] * surv_i
                )
            birth = birth * prod_idx
            d_birth[i] = birth
            d_move_to[i] = birth
        elif i == Fraction.FS or i == Fraction.MS:
            d_birth[i] = torch.zeros((), dtype=dtype, device=device)
            d_move_to[i] = d_move_from[i - 1]
        else:  # adult
            d_birth[i] = torch.zeros((), dtype=dtype, device=device)
            d_move_to[i] = d_move_from[i - 1]
            d_offtake_i = d_offtake_i + d_move_from_i
            d_move_from_i = torch.zeros((), dtype=dtype, device=device)

        d_offtake[i] = d_offtake_i
        d_move_from[i] = d_move_from_i
        d_rate[i] = d_intake[i] + d_move_to[i] - d_move_from[i] - d_death[i] - d_offtake[i]

    d_rate_t = torch.stack(d_rate)
    d_birth_t = torch.stack(d_birth)
    d_death_t = torch.stack(d_death)
    d_intake_t = torch.stack(d_intake)
    d_offtake_t = torch.stack(d_offtake)

    # Java: balance += dBirth - dDeath + dIntake - dOfftake - dRate for each i.
    per_i = d_birth_t - d_death_t + d_intake_t - d_offtake_t - d_rate_t
    balance = per_i.sum()
    balance_f = per_i[: Fraction.FA + 1].sum()
    balance_m = per_i[Fraction.FA + 1 :].sum()

    return {
        "dRate": d_rate_t,
        "dBirth": d_birth_t,
        "dDeath": d_death_t,
        "dIntake": d_intake_t,
        "dOfftake": d_offtake_t,
        "Balance": torch.stack([balance, balance_f, balance_m]),
    }
