"""Unit tests for the pure-function port in :mod:`torchherd.functions`."""

import torch
import pytest

from torchherd.functions import (
    Fraction,
    HfromP,
    PfromH,
    PRescaled,
    surv,
    gamma,
    calculate_population_rates,
)


def t(x):
    return torch.tensor(float(x), dtype=torch.float64)


def test_hfromp_zero_probability_is_zero_hazard():
    assert float(HfromP(t(0.0), t(0.1))) == 0.0


def test_pfromh_zero_hazard_is_zero_probability():
    assert float(PfromH(t(0.0), t(0.1))) == 0.0


def test_hfromp_pfromh_are_consistent():
    """PfromH(HfromP(p, p2), HfromP(p2, p)) recovers p for a valid pair."""
    p, p2 = t(0.1), t(0.05)
    h = HfromP(p, p2)
    h2 = HfromP(p2, p)
    recovered = PfromH(h, h2)
    assert float(recovered) == pytest.approx(float(p), rel=1e-9)


def test_prescaled_by_one_is_identity():
    p, p2 = t(0.1), t(0.05)
    assert float(PRescaled(p, p2, t(1.0))) == pytest.approx(float(p), rel=1e-9)


def test_surv_and_gamma_ranges():
    s = surv(t(0.1), t(0.05))
    assert float(s) == pytest.approx(0.85, rel=1e-12)
    g = gamma(s, t(3.0))
    assert 0.0 < float(g) < 1.0


def test_calculate_population_rates_shapes_and_balance():
    state = [t(v) for v in (100, 80, 200, 100, 80, 150)]
    zeros = [t(0.0)] * 6
    duration = [t(v) for v in (1, 3, 11, 1, 3, 6)]
    a_max = [t(1e6)] * 6
    out = calculate_population_rates(
        state=state,
        pdeath=[t(0.13), t(0.05), t(0.03), t(0.13), t(0.05), t(0.03)],
        pofftake=zeros,
        duration=duration,
        intake=zeros,
        production_nutrition_index=t(1.0),
        maintenance_nutrition_index=t(1.0),
        maintenance_nutrition_mortality_factor=t(2.0),
        a_max=a_max,
        prol=t(1.0),
        parturition=t(0.5),
        female_proportion=t(0.5),
        n=365,
    )
    for key in ("dRate", "dBirth", "dDeath", "dIntake", "dOfftake"):
        assert out[key].shape == (6,)
    assert out["Balance"].shape == (3,)
    # Total balance is the sum of the female and male balances.
    assert float(out["Balance"][0]) == pytest.approx(
        float(out["Balance"][1]) + float(out["Balance"][2]), abs=1e-6
    )


def test_fraction_enum_order():
    assert [f.value for f in Fraction] == [0, 1, 2, 3, 4, 5]
