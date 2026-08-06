"""Integration tests for the :mod:`torchherd` package."""

import math

import torch
import pytest

import torchherd
from torchherd import LivestockModel, run_simulation


STEP_INPUTS = dict(
    iMilkAmount=torch.tensor(1000.0, dtype=torch.float64),
    iForageProduction=torch.tensor(2.0e10, dtype=torch.float64),
    iConcentrateProduction=torch.tensor(5.0e9, dtype=torch.float64),
    iFodderProduction=torch.tensor(5.0e6, dtype=torch.float64),
    iFertilizeFraction=torch.tensor(0.1, dtype=torch.float64),
    iDroppingFraction=torch.tensor(0.2, dtype=torch.float64),
)


def test_package_exposes_version_and_public_api():
    assert isinstance(torchherd.__version__, str)
    for name in ("LivestockModel", "PopulationGrowth", "run_simulation"):
        assert name in torchherd.__all__
        assert hasattr(torchherd, name)


def test_single_step_returns_expected_sections():
    model = LivestockModel(animal="Cattle", activity="Stall", trainable=False)
    model.reset_state()
    out = model.step(**STEP_INPUTS)
    assert set(out) >= {
        "demand", "stress", "supply", "population",
        "weight", "milk", "fodder", "manure", "methane",
    }
    pop = out["population"]
    # The six age/sex classes sum to the reported total.
    total = sum(
        float(pop[k])
        for k in (
            "sJuvenileFemale", "sSubAdultFemale", "sAdultFemale",
            "sJuvenileMale", "sSubAdultMale", "sAdultMale",
        )
    )
    assert math.isclose(total, float(pop["sTotalAnimals"]), rel_tol=1e-9)


@pytest.mark.parametrize("animal,activity", [
    ("Cattle", "Stall"), ("Cattle", "Pasture"), ("Sheep", "Grazing"),
])
def test_multiday_simulation_is_finite(animal, activity):
    records = run_simulation(animal=animal, activity=activity, days=120)
    assert len(records) == 120
    for value in records[-1].values():
        assert math.isfinite(value)
    # Populations stay non-negative over the run.
    assert all(r["total_animals"] >= 0 for r in records)


def test_run_simulation_forcing_override_changes_trajectory():
    baseline = run_simulation(days=20)
    starved = run_simulation(days=20, forcing={"iForageProduction": 0.0,
                                               "iConcentrateProduction": 0.0})
    # With no feed, the herd should end up smaller than the well-fed baseline.
    assert starved[-1]["total_animals"] < baseline[-1]["total_animals"]


def test_model_is_differentiable():
    """A gradient must flow from a herd output back to a model parameter."""
    model = LivestockModel(animal="Cattle", activity="Stall", trainable=True)
    model.reset_state()
    for _ in range(5):
        out = model.step(**STEP_INPUTS)
    loss = out["population"]["sTotalAnimals"]
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "expected at least one parameter to receive a gradient"
    assert any(torch.count_nonzero(g) > 0 for g in grads)


def test_lite_methane_variant_runs():
    model = LivestockModel(use_lite_methane=True, trainable=False)
    model.reset_state()
    out = model.step(**STEP_INPUTS)
    assert "CH4emission" in out["methane"]
    assert float(out["methane"]["CH4emission"]) >= 0.0


def test_sell_out_triggers_on_cash_deficit():
    model = LivestockModel(trainable=False)
    model.reset_state()
    out = model.step(iCashAvail=torch.tensor(-1.0e9, dtype=torch.float64), **STEP_INPUTS)
    sell = out["sell_out"]
    assert sell is not None
    sold = sum(float(sell[f"NumberSoldGroup{i+1}"]) for i in range(6))
    assert sold > 0
