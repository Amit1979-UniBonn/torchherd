"""PyTorch translation of ``SellOutManagement.java``.

Priority-driven sell-out manager: walks groups 1..6 in order, selling animals
to cover the cash deficit.  The Java logic is a deeply nested ``if/else``
cascade; we re-express it as a vectorised loop so the same control flow
applies even with batched tensor inputs.  Boundary decisions (sell from
this group vs. the next) are still hard cutoffs — the Java behaviour is
faithfully reproduced.

The full Java class declares 12 groups (1..12) but only uses groups 1..6 in
``process()``.  We mirror exactly the used subset.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from ._common import DEFAULT_DTYPE, make_parameter, as_tensor


_NUM_GROUPS = 6


def _stack_groups(values: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([as_tensor(v) for v in values])


class SellOutManagement(nn.Module):
    def __init__(
        self,
        *,
        cPriceLivestockGroups: Sequence[float] = (10.0,) * _NUM_GROUPS,
        cMinNumberGroups: Sequence[float] = (10.0,) * _NUM_GROUPS,
        cWeightPerAnimalGroups: Sequence[float] = (300000.0, 185000.0, 45000.0, 35000.0, 150000.0, 250000.0),
        dtype: torch.dtype = DEFAULT_DTYPE,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if len(cPriceLivestockGroups) != _NUM_GROUPS:
            raise ValueError("cPriceLivestockGroups must have exactly 6 entries")
        if len(cMinNumberGroups) != _NUM_GROUPS:
            raise ValueError("cMinNumberGroups must have exactly 6 entries")
        if len(cWeightPerAnimalGroups) != _NUM_GROUPS:
            raise ValueError("cWeightPerAnimalGroups must have exactly 6 entries")

        prices = torch.tensor(list(cPriceLivestockGroups), dtype=dtype)
        mins = torch.tensor(list(cMinNumberGroups), dtype=dtype)
        weights = torch.tensor(list(cWeightPerAnimalGroups), dtype=dtype)
        if trainable:
            self.cPriceLivestockGroups = nn.Parameter(prices.clone())
            self.cWeightPerAnimalGroups = nn.Parameter(weights.clone())
        else:
            self.register_buffer("cPriceLivestockGroups", prices.clone())
            self.register_buffer("cWeightPerAnimalGroups", weights.clone())
        # Minimum-to-keep is a hard reserve, not an optimisation target.
        self.register_buffer("cMinNumberGroups", mins.clone())

        self._dtype = dtype

    def forward(
        self,
        iNumberGroup1: torch.Tensor,
        iNumberGroup2: torch.Tensor,
        iNumberGroup3: torch.Tensor,
        iNumberGroup4: torch.Tensor,
        iNumberGroup5: torch.Tensor,
        iNumberGroup6: torch.Tensor,
        iCashAvail: torch.Tensor,
    ) -> dict:
        numbers = [iNumberGroup1, iNumberGroup2, iNumberGroup3,
                   iNumberGroup4, iNumberGroup5, iNumberGroup6]
        numbers = [as_tensor(n, dtype=self._dtype) for n in numbers]
        cash = as_tensor(iCashAvail, dtype=self._dtype)

        # Early return: cash >= 0 means no sale needed.
        if bool((cash >= 0).all()):
            zero = torch.zeros_like(cash)
            return {
                **{f"NumberSoldGroup{i+1}": zero for i in range(_NUM_GROUPS)},
                "SaleSuccess": torch.ones_like(cash, dtype=torch.bool),
                "Earnings": zero,
            }

        money_requirement = -cash  # positive money requirement
        earnings = torch.zeros_like(cash)
        sold = [torch.zeros_like(cash) for _ in range(_NUM_GROUPS)]
        # ``remaining`` mirrors the Java behaviour: once a group is fully
        # exhausted, the residual demand passes to the next group.
        # ``done`` marks the prefix of groups that have already settled the
        # full requirement and therefore must contribute 0 from then on.
        done = torch.zeros_like(cash, dtype=torch.bool)

        for i in range(_NUM_GROUPS):
            animals_avail = torch.clamp(numbers[i] - self.cMinNumberGroups[i], min=0.0)
            price_per_animal = self.cWeightPerAnimalGroups[i] * self.cPriceLivestockGroups[i]
            price_animals_avail = animals_avail * price_per_animal
            remaining = money_requirement - earnings
            animals_demand = remaining / price_per_animal

            fully_covered = animals_demand <= animals_avail

            sold_here = torch.where(
                done,
                torch.zeros_like(cash),
                torch.where(fully_covered, animals_demand, animals_avail),
            )
            sold[i] = sold_here

            earnings = torch.where(
                done,
                earnings,
                torch.where(fully_covered, money_requirement, earnings + price_animals_avail),
            )
            done = done | fully_covered

        success = earnings == money_requirement
        # Java: when sale succeeded, Earnings = tMoneyRequirement. When it
        # did not, ``tEarnings`` accumulates whatever was raised.  Our loop
        # already reflects that.
        return {
            **{f"NumberSoldGroup{i+1}": sold[i] for i in range(_NUM_GROUPS)},
            "SaleSuccess": success,
            "Earnings": earnings,
        }
