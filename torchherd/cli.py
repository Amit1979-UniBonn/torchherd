"""Command-line interface for :mod:`torchherd`.

Installed as the ``torchherd`` console script (see ``pyproject.toml``).  It is a
thin wrapper around :func:`torchherd.simulation.run_simulation`, kept dependency
-free (``argparse`` + ``csv`` from the standard library) so the command works
with only ``torch`` installed.

Usage
-----
::

    torchherd simulate --animal Cattle --activity Stall --days 365
    torchherd simulate --animal Sheep --days 730 --output herd.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from typing import List, Optional, Sequence

from . import __version__
from .simulation import run_simulation

_ANIMALS = ("Cattle", "Sheep")
_ACTIVITIES = ("Stall", "Pasture", "Grazing")

# Headline variables printed in the terminal summary.
_SUMMARY_FIELDS = (
    ("total_animals", "Total animals"),
    ("adult_female", "Adult females"),
    ("adult_male", "Adult males"),
    ("milk_exported", "Milk exported (g/day)"),
    ("ch4_emission", "CH4 emission (kg/day)"),
    ("manure_storage", "Manure in store"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torchherd",
        description=(
            "TorchHerd — a differentiable PyTorch framework for livestock "
            "population modelling."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"torchherd {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    sim = subparsers.add_parser(
        "simulate", help="Run a demo herd simulation under a constant daily forcing.",
    )
    sim.add_argument("--animal", default="Cattle", choices=_ANIMALS)
    sim.add_argument("--activity", default="Stall", choices=_ACTIVITIES)
    sim.add_argument(
        "--days", type=int, default=365, help="Number of daily timesteps (default: 365).",
    )
    sim.add_argument(
        "-o", "--output", metavar="PATH",
        help="Write the full daily trajectory to this CSV file.",
    )
    return parser


def _write_csv(records: List[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _run_simulate(args: argparse.Namespace) -> int:
    if args.days < 1:
        print("error: --days must be a positive integer", file=sys.stderr)
        return 2

    records = run_simulation(animal=args.animal, activity=args.activity, days=args.days)
    final = records[-1]

    print(f"TorchHerd simulation — {args.animal} / {args.activity} / {args.days} days")
    print("-" * 56)
    for key, label in _SUMMARY_FIELDS:
        print(f"  {label:<24} {final[key]:>16,.2f}")

    if args.output:
        _write_csv(records, args.output)
        print(f"\nWrote {len(records)} daily records to {args.output}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "simulate":
        return _run_simulate(args)

    parser.error(f"unknown command {args.command!r}")
    return 2  # pragma: no cover - argparse exits before reaching this line


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
