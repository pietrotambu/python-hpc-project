#!/usr/bin/env python
"""Usage: kernprof -l -v scripts/profile_jacobi.py"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wall_heating import DEFAULT_ABS_TOL, DEFAULT_LOAD_DIR, DEFAULT_MAX_ITER, jacobi_reference, load_data


try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover
    def profile(func):  # type: ignore[misc]
        return func

jacobi_reference = profile(jacobi_reference)  # type: ignore[name-defined]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--load-dir", default=str(DEFAULT_LOAD_DIR))
    p.add_argument("--building-id", default="10000")
    p.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    p.add_argument("--atol", type=float, default=DEFAULT_ABS_TOL)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    u0, mask = load_data(args.load_dir, args.building_id)
    _ = jacobi_reference(u0, mask, args.max_iter, args.atol)


if __name__ == "__main__":
    main()
