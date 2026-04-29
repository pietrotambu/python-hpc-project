"""Exercise 9: CuPy version of the reference Jacobi solver."""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ImportError as exc:
    cp = None
    _CUPY_IMPORT_ERROR = exc
else:
    _CUPY_IMPORT_ERROR = None


def _require_cupy() -> None:
    if _CUPY_IMPORT_ERROR is not None:
        raise RuntimeError(
            "CuPy is not installed. Install a CUDA-matched wheel."
        ) from _CUPY_IMPORT_ERROR


def jacobi_cupy(
    u: np.ndarray,
    interior_mask: np.ndarray,
    max_iter: int,
    atol: float = 1e-6,
) -> np.ndarray:
    """
    Direct CuPy adaptation of the reference Jacobi implementation.

    This stays intentionally close to the NumPy reference for exercise 9.
    """
    _require_cupy()

    u = cp.array(u, copy=True)
    interior_mask = cp.asarray(interior_mask, dtype=cp.bool_)

    u_inner = u[1:-1, 1:-1]

    for _ in range(max_iter):
        # Same update as in the reference NumPy version
        u_new = 0.25 * (
            u[1:-1, :-2]
            + u[1:-1, 2:]
            + u[:-2, 1:-1]
            + u[2:, 1:-1]
        )

        u_new_interior = u_new[interior_mask]
        delta = cp.abs(u_inner[interior_mask] - u_new_interior).max()
        u_inner[interior_mask] = u_new_interior

        if atol > 0 and float(delta) < atol:
            break

    return cp.asnumpy(u)