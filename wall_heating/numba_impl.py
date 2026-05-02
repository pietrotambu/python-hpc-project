"""Numba-based solvers (exercise 7 CPU JIT + exercise 8 custom CUDA kernel)."""

from __future__ import annotations

import numpy as np

try:
    from numba import cuda, njit
except ImportError as exc:  # pragma: no cover
    cuda = None
    njit = None
    _NUMBA_IMPORT_ERROR = exc
else:
    _NUMBA_IMPORT_ERROR = None


def _require_numba() -> None:
    if _NUMBA_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Numba is not installed. Use the course GPU/Numba environment."
        ) from _NUMBA_IMPORT_ERROR


# ---------------------------------------------------------------------
# Exercise 7: CPU JIT solution
# ---------------------------------------------------------------------
if njit is not None:

    @njit(cache=True)
    def _jacobi_numba_cpu_kernel(
        u0: np.ndarray,
        interior_mask: np.ndarray,
        max_iter: int,
        atol: float,
    ) -> np.ndarray:
        """
        CPU JIT Jacobi solver.

        Outer loop over rows and inner loop over columns gives row-major
        access, which matches NumPy's memory layout well.
        """
        current = u0.copy()
        nxt = u0.copy()
        rows, cols = current.shape

        for _ in range(max_iter):
            delta = 0.0

            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    if interior_mask[i - 1, j - 1]:
                        new_val = 0.25 * (
                            current[i, j - 1]
                            + current[i, j + 1]
                            + current[i - 1, j]
                            + current[i + 1, j]
                        )
                        diff = abs(new_val - current[i, j])
                        if diff > delta:
                            delta = diff
                        nxt[i, j] = new_val
                    else:
                        nxt[i, j] = current[i, j]

            current, nxt = nxt, current

            if delta < atol:
                break

        return current


def jacobi_numba_cpu(
    u: np.ndarray,
    interior_mask: np.ndarray,
    max_iter: int,
    atol: float = 1e-6,
) -> np.ndarray:
    """Exercise 7: drop-in CPU JIT replacement for the reference Jacobi solver."""
    _require_numba()
    return _jacobi_numba_cpu_kernel(u, interior_mask, max_iter, atol)


# ---------------------------------------------------------------------
# Exercise 8: custom CUDA kernel solution
# ---------------------------------------------------------------------
if cuda is not None:

    @cuda.jit
    def _jacobi_cuda_step(
        current: np.ndarray,
        nxt: np.ndarray,
        interior_mask: np.ndarray,
    ) -> None:
        """
        Perform exactly one Jacobi iteration on the GPU.

        Use j, i = cuda.grid(2) so the x dimension maps to columns.
        That matches row-major array layout better on GPU.
        """
        j, i = cuda.grid(2)
        rows, cols = current.shape

        if 1 <= i < rows - 1 and 1 <= j < cols - 1:
            if interior_mask[i - 1, j - 1]:
                nxt[i, j] = 0.25 * (
                    current[i, j - 1]
                    + current[i, j + 1]
                    + current[i - 1, j]
                    + current[i + 1, j]
                )
            # For non-interior points we do nothing:
            # both current and nxt were initialized from the same input,
            # so walls and outside-building points remain fixed.


def jacobi_numba_cuda(
    u: np.ndarray,
    interior_mask: np.ndarray,
    max_iter: int,
    atol: float = 1e-4,
) -> np.ndarray:
    """
    Exercise 8: helper function that repeatedly launches a single-iteration
    CUDA kernel. No atol / early stopping, per assignment instructions.
    """
    _require_numba()
    if cuda is None:
        raise RuntimeError("Numba CUDA support is unavailable.")

    current_d = cuda.to_device(np.asarray(u))
    nxt_d = cuda.to_device(np.asarray(u))
    mask_d = cuda.to_device(np.asarray(interior_mask, dtype=np.bool_))

    # x dimension = columns, y dimension = rows
    threads_per_block = (32, 16)
    blocks_per_grid = (
        (u.shape[1] + threads_per_block[0] - 1) // threads_per_block[0],
        (u.shape[0] + threads_per_block[1] - 1) // threads_per_block[1],
    )

    for _ in range(max_iter):
        _jacobi_cuda_step[blocks_per_grid, threads_per_block](
            current_d, nxt_d, mask_d
        )
        current_d, nxt_d = nxt_d, current_d

    return current_d.copy_to_host()