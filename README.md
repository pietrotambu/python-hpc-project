# python-hpc-project

Wall Heating mini-project for 02613.

Dataset: `/dtu/projects/02613_2025/data/modified_swiss_dwellings`

## Structure

- `wall_heating/` — library code (data loading, solvers, metrics, parallel scheduling)
- `scripts/` — runnable scripts for each task
- `jobs/` — LSF job scripts for the HPC cluster
- `results/` and `figures/` — default output folders

## Quick start

Run from the repo root.

```bash
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026
make run-ref N=20
make inspect N=4
make visualize N=4 MAX_ITER=2000
```

## Run a solver

```bash
python scripts/run_solver.py 40 --solver reference --workers 1
python scripts/run_solver.py 40 --solver reference --workers 8 --schedule static
python scripts/run_solver.py 40 --solver reference --workers 8 --schedule dynamic
python scripts/run_solver.py 40 --solver numba-cpu --workers 1
python scripts/run_solver.py 40 --solver numba-cuda --workers 1 --max-iter 20000
python scripts/run_solver.py 40 --solver cupy --workers 1
```

Solver options:
- `reference` — baseline from the assignment
- `numpy` — optimized NumPy with ping-pong buffers
- `numba-cpu` — CPU JIT (requires `numba`)
- `numba-cuda` — custom CUDA kernel, fixed iterations (requires `numba` + CUDA)
- `cupy` — GPU vectorized (requires `cupy`)

## Task scripts

- Profile Jacobi: `kernprof -l -v scripts/profile_jacobi.py`
- Speedup + Amdahl: `python scripts/benchmark_speedup.py --schedule static` / `--schedule dynamic`
- CuPy nsys profile: `make submit-cupy-nsys N=20`
- Final analysis: `python scripts/analyze_results.py results/all_buildings_stats.csv`

## LSF jobs

Queues used: `hpc` (CPU) and `c02613` (GPU).

```bash
make submit-ref N=20
make submit-static N=80
make submit-dynamic N=80
make submit-numba-cpu N=40
make submit-numba-cuda N=40 MAX_ITER=20000
make submit-cupy N=40
make submit-cupy-nsys N=20
make submit-full
```

Job files in `jobs/`:
- `lsf_reference_hpc.lsf`
- `lsf_speedup_static_hpc.lsf`
- `lsf_speedup_dynamic_hpc.lsf`
- `lsf_numba_cpu_hpc.lsf`
- `lsf_numba_cuda.lsf`
- `lsf_cupy_gpu.lsf`
- `lsf_cupy_nsys.lsf`
- `lsf_full_run_hpc.lsf`

## Notes

- The course conda environment `02613_2026` is shared and pre-provisioned; this repo does not install packages into it.
- GPU tasks require CUDA-enabled packages already present in `02613_2026`.
