SHELL := /bin/bash

CONDA_INIT ?= /dtu/projects/02613_2025/conda/conda_init.sh
CONDA_ENV ?= 02613_2026

N ?= 20
MAX_ITER ?= 20000
ATOL ?= 1e-4
WORKERS ?= 12
CHUNKSIZE ?= 1
SOLVER ?= reference
SCHEDULE ?= dynamic
CSV ?= results/solver_stats.csv

.PHONY: help venv clean zip run-ref run-solver inspect visualize analyze submit-ref submit-static submit-dynamic submit-numba-cpu submit-numba-cuda submit-cupy submit-cupy-nsys submit-full

define RUN_IN_CONDA
bash -lc 'source "$(CONDA_INIT)" && conda activate "$(CONDA_ENV)" && $(1)'
endef

help:
	@echo "Targets:"
	@echo "  make venv                      # verify shared DTU conda env ($(CONDA_ENV))"
	@echo "  make run-ref N=20              # local reference run"
	@echo "  make run-solver N=40 SOLVER=reference WORKERS=8 SCHEDULE=dynamic"
	@echo "  make inspect N=4               # save input visualizations"
	@echo "  make visualize N=4             # save solved visualizations"
	@echo "  make analyze CSV=results/all_buildings_stats.csv"
	@echo "  make submit-ref N=20           # submit LSF CPU job"
	@echo "  make submit-static N=80        # submit LSF static speedup"
	@echo "  make submit-dynamic N=80       # submit LSF dynamic speedup"
	@echo "  make submit-numba-cpu N=40     # submit LSF Numba CPU"
	@echo "  make submit-numba-cuda N=40 MAX_ITER=2000"
	@echo "  make submit-cupy N=40"
	@echo "  make submit-cupy-nsys N=20"
	@echo "  make submit-full               # submit full dataset run"

venv:
	@$(call RUN_IN_CONDA,python -c "import sys; print(sys.executable)")

run-ref: venv
	@mkdir -p results
	@$(call RUN_IN_CONDA,python scripts/simulate_reference.py $(N) --max-iter $(MAX_ITER) --atol $(ATOL) --output-csv results/reference_stats_$(N).csv)

run-solver: venv
	@mkdir -p results
	@$(call RUN_IN_CONDA,python scripts/run_solver.py $(N) --solver $(SOLVER) --workers $(WORKERS) --schedule $(SCHEDULE) --chunksize $(CHUNKSIZE) --max-iter $(MAX_ITER) --atol $(ATOL) --output-csv $(CSV))

inspect: venv
	@mkdir -p figures
	@$(call RUN_IN_CONDA,python scripts/inspect_data.py --n $(N) --output-dir figures)

visualize: venv
	@mkdir -p figures
	@$(call RUN_IN_CONDA,python scripts/visualize_results.py --n $(N) --max-iter $(MAX_ITER) --atol $(ATOL) --output-dir figures)

analyze: venv
	@mkdir -p results figures
	@$(call RUN_IN_CONDA,python scripts/analyze_results.py $(CSV) --summary-txt results/final_summary.txt --output-dir figures)

submit-ref: venv
	@mkdir -p results figures
	N=$(N) bsub -env "all" < jobs/lsf_reference_hpc.lsf

submit-static: venv
	@mkdir -p results figures
	N=$(N) bsub -env "all" < jobs/lsf_speedup_static_hpc.lsf

submit-dynamic: venv
	@mkdir -p results figures
	N=$(N) bsub -env "all" < jobs/lsf_speedup_dynamic_hpc.lsf

submit-numba-cpu: venv
	@mkdir -p results figures
	N=$(N) bsub -env "all" < jobs/lsf_numba_cpu_hpc.lsf

submit-numba-cuda: venv
	@mkdir -p results figures
	N=$(N) MAX_ITER=$(MAX_ITER) bsub -env "all" < jobs/lsf_numba_cuda.lsf

submit-cupy: venv
	@mkdir -p results figures
	N=$(N) bsub -env "all" < jobs/lsf_cupy_gpu.lsf

submit-cupy-nsys: venv
	@mkdir -p results figures
	N=$(N) bsub -env "all" < jobs/lsf_cupy_nsys.lsf

submit-full: venv
	@mkdir -p results figures
	bsub -env "all" < jobs/lsf_full_run_hpc.lsf

zip:
	@zip -r code.zip figures jobs results scripts wall_heating .gitignore Makefile README.md

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache
