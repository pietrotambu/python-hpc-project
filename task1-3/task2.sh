#!/usr/bin/env bash
#BSUB -J wallheat_ref
#BSUB -q hpc
#BSUB -W 01:00
#BSUB -n 1
#BSUB -R "select[model == XeonE5_2650v4] span[hosts=1]"
#BSUB -R "rusage[mem=4096]"
#BSUB -oo results/lsf_ref_%J.out
#BSUB -eo results/lsf_ref_%J.err

set -euo pipefail
cd "$LS_SUBCWD"

mkdir -p results figures /tmp/$USER/matplotlib
export MPLCONFIGDIR=/tmp/$USER/matplotlib
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set +u
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026
set -u

N="${N:-20}"

# Run with 'time -v' to capture precise duration in the .out file
echo "Starting simulation for N=$N..."
/usr/bin/time -v python ./task1-3/simulate.py "$N" > "results/task2_reference_stats_${N}.csv"
echo "Simulation finished."