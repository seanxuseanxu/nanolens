#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q debug
#SBATCH -J testetcSRUN
#SBATCH --mail-user=seanjx@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 30:00
#SBATCH -A m3592

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

module load python/3.11
conda activate /global/homes/s/seanjx/.conda/envs/gigajax

srun -n 1 --mpi=pmix --gpus-per-task=4 --gpus-per-node=4 /global/homes/s/seanjx/gigalens/testetc/nanolens.py