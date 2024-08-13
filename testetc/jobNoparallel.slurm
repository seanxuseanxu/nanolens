#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q debug
#SBATCH -J jobNoparallel
#SBATCH --mail-user=seanjx@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 30:00
#SBATCH -A m3592

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

module load python/3.11
module load cudatoolkit/12.2
module load cudnn/8.9.3_cuda12

export PMIX_MCA_psec=^munge
export MPI4JAX_USE_CUDA_MPI=1

conda activate /global/homes/s/seanjx/.conda/envs/gigajax2.0

srun -n 1 --mpi=pmix --gpus-per-task=1 --gpus-per-node=4 /global/homes/s/seanjx/gigalens/testetc/nanolensNOparallelization.py