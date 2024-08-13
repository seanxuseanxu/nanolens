#!/bin/bash
#SBATCH -N 4
#SBATCH -C gpu
#SBATCH -G 16
#SBATCH -q debug
#SBATCH -J jobParallel
#SBATCH --mail-user=seanjx@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 30:00
#SBATCH -A ntrain4
#SBATCH --reservation=NERSC_hackathon_2024_prep

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

srun -n 4 --gpus-per-task=4 --gpus-per-node=4 /global/homes/s/seanjx/gigalens/testetc/nanolens.py