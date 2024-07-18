#!/bin/bash
#SBATCH -N 1
#SBATCH -C "gpu&hbm80g"
#SBATCH -G 4
#SBATCH -q regular
#SBATCH -J parallel6
#SBATCH --mail-user=seanjx@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 24:00:00
#SBATCH -A m3592

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

module load cudatoolkit/11.7 #Version used at this moment. Compatible with JAX 0.4.7
module load cudnn/8.9.1_cuda11 #Version used at this moment. Compatible with JAX 0.4.7
module load python/3.11
conda activate /global/homes/s/seanjx/.conda/envs/gigajax

python /global/homes/s/seanjx/gigalens/parallel6/nanolens.py
