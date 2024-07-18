#!/bin/bash
#SBATCH -N 1
#SBATCH -C "gpu&hbm80g"
#SBATCH -G 4
#SBATCH -q debug
#SBATCH -J testetc
#SBATCH -t 30:00
#SBATCH -A m3592
#SBATCH --image=seanjx/gigadock:1.0

python /global/homes/s/seanjx/gigalens/testetc/testparallel.py