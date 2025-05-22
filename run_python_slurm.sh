#!/bin/bash -l
#SBATCH -t 300-00:00:00 # timelimit here 3 days
#SBATCH -p gpu        # partition (see sinfo command)
#SBATCH -J ccf     # name
#SBATCH -o stdout    # stdout filename %j will be replaced with job id
#SBATCH -e stderr    # stderr filename
#SBATCH -N 1          # number of requested cluster nodes (servers)
#SBATCH --cpus-per-task=20  # number of cpu nodes per one MPI rank

python3 ccf_calc.py > log 2>&1
