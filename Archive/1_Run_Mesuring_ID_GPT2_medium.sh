#!/bin/bash

# SBATCH --job-name=a
# SBATCH --time=1:10:00
# SBATCH --account=jsong
# SBATCH --partition=standard
# SBATCH --output=LOG/%x-%a.out
# SBATCH --error=LOG/%x-%a.err

# SBATCH -N 1
# SBATCH -c 20
# SBATCH --gres=gpu:volta:1

# SBATCH --mem-per-cpu=15G



echo "jobID: $SLURM_ARRAY_TASK_ID"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
module load anaconda/2023a-pytorch 

python 1_Project_Measuring_ID_GPT2_medium.py
