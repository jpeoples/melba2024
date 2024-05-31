#!/bin/bash
#
#SBATCH --array=0-2999%100
#SBATCH --job-name=MELBAMultivariate
#SBATCH --qos=privileged
#SBATCH --partition=reserved
#SBATCH -c 16
#SBATCH --mem 16GB
#SBATCH -o slurm_logs/multivariate_%a.out
#SBATCH -e slurm_logs/multivariate_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.peoples@queensu.ca
#SBATCH --time=0-0:30:00
#SBATCH --exclude=cac[029,030]


module load StdEnv/2023
module load python/3.11.5

source .venv/bin/activate

python melba.py multivariate_survival $(python melba.py multivariate_survival_args ${SLURM_ARRAY_TASK_ID})
