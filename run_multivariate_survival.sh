#!/bin/bash
#
#SBATCH --array=0-314
#SBATCH --job-name=MELBAMultivariate
#SBATCH --qos=privileged
#SBATCH --partition=reserved
#SBATCH -c 12
#SBATCH --mem 8GB
#SBATCH -o slurm_logs/mv_final_%A_%a.out
#SBATCH -e slurm_logs/mv_final_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.peoples@queensu.ca
#SBATCH --time=0-2:00:00
#SBATCH --exclude=cac[029,030]


module load StdEnv/2023
module load python/3.11.5

source .venv/bin/activate

INDEX=$SLURM_ARRAY_TASK_ID
echo $INDEX
python melba.py multivariate_survival --jobs 12 $(python melba.py multivariate_survival_args ${INDEX})
echo COMPLETED
