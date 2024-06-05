#!/bin/bash
#
#SBATCH --array=0-50%100
#SBATCH --job-name=MELBAMultivariate
#SBATCH --qos=privileged
#SBATCH --partition=reserved
#SBATCH -c 16
#SBATCH --mem 16GB
#SBATCH -o slurm_logs/multivariate_redo_%A_%a.out
#SBATCH -e slurm_logs/multivariate_redo_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.peoples@queensu.ca
#SBATCH --time=0-0:30:00
#SBATCH --exclude=cac[029,030]

CASES_TO_REDO=(3880 7249 13139 13168 13189 13222 13480 13481 13530 13531 13855 13911 13958 13964 14470 14524 14525 14531 15132 15240 15262 15266 15316 15317 15369 15370 15386 15388 15389 15406 15407 15408 15424 15708 15710 15766 15783 15784 15819 15820 15821 15838 15846 15847 15888 15889 15903 15910 15911 15914 16045)

module load StdEnv/2023
module load python/3.11.5

source .venv/bin/activate

INDEX=${CASES_TO_REDO[$SLURM_ARRAY_TASK_ID]}
echo $INDEX
python melba.py multivariate_survival $(python melba.py multivariate_survival_args ${INDEX})
echo COMPLETED
