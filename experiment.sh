#!/bin/bash

#SBATCH --job-name=array
#SBATCH --output=experiments/logs/array_%A_%a.out
#SBATCH --error=experiments/logs/array_%A_%a.err
#SBATCH --array=1-10
#SBATCH --time=35:00:00
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --mem=5G
#SBATCH --account=pi-cdonnat

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "My SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
# Add lines here to run your computations
job_id=$SLURM_ARRAY_JOB_ID
#module load libgmp
module load python

result_file="new_exp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "result file is ${result_file}"
cd $SCRATCH/$USER/CP_LLM/
Rscript experiments/experiment.py $result_file $SLURM_ARRAY_TASK_ID  $1 $2 $3 $4 $5
# $1 : n_train
# $2 : n_calib
# $3 : temp
# $4 : delta
# $5 : epsilon

