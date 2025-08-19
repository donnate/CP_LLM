#!/bin/bash

#SBATCH --job-name=array
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --array=1-5
#SBATCH --time=6:00:00
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
#module load python

#source ~/miniconda3/etc/profile.d/conda.sh
module load python/anaconda-2022.05
conda activate cp_env
#module unload load python/anaconda-2022.05

result_file="new_exp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "result file is ${result_file}"
cd $SCRATCH/$USER/CP_LLM/
python experiment.py --exp_name $result_file --seed $SLURM_ARRAY_TASK_ID  --n_train $1 --n_calib $2 --temp $3 --delta $4 --epsilon $5
# $1 : n_train
# $2 : n_calib
# $3 : temp
# $4 : delta
# $5 : epsilon

