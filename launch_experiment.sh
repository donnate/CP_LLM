#!/bin/bash
  
# Define the values for the variables
#n_values="300 500 1000 10000"
#n_train_values="100 500 1000 5000"
n_train_values="1000"
n_calib_values="100 500 1000"
#n_calibs_values="100 500 1000 5000"
temp_values="0.5 1.0 2.0 10.0"
delta_values="0.1 0.5 1"
epsilon_values="0.1 0.3 0.5 0.7 0.9"

for n_train in $n_train_values; do
  for n_calib in $n_calib_values; do
    for temp in $temp_values; do
       for delta in $delta_values; do
          for epsilon in $epsilon_values; do
            # Submit the job with the current values
            sbatch experiment.sh "$n_train" "$n_calib" "$temp" "$delta" "$epsilon"
           done 
       done
    done
  done
done
# $1 : N
# $2 : r
# $3 : r_pcas
# $4 : criterion (prediction/ correlation) for CV
# $5 : normalized diagonal (0/1)
# $6 : ratio p/n
