#!/bin/bash
#SBATCH --gres=gpu:volta:1
# Loading the required module
# source /etc/profile
# module load anaconda/2020a
 

# Run the script
# The last number is an argument to set the number of cities.
# [1, 20, 30, 100, 200]
python -u src/nn/tspconv.py 1
