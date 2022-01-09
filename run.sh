#!/bin/bash
#SBATCH --gres=gpu:volta:1
# Loading the required module
# source /etc/profile
# module load anaconda/2020a
 

# Run the script
python -u src/nn/tspconv.py 200
