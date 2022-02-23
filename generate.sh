#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --constraint=xeon-p8
# Loading the required module
# source /etc/profile
# module load anaconda/2020a
# export PYTHONPATH=/home/gridsan/jchin/image-tsp

# Run the script
python -u src/main/generate_data.py generate 800 40 "(1024, 1024, 3)"