#!/bin/bash

#SBATCH --job-name "JPAC_training"
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --gpus 2

python GAN.py 