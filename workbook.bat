#!/bin/bash
#SBATCH --job-name=Workbook
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

jupyter notebook --no-browser --port 8890