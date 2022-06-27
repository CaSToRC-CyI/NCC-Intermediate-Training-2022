#!/bin/sh
#SBATCH --job-name=ex1
#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=edu16
#SBATCH --reservation=NCC

source ../modules.sh

python script.py
