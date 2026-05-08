#!/bin/bash
#SBATCH -J hw
#SBATCH -o hw.o%j
#SBATCH -t 20:00:00
#SBATCH -N 1 -n 4
#SBATCH --gres gpu:1


source activate census-cluster

CUDA_VISIBLE_DEVICES=7 python scripts/train_census_foundation_models.py --config configs/census_foundation_models.json
