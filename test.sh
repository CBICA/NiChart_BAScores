#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=150GB
#SBATCH --output=log/NiChart_BASCORES_adamw_brainagereg.log
#SBATCH --error=log/NiChart_BASCORES_adamw_brainagereg_error.log
#SBATCH --partition=ailong
#SBATCH --time=21-00:00:00

python3 test_attention.py