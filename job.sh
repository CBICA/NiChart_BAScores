#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-gpu=500G
#SBATCH --output=log/NiChart_BASCORES_adamw_brainagereg.log
#SBATCH --error=log/NiChart_BASCORES_adamw_brainagereg_error.log
#SBATCH --partition=ailong

NiChart_BAScores evaluate \
                 --in_dir ../Datasets/BAScores/ISTAG_CN_All/ \
                 --label_dict ../Datasets/BAScores/ISTAG_CN_All/demog.csv \
                 --device cuda \
                 --model_type single \
                 --model_weights ../BAScores_brain_age_single_BL_AdamW.pth \
                 --target Age \
                 --verbose