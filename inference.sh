#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a40:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=100GB
#SBATCH --output=log/inference.log
#SBATCH --error=log/inference_error.log
#SBATCH --partition=ailong
#SBATCH --time=10:00:00

NiChart_BAScores inference \
                 --in_dir ../Datasets/ADNI/ad_regression/eval_AD \
                 --out_dir ../test \
                 --csv ../ad_preds.csv \
                 --device cuda \
                 --model_type single \
                 --model resnet18 \
                 --mode regression \
                 --model_weights weights/BAScores_brain_age_SPARE-BA-Harmonized_AdamW_highl2_resnet18.pth \
                 --return_attention