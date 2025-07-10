#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a40:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=200GB
#SBATCH --output=log/inference.log
#SBATCH --error=log/inference_error.log
#SBATCH --partition=ailong
#SBATCH --time=10:00:00

NiChart_BAScores evaluate \
                 --in_dir ../Datasets/BAScores/ISTAG_CN_All/istaging_2_new/CN_split \
                 --model resnet18 \
                 --mode regression \
                 --model_type single \
                 --model_weights weights/BAScores_brain_age_ISTAGING2_new_healthy_only_AdamW_resnet18.pth \
                 --label_dict ../Datasets/BAScores/ISTAG_CN_All/istaging_2_new/istag_controlNoOverlap.csv \
                 --device cuda \
                 --target Age \
                 --verbose \
                 --plot_path healthy_only_adamw.png
                 