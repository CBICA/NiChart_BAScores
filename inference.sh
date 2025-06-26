#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=2000
#SBATCH --output=log/inference.log
#SBATCH --error=log/inference_error.log
#SBATCH --partition=aishort
#SBATCH --time=1:00:00

NiChart_BAScores evaluate \
                 --in_dir ../Datasets/BAScores/ISTAG_CN_All/BL_only_training \
                 --model resnet18-cbam \
                 --mode regression \
                 --model_type single \
                 --model_weights weights/BAScores_brain_age_SPARE-BA-Harmonized_AdamW_highl2_resnet18-cbam.pth \
                 --label_dict ../Datasets/BAScores/SPARE_Lists/SPARE-BA-Harmonized.csv \
                 --target Age \
                 --device cuda \
                 --verbose \
                 --plot_path cbam_in_weights.png
                 