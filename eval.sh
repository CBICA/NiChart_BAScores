#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a40:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=100G
#SBATCH --output=log/evaluate.log
#SBATCH --error=log/eval_error.log
#SBATCH --partition=aishort

NiChart_BAScores evaluate \
                 --in_dir ../Datasets/ADNI/ad_regression \
                 --model resnet18 \
                 --model_type single \
                 --model_weights weights/BAScores_brain_age_SPARE-BA-Harmonized_AdamW_highl2.pth \
                 --label_dict ../Datasets/ADNI/out_list_ALL.csv \
                 --target Age \
                 --device cuda \
                 --verbose \
                 --plot_path y_pred_vs_y_hat_AD.png