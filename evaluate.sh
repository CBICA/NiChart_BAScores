#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a40:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=200GB
#SBATCH --output=log/evaluate.log
#SBATCH --error=log/evaluate_error.log
#SBATCH --partition=ailong
#SBATCH --time=10:00:00

NiChart_BAScores evaluate \
                 --in_dir ../Datasets/BAScores/pairwise/cn_to_ad_training/ \
                 --model resnet18 \
                 --mode binary \
                 --model_type pairwise \
                 --meta True \
                 --model_weights weights/BAScores_cn_vs_ad_pairwisefrompretrained_ISTAGING2_new_Adadelta_resnet18.pth \
                 --label_dict ../Datasets/BAScores/pairwise/cn_to_ad.csv \
                 --device cuda \
                 --target Diagnosis_nearest_2.0 \
                 --verbose \
                 --plot_path pairwise.png
                 
