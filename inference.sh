#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a40:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=350GB
#SBATCH --output=log/inference.log
#SBATCH --error=log/inference_error.log
#SBATCH --partition=ailong
#SBATCH --time=10:00:00

NiChart_BAScores inference \
                 --in_dir ../Datasets/BAScores/pairwise/cn_to_ad_training/train \
                 --out_dir ../ \
                 --in_csv ../Datasets/BAScores/pairwise/istaging_interp.csv \
                 --csv cn_vs_ad_pairwise.csv \
                 --model resnet18 \
                 --meta True \
                 --mode regression \
                 --model_type pairwise \
                 --model_weights weights/BAScores_ba_delta_revised_meta_ISTAGING2_new_Adadelta_resnet18.pth \
                 --device cuda \
                 --plot_path cn_vs_ad_tsne.png
