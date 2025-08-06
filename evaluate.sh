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
                 --in_dir ../Datasets/BAScores/pairwise/pairwise_ba_delta_training/ \
                 --model resnet18 \
                 --mode regression \
                 --meta True \
                 --model_type pairwise \
                 --model_weights weights/BAScores_ba_delta_pairwise_plusmeta_ISTAGING2_new_Adadelta_resnet18.pth \
                 --label_dict ../Datasets/BAScores/pairwise/istaging_interp.csv \
                 --device cuda \
                 --target Age \
                 --verbose \
                 --plot_path pairwise_meta.png
                 
