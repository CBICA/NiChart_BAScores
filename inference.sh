#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=350GB
#SBATCH --output=log/inference.log
#SBATCH --error=log/inference_error.log
#SBATCH --partition=ailong
#SBATCH --time=10:00:00

NiChart_BAScores inference \
                 --in_dir ../Datasets/BAScores/pairwise/pairwise_ba_delta_training/eval \
                 --out_dir ../ \
                 --in_csv ../Datasets/BAScores/pairwise/istaging_interp.csv \
                 --csv pairwise_simple.csv \
                 --model resnet18 \
                 --mode regression \
                 --model_type pairwise \
                 --model_weights weights/BAScores_ba_delta_pairwise_ISTAGING2_new_Adadelta_resnet18.pth \
                 --device cuda \
