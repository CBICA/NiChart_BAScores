#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-gpu=20
#SBATCH --mem-per-gpu=500GB
#SBATCH --output=log/inference.log
#SBATCH --error=log/inference_error.log
#SBATCH --partition=ailong
#SBATCH --time=10:00:00

NiChart_BAScores inference \
                 --in_dir ../Datasets/BAScores/ISTAG_CN_All/istaging_2_new/hypertension/hypertension_PURE \
                 --out_dir ../Datasets/BAScores/ISTAG_CN_All/istaging_2_new/attention_maps/hypertension/hypertension_PURE \
                 --csv ../test.csv \
                 --model resnet18 \
                 --mode regression \
                 --model_type single \
                 --model_weights weights/BAScores_brain_age_ISTAGING2_new_healthy_only_Adadelta_resnet18.pth \
                 --device cuda \
                 --return_attention
