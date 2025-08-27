#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem-per-gpu=500GB
#SBATCH --output=log/inference.log
#SBATCH --error=log/inference_error.log
#SBATCH --partition=ailong
#SBATCH --time=10:00:00

NiChart_BAScores inference \
                 --in_dir ../Datasets/updated_100k_nichart/dlicv_aligned \
                 --out_dir ../ \
                 --in_csv ../Datasets/updated_100k_nichart/list_fname_new.csv \
                 --csv nichart_100k_updated.csv \
                 --model resnet18 \
                 --mode regression \
                 --model_type single \
                 --model_weights weights/BAScores_ba_delta_revised_meta_ISTAGING2_new_Adadelta_resnet18.pth \
                 --device cuda \
