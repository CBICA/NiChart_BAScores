#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a40:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=100GB
#SBATCH --output=log/inference.log
#SBATCH --error=log/inference_error.log
#SBATCH --partition=ailong
#SBATCH --time=10:00:00

NiChart_BAScores evaluate \
                 --in_dir ../Datasets/BAScores/ISTAG_DISEASED_All/CN_AD_attention_maps \
                 --model resnet18 \
                 --mode binary \
                 --model_type single \
                 --model_weights weights/BAScores_cn_vs_ad_ISTAGING2_Adadelta_resnet18.pth \
                 --label_dict ../Datasets/BAScores/ISTAG_DISEASED_All/ISTAGING_2_0_ONEHOT.csv \
                 --target DX_AD \
                 --device cuda \
                 --verbose \
                 --plot_path cn_vs_ad_resnet18.png \