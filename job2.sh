#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=500GB
#SBATCH --output=log/cn_vs_ad_pairwisefrompretrained.log
#SBATCH --error=log/cn_vs_ad_pairwisefrompretrained_error.log
#SBATCH --partition=ailong
#SBATCH --time=15-00:00:00

NiChart_BAScores train \
                 --in_dir ../Datasets/BAScores/pairwise/cn_to_mci_training \
                 --out_dir weights\
                 --mode binary \
                 --label_dict ../Datasets/BAScores/pairwise/istaging_interp_cn_mci.csv \
                 --model_name BAScores_cn_vs_mci_pairwisefrompretrained_ISTAGING2_new_Adadelta_resnet18.pth \
                 --pretrained_pairwise weights/BAScores_ba_delta_revised_meta_ISTAGING2_new_Adadelta_resnet18.pth \
                 --device cuda \
                 --model_type pairwise \
                 --model resnet18 \
                 --meta True \
                 --target Diagnosis_nearest_2.0 \
                 --optimizer Adadelta \
                 --batch_size 32 \
                 --lr 0.5 \
                 --epochs 2000 \
                 --patience 20 \
                 --verbose \
                 
