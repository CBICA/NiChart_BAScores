### How to perform training

```bash
NiChart_BAScores train \
                 --in_dir ../Datasets/BAScores/ISTAG_CN_All/ \
                 --out_dir ../ \
                 --label_dict ../Datasets/BAScores/ISTAG_CN_All/demog.csv \
                 --model_name BAScores_brain_age_single_BL_AdamW.pth \
                 --device cuda \
                 --model_type single \
                 --model resnet18 \
                 --target Age \
                 --optimizer AdamW \
                 --batch_size 64 \
                 --lr 0.001 \
                 --weight_decay 0.01 \
                 --epochs 100 \
                 --patience 10 \
                 --verbose \
```

The following will perform training as we have the "train" command in the CLI. Then, it will search
for the `train`, `test` and `eval` folders inside the `in_dir` and it will output the weights of the trained
model as `model_name` in the `out_dir`. The `label_dict` is the path of the csv file that contains the `MRID` and `target` columns of the images. Note that (at least for now) your image names should be like this: `1110526_2_0_T1_LPS_dlicv_aligned.nii.gz` as our training process only supports images that are LPS oriented and preprocessed with DLICV. Note that the `MRID` of the image above is `1110526_2_0`.
We support two types of models, the `single` subject models and `pairwise` models. You can select either one of those using the `model_type`. Note that we support limited encoders(like resnet18 and resnet34) and we hope to support and test more in the future, you can select the encoder you want using `model`. The `optimizer`, `batch_size`, `lr`, `weight_decay`, `epochs` and `patience` are typical hyperparameters for training. `verbose` will output the results directly in the terminal, you can make the output go to a log file of your preference by simply doing `NiChart_BAScores train ... > output.log`