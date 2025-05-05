### How to perform inference


```bash
NiChart_BAScores inference \
                 --in_dir ../Datasets/BAScores/ISTAG_CN_All/eval \
                 --device cuda \
                 --model_type single \
                 --model resnet18 \
                 --model_weights BAScores_brain_age_single_BL_AdamW.pth \
                 --csv_name y_preds.csv \
                 --batch_size 256 \
```

The above will perform inference as we have the "inference" command in the CLI, it will search for all the raw niftii images and collect them in batches(`batch_size`). The output will be a csv file with the predicted values and indexes.