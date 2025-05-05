### How to perform evaluation

```bash
NiChart_BAScores evaluate \
                 --in_dir ../Datasets/BAScores/ISTAG_CN_All/ \
                 --model resnet18 \
                 --model_type single \
                 --model_weights BAScores_brain_age_single_BL_AdamW.pth \
                 --label_dict ../Datasets/BAScores/ISTAG_CN_all/demog.csv \
                 --target Age \
                 --device cuda \
                 --verbose \
```

The above will perform evaluation on the images of the input directory as we have the "evaluate" command in the CLI. Then, it will search for the `eval` folder inside the `in_dir` and it will print the values of the metrics we use on the input images. You can select which model you want with the `model` argument and the model type with the `model_type`. The evaluate function is used in training as well - as the last step - so you have to provide the model weights to only perform evaluation.