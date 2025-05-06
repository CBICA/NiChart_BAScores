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
                 --plot_path y_pred_vs_y_hat.png
```

- in_dir: The input directory that contains the train, test and eval folders with LPS oriented, DLICV preprocessed T1 images
- model: The type of backbone, currently supported: [resnet18, resnet34]
- model_type: Either single or pairwise
- model_weights: The path to a .pth file that contains the weights of the selected model
- label_dict: The path to a .csv file that must contain a column with the MRID's and a column with the values of the targets
- target: The name of the target value(same as in the label dict)
- device: Either cuda, mps or cpu
- verbose: Set if you want to output information about training
- plot_path: A path that a predictions vs ground truth plot will be saved(with all the metrics)


