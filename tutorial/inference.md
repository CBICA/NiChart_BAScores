### How to perform inference


```bash
NiChart_BAScores inference \
                 --in_dir ../Datasets/BAScores/ISTAG_CN_All/eval \
                 --out_dir . \
                 --csv_name y_pred.csv \
                 --device cuda \
                 --model_type single \
                 --model resnet18 \
                 --model_weights BAScores_brain_age_single_BL_AdamW.pth \
                 --batch_size 256 \
```

- in_dir: The input directory that contains the train, test and eval folders with LPS oriented, DLICV preprocessed T1 images
- out_dir: The output directory that the csv file that contains the predicted results with name: `csv_name` will be saved
- csv_name: See above
- device: Either cuda, mps or cpu
- model: The type of backbone, currently supported: [resnet18, resnet34]
- model_type: Either single or pairwise
- model_weights: The path to a .pth file that contains the weights of the selected model
- batch_size: The batch size that the loader will use. This will make things faster but will increase the GPU overhead
