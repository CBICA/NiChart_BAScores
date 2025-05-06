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

- in_dir: The input directory that contains the train, test and eval folders with LPS oriented, DLICV preprocessed T1 images
- out_dir: The output directory that the file of the model weights with name: `model_name` will be saved
- label_dict: The path to a .csv file that must contain a column with the MRID's and a column with the values of the targets
- model_name: See above
- device: Either cuda, mps or cpu
- model: The type of backbone, currently supported: [resnet18, resnet34]
- model_type: Either single or pairwise
- target: The name of the target value(same as in the label dict)
- optimizer: We support a limited amount of tested optimizer for brain age values, you can see them at the [main](../BAScores/__main__.py)
- batch_size: The batch size that the loader will use
- lr: The learning rate
- weight_decay: The L2 regularization value
- epochs: The number of epochs that the model will be trained for
- patience: The number of epochs that the model will continue to train without the loss getting better
- verbose: Set this if you want to output information about training and evaluation of the model
