# NiChart Brain Aging Scores

## Overview
NiChart_BAScores is an integrated NiChart package that helps users train/evaluate models or perform inference using our pre-trained models.

The user can either use the tool directly in NiChart or use the CLI by first installing it:
> [!Note]
> python 3.9 and higher required!
```bash
pip install -e .
```

### Training
To train a model, you first need a directory that contains `train`, `test` and `eval` folders with niftii LPS oriented and DLICV preprocessed images(see [DLICV](https://github.com/CBICA/DLICV)) and a `csv` file that maps the `MRID`'s to the target value. Then, a simple example of training a `resnet18` on your data is by simply doing the following:
```bash
NiChart_BAScores train \
                 --in_dir [in_dir] \
                 --out_dir [out_dir] \
                 --label_dict [label_dict_path] \
                 --model_type single \
                 --model resnet18 \
                 --target Age \
```

### Evaluation
Evaluation is performed by default as the last step in training, but, the user has the option to select any weight and model and evaluate it on a dataset. A simple example is the following:
```bash
NiChart_BAScores evaluate \
                 --in_dir [in_dir] \
                 --model resnet18 \
                 --model_type single \
                 --model_weights [model_weights(.pth)] \
                 --label_dict [label_dict_path] \
                 --target Age \
                 --plot_path y_pred_vs_y_hat.png
```
Note that in `in_dir`, there should exist a `eval` folder.

### Inference
Last but not least, the user has the option to perform inference using any pretrained model:
```bash
NiChart_BAScores inference \
                 --in_dir [in_dir] \
                 --out_dir [out_dir] \
                 --csv_name [csv_name] \
                 --model_type single \
                 --model resnet18 \
                 --model_weights [model_weights(.pth)] \
```
Note that now in `in_dir` there should be only raw niftii files and not folders!

> [!Note]
> You can see all the details about the arguments of the CLI in the [tutorial](/tutorial/) folder.
