# NiChart Brain Ageing Scores

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
                 --in_dir [input directory] \
                 --out_dir [output directory] \
                 --mode [mode] \
                 --label_dict [path to .csv] \
                 --model_type [single, pairwise] \
                 --model [model name(for example: resnet18)] \
                 --target [target name(same as in label dict)] \
```
> [!Note]
> Note that the niftii files must have the following suffix corresponding that they are preprocessed as it should: `_T1_LPS_dlicv_aligned.nii.gz` otherwise, the preprocessing pipeline must be applied first

### Evaluation
Evaluation is performed by default as the last step in training, but, the user has the option to select any weight and model and evaluate it on a dataset. A simple example is the following:
```bash
NiChart_BAScores evaluate \
                 --in_dir [input directory] \
                 --mode [mode] \
                 --model [model name(for example: resnet18)] \
                 --model_type [single, pairwise] \
                 --model_weights [model_weights(.pth)] \
                 --label_dict [path to .csv] \
                 --target [target name(same as in label dict)] \
                 --plot_path [image name(.png)]
```
Note that in `in_dir`, there should exist a `eval` folder.

### Inference
Last but not least, the user has the option to perform inference using any pretrained model:
```bash
NiChart_BAScores inference \
                 --in_dir [input directory] \
                 --out_dir [output directory] \
                 --csv_name [output csv name(.csv)] \
                 --model_type [single, pairwise] \
                 --model [model name(for example: resnet18)] \
                 --model_weights [model weights name(.pth)] \
```
Note that now in `in_dir` there should be only raw niftii files and not folders!

> [!Note]
> You can see all the details about the arguments of the CLI in the [tutorial](/tutorial/) folder.
