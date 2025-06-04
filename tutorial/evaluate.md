### How to perform evaluation

```bash
NiChart_BAScores evaluate \
                 --in_dir [input directory] \
                 --model [model name(for example: resnet18)] \
                 --mode [binary, multiclass, regression] \
                 --model_type [single, pairwise] \
                 --model_weights [model weights name(.pth)] \
                 --label_dict [path to .csv] \
                 --target [target name(same as label dict)] \
                 --device [cuda, mps, cpu] \
                 --verbose \
                 --plot_path [image name(.png)]
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


