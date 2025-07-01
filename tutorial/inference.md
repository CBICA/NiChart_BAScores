### How to perform inference


```bash
NiChart_BAScores inference \
                 --in_dir [input directory]
                 --out_dir [output directory] \
                 --csv [full path of the .csv output file] \
                 --device [cuda, mps, cpu] \
                 --model_type [single, pairwise] \
                 --model [model name(for example: resnet18)] \
                 --mode [regression, binary, multiclass]
                 --model_weights [model weights name(.pth)] \
                 --return_attention
```

- in_dir: The input directory that contains the train, test and eval folders with LPS oriented, DLICV preprocessed T1 images
- out_dir: The output directory that the csv file that contains the predicted results with name: `csv_name` will be saved
- csv: The full path of the .csv output file
- device: Either cuda, mps or cpu
- model: The type of backbone, currently supported: [resnet18, resnet34]
- model_type: Either single or pairwise
- mode: Either regression, binary or multiclass
- model_weights: The path to a .pth file that contains the weights of the selected model
- return_attention: Set if you want to save the attention maps of the passed
  images in the specified output directory
