### How to perform inference


```bash
NiChart_BAScores inference \
                 --in_dir [input directory]
                 --out_dir [output directory] \
                 --csv_name [output csv name(.csv)] \
                 --device [cuda, mps, cpu] \
                 --model_type [single, pairwise] \
                 --model [model name(for example: resnet18)] \
                 --model_weights [model weights name(.pth)] \
                 --batch_size [batch_size(int)] \
```

- in_dir: The input directory that contains the train, test and eval folders with LPS oriented, DLICV preprocessed T1 images
- out_dir: The output directory that the csv file that contains the predicted results with name: `csv_name` will be saved
- csv_name: See above
- device: Either cuda, mps or cpu
- model: The type of backbone, currently supported: [resnet18, resnet34]
- model_type: Either single or pairwise
- model_weights: The path to a .pth file that contains the weights of the selected model
- batch_size: The batch size that the loader will use. This will make things faster but will increase the GPU overhead
