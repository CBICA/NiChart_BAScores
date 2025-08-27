### How to perform training

```bash
NiChart_BAScores train \
                 --in_dir [input directory] \
                 --out_dir [output directory] \
                 --mode [binary, multiclass, regression] \
                 --label_dict [path to .csv] \
                 --model_name [output model name(.pth)] \
                 --device [cuda, mps, cpu] \
                 --model_type [single, pairwise] \
                 --model [model name(for example: resnet18)] \
                 --meta [True of False(for pairwise only)] \
                 --pretrained_single [path to single input model weights] \
                 --pretrained_pairwise [path to pairwise model weights] \
                 --target [target name(same as in label dict)] \
                 --optimizer [Optimizer name(See below)] \
                 --batch_size [batch size(int)] \
                 --lr [learning rate(float)] \
                 --weight_decay [weight decay/L2 regularization(float)] \
                 --epochs [total epochs(int)] \
                 --patience [patiance(int)] \
                 --verbose \
```

- in_dir: The input directory that contains the train, test and eval folders with LPS oriented, DLICV preprocessed T1 images
- out_dir: The output directory that the file of the model weights with name: `model_name` will be saved
- mode: Either regression, binary or multiclass. The last two corresponds to classification.
- label_dict: The path to a .csv file that must contain a column with the MRID's and a column with the values of the targets
- model_name: See above
- device: Either cuda, mps or cpu
- model: The type of backbone, currently supported: [resnet18, resnet34]
- model_type: Either single or pairwise
- meta: Only for the pairwise models, if set to True for single input models it won't affect anything. If True, it enables concatenation of the features with the difference.
- pretrained_single: You can set pretrained weights as a starting phase for training, you can only pass single input type of weights here
- pretrained_pairwise: You can use pairwise weights as well, if you do that, only the backbone weights will be loaded(can't use single input type weights for pairwise models)
- target: The name of the target value(same as in the label dict)
- optimizer: We support a limited amount of tested optimizer for brain age values, you can see them at the [main](../BAScores/__main__.py)
- batch_size: The batch size that the loader will use
- lr: The learning rate
- weight_decay: The L2 regularization value
- epochs: The number of epochs that the model will be trained for
- patience: The number of epochs that the model will continue to train without the loss getting better
- verbose: Set this if you want to output information about training and evaluation of the model
