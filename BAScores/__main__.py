import argparse
from typing import Any

import torch

from BAScores.loader import create_dataloaders
from BAScores.models.alexnet3d import AlexNet3D
from BAScores.models.pairwise import PairwiseModel3D
from BAScores.models.resnet3d import ResNet3D
from BAScores.pairwise.evaluate import evaluate as evaluate_pairwise
from BAScores.pairwise.inference import inference as inference_pairwise
from BAScores.pairwise.train import train as train_pairwise
from BAScores.single_subject.evaluate import evaluate as evaluate_single
from BAScores.single_subject.inference import inference as inference_single
from BAScores.single_subject.train import train as train_single


def available_models(num_classes: int, device: str, dropout: float) -> dict:
    return {
        "resnet18": ResNet3D(
            arch=((2, 64), (2, 128), (2, 256), (2, 512)),
            num_classes=1 if num_classes == -1 else num_classes,
            device=device,
        ),
        "resnet18-cbam": ResNet3D(
            arch=((2, 64), (2, 128), (2, 256), (2, 512)),
            num_classes=1 if num_classes == -1 else num_classes,
            device=device,
            cbam=True,
        ),
        "resnet34": ResNet3D(
            arch=((3, 64), (4, 128), (6, 256), (3, 512)),
            num_classes=1 if num_classes == -1 else num_classes,
            device=device,
        ),
        "resnet34-cbam": ResNet3D(
            arch=((3, 64), (4, 128), (6, 256), (3, 512)),
            num_classes=1 if num_classes == -1 else num_classes,
            device=device,
            cbam=True,
        ),
        "alexnet": AlexNet3D(
            num_classes=1 if num_classes == -1 else num_classes,
            device=device,
            dropout=dropout,
        ),
    }


def select_optimizer(
    model: Any, optimizer: str, lr: float, weight_decay: float
) -> torch.optim:
    AVAILABLE_OPTIMIZERS = {
        "Adam": torch.optim.Adam(
            params=model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        ),
        "AdamW": torch.optim.AdamW(
            params=model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            amsgrad=True,
        ),
        "Adadelta": torch.optim.Adadelta(
            params=model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        ),
        "SGD": torch.optim.SGD(
            params=model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        ),
    }  # We support a limited amount of optimizers as they are the ones that we tested on our task and had the best results
    assert (
        optimizer in AVAILABLE_OPTIMIZERS.keys()
    ), f"Please provide a currently supported optimizer: {AVAILABLE_OPTIMIZERS.keys()}"
    return AVAILABLE_OPTIMIZERS[optimizer]


def run_train(args: Any) -> None:

    assert args.lr > 0.0
    assert args.weight_decay >= 0.0
    assert args.batch_size > 0
    assert args.epochs > 0
    assert args.patience >= 0

    assert args.mode in [
        "regression",
        "multiclass",
        "binary",
    ]
    if args.mode == "multiclass":
        assert (
            args.num_classes > 0
        ), "Please set the number of classes if you perform multiclass classification"
    if args.mode == "regression":
        assert args.num_classes == -1
    assert args.model_type in [
        "single",
        "pairwise",
    ], "Please provide one of the following model types: [single, pairwise]"
    assert args.device in [
        "cuda",
        "mps",
        "cpu",
    ], "Please provide one of the following devices: [cuda, mps, cpu]"

    AVAILABLE_MODELS = available_models(args.num_classes, args.device, args.dropout)

    assert (
        args.model in AVAILABLE_MODELS.keys()
    ), f"Please provide a currently supported model: {AVAILABLE_MODELS.keys()}"

    encoder = AVAILABLE_MODELS[args.model]
    if args.mode == "regression":
        loss_fn = torch.nn.L1Loss(reduction="mean")  # MAE loss
    elif args.mode == "multiclass":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:  # binary
        loss_fn = torch.nn.BCEWithLogitsLoss()

    train_dataloader, test_dataloader, eval_dataloader = create_dataloaders(
        in_dir=args.in_dir,
        batch_size=args.batch_size,
        label_dict_csv=args.label_dict,
        target=args.target,
        pairwise=True if args.model_type == "pairwise" else False,
    )
    if args.model_type == "single":
        model = encoder
        optimizer = select_optimizer(
            model=model,
            optimizer=args.optimizer,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        _ = train_single(
            model=model,
            mode=args.mode,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            eval_dataloader=eval_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=args.epochs,
            patience=args.patience,
            num_classes=None if args.num_classes == -1 else args.num_classes,
            device=args.device,
            target_dir=args.out_dir,
            model_name=args.model_name,
            verbose=args.verbose,
        )
    else:
        model = PairwiseModel3D(encoder, meta=args.meta, device=args.device)
        optimizer = select_optimizer(
            model=model,
            optimizer=args.optimizer,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        _ = train_pairwise(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            eval_dataloader=eval_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=args.epochs,
            patience=args.patience,
            device=args.device,
            target_dir=args.out_dir,
            model_name=args.model_name,
            verbose=args.verbose,
        )


def run_evaluate(args: Any) -> None:

    if args.mode == "multiclass":
        assert (
            args.num_classes > 0
        ), "Please set the number of classes if you perform multiclass classification"
    if args.mode == "regression":
        assert args.num_classes == -1

    AVAILABLE_MODELS = available_models(args.num_classes, args.device, args.dropout)
    assert (
        args.model in AVAILABLE_MODELS.keys()
    ), f"Please provide a currently supported model: {AVAILABLE_MODELS.keys()}"
    encoder = AVAILABLE_MODELS[args.model]
    _, _, eval_dataloader = create_dataloaders(
        in_dir=args.in_dir,
        batch_size=1,
        label_dict_csv=args.label_dict,
        target=args.target,
        pairwise=True if args.model_type == "pairwise" else False,
    )
    if args.model_type == "single":
        model = encoder
        _ = evaluate_single(
            model=model,
            mode=args.mode,
            dataloader=eval_dataloader,
            num_classes=None if args.num_classes == -1 else args.num_classes,
            device=args.device,
            model_weights=args.model_weights,
            verbose=args.verbose,
            plot_path=None if args.plot_path == "None" else args.plot_path,
        )
    else:
        model = PairwiseModel3D(encoder=encoder, meta=args.meta, device=args.device)
        _ = evaluate_pairwise(
            model=model,
            mode=args.mode,
            dataloader=eval_dataloader,
            device=args.device,
            model_weights=args.model_weights,
            verbose=args.verbose,
            plot_path=None if args.plot_path == "None" else args.plot_path,
        )


def run_inference(args: Any) -> None:
    assert args.mode in ["regression", "multiclass", "binary"]
    AVAILABLE_MODELS = available_models(args.num_classes, args.device, args.dropout)
    assert (
        args.model in AVAILABLE_MODELS.keys()
    ), f"Please provide a currently supported model: {AVAILABLE_MODELS.keys()}"
    encoder = AVAILABLE_MODELS[args.model]
    if args.model_type == "single":
        model = encoder
        inference_single(
            model=model,
            mode=args.mode,
            model_weights=args.model_weights,
            in_dir=args.in_dir,
            out_dir=args.out_dir,
            csv=args.csv,
            return_attention=True if args.return_attention else False,
            device=args.device,
        )
    else:
        # This needs a bit of work
        model = PairwiseModel3D(encoder=encoder, meta=args.meta, device=args.device)
        assert (
            args.in_csv != "None"
        ), "You have to provide the input csv for pairwise inference"
        inference_pairwise(
            model=model,
            mode=args.mode,
            model_weights=args.model_weights,
            in_dir=args.in_dir,
            out_dir=args.out_dir,
            in_csv=args.in_csv,
            csv=args.csv,
            device=args.device,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="NiChart_BASscores",
        description="BAScores - Image to biomarkers models for NiChart",
        usage="",
        add_help=False,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    train = subparsers.add_parser("train", help="Selects training mode")
    train.add_argument(
        "-i",
        "--in_dir",
        type=str,
        required=True,
        help="[REQUIRED] Input folder with LPS oriented T1 sMRI images in Nifti format (nii.gz). \
              The input folder must contain a train, test and eval folder.",
    )

    train.add_argument(
        "--label_dict",
        type=str,
        required=True,
        help="[REQUIRED] The name of the .csv file that contains information about your dataset. \
              The .csv file must contain MRID and target columns that corresponds to the continuous target value of the specific MRID.",
    )

    train.add_argument(
        "--mode",
        type=str,
        required=True,
        help="[REQUIRED] Either regression, multiclass or binary",
    )

    train.add_argument(
        "--num_classes",
        type=int,
        required=False,
        default=-1,
        help="Only if mode != regression. Set the number of classes",
    )

    train.add_argument(
        "--target",
        type=str,
        required=True,
        help="[REQUIRED] The target column to predict(i.e. Age)",
    )

    train.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default=".",
        required=False,
        help="[REQUIRED] Output directory where the model weights will be saved.",
    )

    train.add_argument(
        "--model_name",
        type=str,
        default="NiChart_BAScores_best.pth",
        required=False,
        help="The name of the .pth model that will be saved after training. Default: NiChart_BAScores_best.pth",
    )

    train.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda",
        required=False,
        help="[REQUIRED] The device that training/inference will run with.",
    )

    train.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="[REQUIRED] Either single or pairwise",
    )

    train.add_argument(
        "--model",
        type=str,
        default="resnet18",
        required=False,
        help="The encoder that will be used. Currently available: [resnet18, resnet34]",
    )

    train.add_argument(
        "--meta",
        type=bool,
        required=False,
        default=False,
        help="set meta to True for pairwise models",
    )

    train.add_argument(
        "--lr",
        type=float,
        default=0.01,
        required=False,
        help="The initial learning rate for the adaptive optimizers. Default: 0.001(Need to change if Adadelta is used!)",
    )

    train.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        required=False,
        help="The passed L2 regularization value. Default: 0.0",
    )

    train.add_argument(
        "--epochs",
        type=int,
        default=50,
        required=False,
        help="The total number of epochs that the model will be trained. Default: 50",
    )

    train.add_argument(
        "--patience",
        type=int,
        default=10,
        required=False,
        help="The maximum number of epochs that the model will continue to train without the loss going down. Default: 10",
    )

    train.add_argument(
        "--batch_size",
        type=int,
        default=32,
        required=False,
        help="The batch size that will be used for training. Default: 32",
    )

    train.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        required=False,
        help="The optimizer that will be used for training. Default: AdamW",
    )

    train.add_argument(
        "--verbose",
        action="store_true",
        help="Provides additional details(in log files) when set.",
    )

    train.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        required=False,
        help="[Only for AlexNet3D] Set the dropout value for the last layers",
    )
    train.set_defaults(func=run_train)

    # evaluate
    evaluate = subparsers.add_parser("evaluate", help="Selects evaluate mode")

    evaluate.add_argument(
        "-i",
        "--in_dir",
        type=str,
        required=True,
        help="[REQUIRED] Input dir with the eval folder",
    )

    evaluate.add_argument(
        "--model",
        type=str,
        default="resnet18",
        required=False,
        help="The encoder that will be used. Currently available: [resnet18, resnet34]",
    )

    evaluate.add_argument(
        "--mode",
        type=str,
        required=True,
        help="[REQUIRED] Either regression, multiclass or binary",
    )

    evaluate.add_argument(
        "--num_classes",
        type=int,
        required=False,
        default=-1,
        help="Only if mode != regression. Set the number of classes",
    )

    evaluate.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="[REQUIRED] Either single or pairwise",
    )

    evaluate.add_argument(
        "--meta",
        type=bool,
        required=False,
        default=False,
        help="Set meta to True for pairwise models",
    )

    evaluate.add_argument(
        "--model_weights",
        type=str,
        required=True,
        help="[REQUIRED] The path to the weights that will be used for inference",
    )

    evaluate.add_argument(
        "--label_dict",
        type=str,
        required=True,
        help="[REQUIRED] The name of the .csv file that contains information about your dataset. \
              The .csv file must contain MRID and target columns that corresponds to the continuous target value of the specific MRID.",
    )

    evaluate.add_argument(
        "--target",
        type=str,
        required=True,
        help="[REQUIRED] The target column to predict(i.e. Age)",
    )

    evaluate.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda",
        required=False,
        help="The device that evaluation will run with. Default: cuda",
    )

    evaluate.add_argument(
        "--verbose",
        action="store_true",
        help="Provides additional details when set.",
    )

    evaluate.add_argument(
        "--plot_path",
        type=str,
        default="None",
        required=False,
        help="The path that the prediction vs ground truth plot will be saved. Default: None",
    )

    evaluate.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        required=False,
        help="[Only for AlexNet3D] Set the dropout value for the last layers",
    )
    evaluate.set_defaults(func=run_evaluate)

    # inference
    inference = subparsers.add_parser("inference", help="Selects inference mode")

    inference.add_argument(
        "-i",
        "--in_dir",
        type=str,
        required=True,
        help="[REQUIRED] Input folder with LPS oriented T1 sMRI images in Nifti format (nii.gz).",
    )

    inference.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default=".",
        required=False,
        help="Output directory where the output csv will be saved. Default: Current dir",
    )

    inference.add_argument(
        "--in_csv",
        type=str,
        required=False,
        default="None",
        help="Provide the input csv: in_csv only with pairwise models",
    )

    inference.add_argument(
        "--csv",
        type=str,
        default="inference_res.csv",
        required=False,
        help="The name of the csv that contains the indexes and values of the predictions. Default: inference_res.csv",
    )

    inference.add_argument(
        "--model",
        type=str,
        default="resnet18",
        required=False,
        help="The encoder that will be used. Currently available: [resnet18, resnet34]",
    )

    inference.add_argument(
        "--mode",
        type=str,
        required=True,
        help="[REQUIRED] Either regression, binary or multiclass",
    )

    inference.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="[REQUIRED] Either single or pairwise",
    )

    inference.add_argument(
        "--meta",
        type=bool,
        required=False,
        default=False,
        help="Set meta to True for pairwise models",
    )

    inference.add_argument(
        "--model_weights",
        type=str,
        required=True,
        help="[REQUIRED] The path to the weights that will be used for inference",
    )

    inference.add_argument(
        "--num_classes",
        type=int,
        required=False,
        default=-1,
        help="Only if mode != regression. Set the number of classes",
    )

    inference.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda",
        required=False,
        help="The device that evaluation will run with. Default: cuda",
    )

    inference.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        required=False,
        help="[Only for AlexNet3D] Set the dropout value for the last layers",
    )

    inference.add_argument(
        "--return_attention",
        action="store_true",
        help="Returns the attention maps in the specified output directory",
    )
    inference.set_defaults(func=run_inference)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
