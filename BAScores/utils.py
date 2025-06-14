from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def get_prefix_oasis(img_name: str) -> str:
    """
    TODO: This one was specific, but will change
    """
    return str(img_name[: len(img_name) - 37])


def get_prefix(img_name: str) -> str:
    """
    Returns the prefix(the name or label) of the images. Note that, the images
    have a suffix of .nii.gz and the dlicv and LPS orientation labels.
    """
    return str(img_name[: len(img_name) - 28])


def load_single_model_weights(
    model: nn.Module, model_weights: str, device: str
) -> None:
    pre_state_dict = torch.load(model_weights, map_location=torch.device(device))

    model_state_keys = set(model.state_dict().keys())
    # Have to look at this a bit more
    # checkpoint_keys = set(pre_state_dict.keys())

    new_state_dict = OrderedDict()
    for k, v in pre_state_dict.items():
        if k.startswith("module."):
            new_k = k.replace("module.", "")
        else:
            new_k = k
        if new_k in model_state_keys:
            new_state_dict[new_k] = v
        else:
            print(
                f"Warning: Key '{new_k}' from checkpoint does not match any key in the model."
            )

    model.load_state_dict(new_state_dict, strict=False)


def load_pairwise_model_weights(
    model: nn.Module, model_weights: str, device: str
) -> None:
    pre_state_dict = torch.load(model_weights, map_location=torch.device(device))

    model_state_keys = set(model.state_dict().keys())
    checkpoint_keys = set(pre_state_dict.keys())

    if model_state_keys != checkpoint_keys:
        new_state_dict = OrderedDict()

        for k, v in pre_state_dict.items():
            if k.startswith("module."):  # If saved with DataParallel
                new_k = k.replace("module.", "")
            elif not any(
                k.startswith(prefix) for prefix in ["net.", "net.0.", "net.0.net."]
            ):
                new_k = "net." + k
            else:
                new_k = k

            new_state_dict[new_k] = v

        model.load_state_dict(
            new_state_dict, strict=False
        )  # Allow partial loading if needed
    else:
        model.load_state_dict(pre_state_dict)


def plot_preds_vs_truth(
    y_pred: list, y_hat: list, stats: dict, out_path: str = "."
) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, y_hat, alpha=0.6, color="blue")

    x_vals = np.linspace(min(y_pred), max(y_pred), 100)
    plt.plot(x_vals, x_vals, color="red", linestyle="--", label="Correlation")
    metrics_text = (
        f"MAE: {stats['eval_mae']:.4f}\n"
        f"MSE: {stats['eval_mse']:.4f}\n"
        f"NRMSE: {stats['eval_nrmse']:.4f}\n"
        f"R²: {stats['eval_r2']:.4f}"
    )
    plt.text(
        0.05,
        0.95,
        metrics_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.5, facecolor="white"),
    )
    plt.xlabel("Predicted Values", fontsize=14)
    plt.ylabel("Ground Truth Values", fontsize=14)
    plt.title("Predictions vs Ground Truth", fontsize=16)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(out_path, format="png", dpi=300)


class EarlyStopper:
    """
    Performs early stopping as PyTorch doesn't have an implemented one
    """

    def __init__(
        self, patience: int = 1, min_delta: float = 0.0, increase: bool = False
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.increase = increase
        self.best_value = float("-inf") if increase else float("inf")

    def early_stop(self, validation_val: float) -> bool:
        if not self.increase:
            if validation_val <= self.best_value:
                self.best_value = validation_val
                self.counter = 0
            else:
                self.counter += 1
        else:
            if validation_val >= self.best_value:
                self.best_value = validation_val
                self.counter = 0
            else:
                self.counter += 1

        print(
            f"Patience: {self.patience}, Counter: {self.counter}, Best: {self.best_value}, Current: {validation_val}"
        )
        return self.counter > self.patience


def init_weights(net: nn.Module) -> None:
    for block in net:
        for layer in block.children():
            if isinstance(layer, nn.Conv3d) or isinstance(layer, nn.LazyConv3d):
                for name, weight in layer.named_parameters():
                    if "weight" in name:
                        nn.init.kaiming_normal_(weight)
                    if "bias" in name:
                        nn.init.constant_(weight, 0.0)
