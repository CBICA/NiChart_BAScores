from collections import OrderedDict
from typing import Optional

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.nn import functional as F


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

    model.load_state_dict(new_state_dict, strict=True)


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
            elif k.startswith("linear.0."):
                if not model.meta:
                    new_k = k.replace("linear.0", "linear")
                else:
                    new_k = k
            elif k.startswith("net."):
                new_k = k.replace("net.", "")
            else:
                new_k = k

            new_state_dict[new_k] = v

        model.load_state_dict(new_state_dict, strict=True)
    else:
        model.load_state_dict(pre_state_dict)


def plot_preds_vs_truth(
    y_pred: list,
    y_hat: list,
    stats: dict,
    mode: str,
    out_path: str = ".",
) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, y_hat, alpha=0.6, color="blue")

    x_vals = np.linspace(min(y_pred), max(y_pred), 100)
    plt.plot(x_vals, x_vals, color="red", linestyle="--", label="Correlation")
    if mode == "regression":
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
    else:
        metrics_text = (
            f"Accuracy: {stats['eval_acc']}\n"
            f"AUC: {stats['eval_auc']}\n"
            f"Recall: {stats['eval_recall']}\n"
            f"Precision: {stats['eval_precision']}\n"
            f"Specificity: {stats['eval_specificity']}\n"
            f"F1: {stats['eval_f1']}\n"
        )

        cm = confusion_matrix(y_hat, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted Labels", fontsize=14)
        plt.ylabel("True Labels", fontsize=14)
        plt.title("Confusion Matrix", fontsize=16)

        plt.gcf().text(
            1.02,
            0.5,
            metrics_text,
            fontsize=12,
            verticalalignment="center",
            bbox=dict(boxstyle="round", alpha=0.5, facecolor="white"),
        )
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for metrics
        plt.savefig(out_path, format="png", dpi=300)
        plt.close()


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


def save_2d_attention(
    attention: torch.Tensor, output_dir: str, out_name: str, idx: Optional[int] = None
) -> None:
    _attention = attention.clone()
    _attention = _attention.cpu().numpy()

    plt.imshow(_attention)
    if idx is not None:
        plt.savefig(
            f"{output_dir}/{out_name}{idx}.png", bbox_inches="tight", pad_inches=0
        )
    else:
        plt.savefig(out_name, bbox_inches="tight", pad_inches=0)


def save_3d_attention(
    attention: torch.Tensor,
    niftii_header: nib.nifti1.Nifti1Header,
    output_dir: str,
    out_name: str,
) -> None:
    _attention = attention.clone()

    # normalize
    _attention -= _attention.mean((0, 1, 2), keepdim=True)
    _attention /= _attention.std((0, 1, 2), keepdim=True)

    dims = niftii_header.get_data_shape()
    if dims != _attention.shape:
        _attention = _attention.unsqueeze(0).unsqueeze(0)
        _attention = F.interpolate(
            _attention, size=dims, mode="trilinear", align_corners=False
        )
        _attention = _attention.squeeze(0).squeeze(0)
    _attention = _attention.cpu().numpy()
    att_img = nib.Nifti1Image(_attention, None, header=niftii_header)
    if ".nii.gz" in out_name:
        nib.save(att_img, f"{output_dir}/{out_name}")
    else:
        nib.save(att_img, f"{output_dir}/{out_name}.nii.gz")


def plot_tsne_clusters(features: list, out_path: str) -> None:
    plt.figure(figsize=(10, 8))
    plt.scatter(
        features[:, 0],
        features[:, 1],
        cmap=plt.cm.get_cmap("tab10", 10),
        s=15,
    )
    plt.title("t-SNE clusters plot")
    plt.savefig(out_path, format="png", dpi=300)
    plt.close()
