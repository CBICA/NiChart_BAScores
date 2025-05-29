from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchmetrics import (
    AUROC,
    Accuracy,
    F1Score,
    MeanAbsoluteError,
    MeanSquaredError,
    NormalizedRootMeanSquaredError,
    Precision,
    R2Score,
    Recall,
    Specificity,
)
from typing_extensions import Literal

from BAScores.utils import load_single_model_weights, plot_preds_vs_truth


def evaluate(
    model: torch.nn.Module,
    mode: str,
    dataloader: DataLoader,
    num_classes: Optional[int] = None,
    device: Literal["cuda", "mps", "cpu"] = "cuda",
    model_weights: Optional[str] = None,
    verbose: bool = False,
    plot_path: Optional[str] = None,
) -> dict:

    if model_weights is not None:
        load_single_model_weights(model, model_weights, device)

    model.to(device)
    model.eval()

    if mode == "regression":
        assert num_classes is None
        eval_stats: dict = {
            "eval_mae": [],
            "eval_mse": [],
            "eval_nrmse": [],
            "eval_r2": [],
        }

        mae = MeanAbsoluteError().to(device)
        mse = MeanSquaredError().to(device)
        nrmse = NormalizedRootMeanSquaredError().to(device)
        r2_score = R2Score().to(device)
    else:
        eval_stats = {
            "eval_acc": [],
            "eval_auc": [],
            "eval_recall": [],
            "eval_precision": [],
            "eval_specificity": [],
            "eval_f1": [],
        }

        if mode == "multiclass":
            assert num_classes is not None
            acc = Accuracy(task=mode, average="macro", num_classes=num_classes).to(
                device
            )
            auc = AUROC(task=mode, average="macro", num_classes=num_classes).to(device)
            recall = Recall(task=mode, average="macro", num_classes=num_classes).to(
                device
            )
            precision = Precision(
                task=mode, average="macro", num_classes=num_classes
            ).to(device)
            specificity = Specificity(
                task=mode, average="macro", num_classes=num_classes
            ).to(device)
            f1_score = F1Score(task=mode, average="macro", num_classes=num_classes).to(
                device
            )
        else:
            acc = Accuracy(task=mode).to(device)
            auc = AUROC(task=mode).to(device)
            recall = Recall(task=mode).to(device)
            precision = Precision(task=mode).to(device)
            specificity = Specificity(task=mode).to(device)
            f1_score = F1Score(task=mode).to(device)

    y_preds = []
    y_hats = []
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            X, y = X.float(), y.float()

            y_pred = model(X).squeeze(dim=-1)

            if len(y_pred) > 1:
                y_preds.extend(y_pred.cpu().tolist())
                y_hats.extend(y.cpu().tolist())
            else:
                y_preds.append(y_pred.cpu().item())
                y_hats.append(y.cpu().item())

            if mode == "regression":
                mae.update(y_pred, y)
                mse.update(y_pred, y)
                nrmse.update(y_pred, y)
                r2_score.update(y_pred, y)
            else:
                y_pred_classes = (
                    (torch.sigmoid(y_pred) > 0.5).float()
                    if mode == "binary"
                    else torch.argmax(y_pred, dim=1)
                )
                acc.update(y_pred_classes, y)
                auc.update(y_pred_classes, y)
                recall.update(y_pred_classes, y)
                precision.update(y_pred_classes, y)
                specificity.update(y_pred_classes, y)
                f1_score.update(y_pred_classes, y)

    if mode == "regression":
        eval_stats["eval_mae"] = mae.compute().item()
        eval_stats["eval_mse"] = mse.compute().item()
        eval_stats["eval_nrmse"] = nrmse.compute().item()
        eval_stats["eval_r2"] = r2_score.compute().item()
    else:
        eval_stats["eval_acc"] = acc.compute().item()
        eval_stats["eval_auc"] = auc.compute().item()
        eval_stats["eval_recall"] = recall.compute().item()
        eval_stats["eval_precision"] = precision.compute().item()
        eval_stats["eval_specificity"] = specificity.compute().item()
        eval_stats["eval_f1"] = f1_score.compute().item()

    if plot_path is not None:
        plot_preds_vs_truth(y_preds, y_hats, eval_stats, plot_path)

    if verbose:
        if mode == "regression":
            print(
                f"eval_mae: {eval_stats['eval_mae']:.4f} | "
                f"eval_mse: {eval_stats['eval_mse']:.4f} | "
                f"eval_nrmse: {eval_stats['eval_nrmse']:.4f} | "
                f"eval_r2: {eval_stats['eval_r2']:.4f}"
            )
        else:
            print(
                f"eval_acc: {eval_stats['eval_acc']:.4f} | "
                f"eval_auc: {eval_stats['eval_auc']:.4f} | "
                f"eval_recall: {eval_stats['eval_recall']:.4f} | "
                f"eval_precision: {eval_stats['eval_precision']:.4f} | "
                f"eval_specificity: {eval_stats['eval_specificity']:.4f} | "
                f"eval_f1: {eval_stats['eval_f1']:.4f}"
            )

    return eval_stats
