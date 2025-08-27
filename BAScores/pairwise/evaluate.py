from typing import Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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

from BAScores.utils import (
    load_pairwise_model_weights,
    plot_preds_vs_truth,
    plot_tsne_clusters,
)


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
        load_pairwise_model_weights(
            model=model,
            model_weights=model_weights,
            device=device,
        )

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
    y_features = []
    with torch.no_grad():
        for batch, (I1, I2, y1, y2) in enumerate(dataloader):
            I1, I2 = I1.to(device), I2.to(device)
            y1, y2 = y1.to(device), y2.to(device)

            I1, I2 = I1.float(), I2.float()
            if mode == "regression":
                y1, y2 = y1.float(), y2.float()
                y = y2 - y1
            else:
                y1, y2 = y1.long(), y2.long()
                y = y2

            y_pred, y_feature = model(I1, I2, return_features=True)
            y_pred = y_pred.squeeze(dim=-1)

            if mode != "regression":
                y_feature = np.array(y_feature.squeeze().cpu())

                if y_feature.ndim != 1:
                    y_feature = y_feature.flatten()
                y_features.append(y_feature)

            if mode == "regression":
                y_preds.append(y_pred.cpu().item())
                y_hats.append(y.cpu().item())

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

                y_preds.append(y_pred_classes.cpu().item())
                y_hats.append(y.cpu().item())

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
        plot_preds_vs_truth(y_preds, y_hats, eval_stats, mode, plot_path)
        if mode != "regression":
            PCA_features = PCA(n_components=50).fit_transform(y_features)
            TSNE_features = TSNE(
                n_components=2,
                learning_rate="auto",
                init="pca",
                perplexity=30,
            ).fit_transform(PCA_features)
            plot_tsne_clusters(TSNE_features, y_hats, plot_path[:-4] + "_tsne.png")

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
