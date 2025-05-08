from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchmetrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    NormalizedRootMeanSquaredError,
    R2Score,
)
from typing_extensions import Literal

from BAScores.utils import load_single_model_weights, plot_preds_vs_truth


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: Literal["cuda", "mps", "cpu"] = "cuda",
    model_weights: Optional[str] = None,
    verbose: bool = False,
    plot_path: Optional[str] = None,
) -> dict:

    if model_weights is not None:
        load_single_model_weights(model, model_weights, device)

    model.to(device)
    model.eval()

    mae = MeanAbsoluteError().to(device)
    mse = MeanSquaredError().to(device)
    nrmse = NormalizedRootMeanSquaredError().to(device)
    r2_score = R2Score().to(device)

    eval_stats: dict = {
        "eval_mae": [],
        "eval_mse": [],
        "eval_nrmse": [],
        "eval_r2": [],
    }

    y_preds = []
    y_hats = []
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            X = X.float()
            y = y.float()

            y_pred = model(X).squeeze(dim=-1)

            if len(y_pred) > 1:
                y_preds.extend(y_pred.cpu().item())
                y_hats.extend(y.cpu().tolist())
            else:
                y_preds.append(y_pred.cpu().item())
                y_hats.append(y.cpu().tolist())

            mae.update(y_pred, y)
            mse.update(y_pred, y)
            nrmse.update(y_pred, y)
            r2_score.update(y_pred, y)

    eval_stats["eval_mae"] = mae.compute().item()
    eval_stats["eval_mse"] = mse.compute().item()
    eval_stats["eval_nrmse"] = nrmse.compute().item()
    eval_stats["eval_r2"] = r2_score.compute().item()

    if plot_path is not None:
        plot_preds_vs_truth(y_preds, y_hats, eval_stats, plot_path)

    if verbose:
        print(
            f"eval_mae: {eval_stats['eval_mae']:.4f} | "
            f"eval_mse: {eval_stats['eval_mse']:.4f} | "
            f"eval_nrmse: {eval_stats['eval_nrmse']:.4f} | "
            f"eval_r2: {eval_stats['eval_r2']:.4f}"
        )

    return eval_stats
