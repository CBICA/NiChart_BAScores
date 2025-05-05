from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchmetrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    NormalizedRootMeanSquaredError,
    R2Score,
)

from BAScores.utils import load_single_model_weights


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    model_weights: Optional[str] = None,
    verbose: bool = False,
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

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            X = X.float()
            y = y.float()

            y_pred = model(X).squeeze(dim=-1)

            mae.update(y_pred, y)
            mse.update(y_pred, y)
            nrmse.update(y_pred, y)
            r2_score.update(y_pred, y)

    eval_stats["eval_mae"] = mae.compute().item()
    eval_stats["eval_mse"] = mse.compute().item()
    eval_stats["eval_nrmse"] = nrmse.compute().item()
    eval_stats["eval_r2"] = r2_score.compute().item()

    if verbose:
        print(
            f"eval_mae: {eval_stats['eval_mae']:.4f} | "
            f"eval_mse: {eval_stats['eval_mse']:.4f} | "
            f"eval_nrmse: {eval_stats['eval_nrmse']:.4f} | "
            f"eval_r2: {eval_stats['eval_r2']:.4f}"
        )

    return eval_stats
