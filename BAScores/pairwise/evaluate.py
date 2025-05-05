from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchmetrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    NormalizedRootMeanSquaredError,
    R2Score,
)

from BAScores.utils import load_pairwise_model_weights, plot_preds_vs_truth


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    model_weights: Optional[str] = None,
    verbose: bool = False,
    plot_path: Optional[str] = None,
) -> dict:

    if model_weights is not None:
        load_pairwise_model_weights(
            model=model, model_weights=model_weights, device=device
        )

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
        for batch, (I1, I2, y1, y2) in enumerate(dataloader):
            I1, I2 = I1.to(device, dtype=torch.float32), I2.to(
                device, dtype=torch.float32
            )
            y1, y2 = y1.to(device), y2.to(device)

            y1 = y1.float()
            y2 = y2.float()

            y_pred = model(I1, I2).squeeze(dim=-1)

            if len(y_pred) > 1:
                y_preds.extend(y_pred.cpu().item())
                y_hats.extend((y1 - y2).cpu().tolist())
            else:
                y_preds.append(y_pred.cpu().item())
                y_hats.append((y1 - y2).cpu().tolist())

            mae.update(y_pred, y1 - y2)
            mse.update(y_pred, y1 - y2)
            nrmse.update(y_pred, y1 - y2)
            r2_score.update(y_pred, y1 - y2)

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
