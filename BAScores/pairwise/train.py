from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError, NormalizedRootMeanSquaredError, R2Score
from tqdm.auto import tqdm
from typing_extensions import Literal

from BAScores.pairwise.evaluate import evaluate
from BAScores.utils import EarlyStopper


def train_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: Literal["cuda", "mps", "cpu"] = "cuda",
) -> Dict:

    model.train()

    stats = {"train_mae": 0.0, "train_mse": 0.0, "train_nrmse": 0.0, "train_r2": 0.0}

    mse = MeanSquaredError().to(device)
    nrmse = NormalizedRootMeanSquaredError().to(device)
    r2_score = R2Score().to(device)

    for I1, I2, y1, y2 in dataloader:
        I1, I2 = I1.to(device), I2.to(device)
        y1, y2 = y1.to(device), y2.to(device)

        I1, I2 = I1.float(), I2.float()
        y1, y2 = y1.float(), y2.float()
        y = y2 - y1

        # zero grad
        optimizer.zero_grad()

        # forward
        y_pred = model(I1, I2).squeeze()

        # loss computation
        loss = loss_fn(y_pred, y)

        # loss backward
        loss.backward()

        # optimizer step
        optimizer.step()

        stats["train_mae"] += loss.item()
        mse.update(y_pred, y)
        nrmse.update(y_pred, y)
        r2_score.update(y_pred, y)

    stats["train_mae"] = stats["train_mae"] / float(len(dataloader))
    stats["train_mse"] = mse.compute().item()
    stats["train_nrmse"] = nrmse.compute().item()
    stats["train_r2"] = r2_score.compute().item()

    return stats


def test_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device: Literal["cuda", "mps", "cpu"] = "cuda",
) -> Dict:

    # eval mode
    model.eval()
    stats = {"test_mae": 0.0, "test_mse": 0.0, "test_nrmse": 0.0, "test_r2": 0.0}

    mse = MeanSquaredError().to(device)
    nrmse = NormalizedRootMeanSquaredError().to(device)
    r2_score = R2Score().to(device)

    with torch.no_grad():
        for I1, I2, y1, y2 in dataloader:
            I1, I2 = I1.to(device), I2.to(device)
            y1, y2 = y1.to(device), y2.to(device)

            I1, I2 = I1.float(), I2.float()
            y1, y2 = y1.float(), y2.float()
            y = y2 - y1

            test_pred_logits = model(I1, I2).squeeze()
            loss = loss_fn(test_pred_logits, y)

            stats["test_mae"] += loss.item()
            mse.update(test_pred_logits, y)
            nrmse.update(test_pred_logits, y)
            r2_score.update(test_pred_logits, y)

    stats["test_mae"] = stats["test_mae"] / float(len(dataloader))
    stats["test_mse"] = mse.compute().item()
    stats["test_nrmse"] = nrmse.compute().item()
    stats["test_r2"] = r2_score.compute().item()

    return stats


def train(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    patience: int,
    device: Literal["cuda", "mps", "cpu"] = "cuda",
    target_dir: str = ".",
    model_name: str = "NiChart_BAScores_best.pth",
    verbose: bool = False,
) -> Dict[str, List]:

    results: Dict[str, List] = {
        "train_mae": [],
        "train_mse": [],
        "train_nrmse": [],
        "train_r2": [],
        "test_mae": [],
        "test_mse": [],
        "test_nrmse": [],
        "test_r2": [],
    }

    best_loss = float("inf")
    early_stopper = EarlyStopper(patience=patience)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    for epoch in tqdm(range(epochs)):
        train_stats = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        test_stats = test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )

        if test_stats["test_mae"] < best_loss:
            best_loss = test_stats["test_mae"]
            save_model(model, target_dir, model_name)
            if verbose:
                print(
                    f"[INFO] New best model saved at {target_dir} with test MAE: {best_loss:.4f}"
                )

        if early_stopper.early_stop(test_stats["test_mae"]):
            if verbose:
                print("Early stopping the training!")
            break

        if verbose:
            print(
                f"Epoch: {epoch+1} | "
                f"train_mae: {train_stats['train_mae']:.4f} | "
                f"train_mse: {train_stats['train_mse']:.4f} | "
                f"train_nrmse: {train_stats['train_nrmse']:.4f} | "
                f"train_r2: {train_stats['train_r2']:.4f} | "
                f"test_mae: {test_stats['test_mae']:.4f} | "
                f"test_mse: {test_stats['test_mse']:.4f} | "
                f"test_nrmse: {test_stats['test_nrmse']:.4f} | "
                f"test_r2: {test_stats['test_r2']:.4f}"
            )

        results["train_mae"].append(train_stats["train_mae"])
        results["train_mse"].append(train_stats["train_mse"])
        results["train_nrmse"].append(train_stats["train_nrmse"])
        results["train_r2"].append(train_stats["train_r2"])
        results["test_mae"].append(test_stats["test_mae"])
        results["test_mse"].append(test_stats["test_mse"])
        results["test_nrmse"].append(test_stats["test_nrmse"])
        results["test_r2"].append(test_stats["test_r2"])

    eval_stats = evaluate(
        model=model,
        dataloader=eval_dataloader,
        device=device,
    )

    if verbose:
        print(
            "Evaluation results:\n"
            f"eval_mae: {eval_stats['eval_mae']:.4f} | "
            f"eval_mse: {eval_stats['eval_mse']:.4f} | "
            f"eval_nrmse: {eval_stats['eval_nrmse']:.4f} | "
            f"eval_r2: {eval_stats['eval_r2']:.4f}"
        )

    results["eval_mae"] = eval_stats["eval_mae"]
    results["eval_mse"] = eval_stats["eval_mse"]
    results["eval_nrmse"] = eval_stats["eval_nrmse"]
    results["eval_r2"] = eval_stats["eval_r2"]

    return results


def save_model(model: torch.nn.Module, target_dir: str, model_name: str) -> None:

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
