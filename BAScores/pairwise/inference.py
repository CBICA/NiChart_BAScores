import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from BAScores.loader import PairwiseDataloader
from BAScores.utils import load_pairwise_model_weights


def inference(
    model: torch.nn.Module,
    model_weights: str,
    in_dir: str,
    out_dir: str,
    csv_name: str,
    device: str,
    batch_size: int = 16,
) -> None:

    load_pairwise_model_weights(model, model_weights, device)

    pairwise_loader = PairwiseDataloader(
        mode="inference",
        in_dir=in_dir,
    )

    dataloader = DataLoader(
        pairwise_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count() // 2,  # type: ignore
        collate_fn=lambda x: torch.utils.data.dataloader.default_collate(x),
    )

    model.to(device)
    model.eval()

    y_preds = []
    with torch.no_grad():
        for idx, (I1, I2) in enumerate(dataloader):
            I1, I2 = I1.to(device), I2.to(device)
            I1, I2 = I1.float(), I2.float()

            y_pred = model(I1, I2).squeeze(dim=-1)
            if len(y_pred > 1):
                y_preds.extend(y_pred.cpu().tolist())
            else:
                y_preds.append(y_pred.cpu().item())

    inference_res = pd.DataFrame({"Prediction": y_preds})
    out_path = os.path.join(out_dir, csv_name)
    inference_res.to_csv(out_path, index=True)
