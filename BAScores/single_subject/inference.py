import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from BAScores.loader import SingleSubjectDataloader
from BAScores.utils import load_single_model_weights


def inference(
    model: torch.nn.Module,
    model_weights: str,
    in_dir: str,
    out_dir: str,
    csv_name: str,
    device: str,
    batch_size: int = 16,
) -> None:

    load_single_model_weights(model, model_weights, device)

    single_loader = SingleSubjectDataloader(
        mode="inference",
        in_dir=in_dir,
    )
    dataloader = DataLoader(
        single_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count() // 2,  # type: ignore
        collate_fn=lambda x: torch.utils.data.dataloader.default_collate(x),
    )

    model.to(device)
    model.eval()

    y_preds = []
    with torch.no_grad():
        for idx, imgs in enumerate(dataloader):
            imgs = imgs.to(device)
            imgs = imgs.float()

            y_pred = model(imgs).squeeze(dim=-1)
            if len(y_pred) > 1:
                y_preds.extend(y_pred.cpu().tolist())
            else:
                y_preds.append(y_pred.cpu().item())

    inference_res = pd.DataFrame({"Prediction": y_preds})
    out_path = os.path.join(out_dir, csv_name)
    inference_res.to_csv(out_path, index=True)
