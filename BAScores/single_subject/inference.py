import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing_extensions import Literal

from BAScores.loader import SingleSubjectDataloader
from BAScores.utils import load_single_model_weights


def inference(
    model: torch.nn.Module,
    model_weights: str,
    in_dir: str,
    out_dir: str,
    csv_name: str,
    device: Literal["cuda", "mps", "cpu"] = "cuda",
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
        for idx, (imgs, mrids) in enumerate(dataloader):
            imgs = imgs.to(device)
            imgs = imgs.float()

            y_pred = model(imgs).squeeze(dim=-1)
            y_pred = y_pred.cpu().tolist()
            if isinstance(mrids, (list, tuple)):
                for mrid, pred in zip(mrids, y_pred):
                    y_preds.append((mrid, pred))
            else:
                y_preds.append((mrids, y_pred))

    inference_res = pd.DataFrame(
        {
            "MRID": [y_pred[0] for y_pred in y_preds],
            "Prediction": [y_pred[1] for y_pred in y_preds],
        }
    )
    out_path = os.path.join(out_dir, csv_name)
    inference_res.to_csv(out_path, index=False)
