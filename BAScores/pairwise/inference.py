import os

import pandas as pd

# import numpy as np
import torch
from torch.utils.data import DataLoader

from BAScores.loader import PairwiseDataloader, validation_transform
from BAScores.utils import load_pairwise_model_weights


def inference(
    model: torch.nn.Module,
    model_weights: str,
    eval_dir: str,
    label_dict_csv: str,
    target: str,
    device: str,
    verbose: bool = False,
) -> list:
    # load weights of the pariwise model
    load_pairwise_model_weights(model, model_weights, device)

    # create eval dataloader
    labels = pd.read_csv(label_dict_csv)
    label_dict = {row["MRID"]: row[target] for _, row in labels.iterrows()}
    eval_pairwise = PairwiseDataloader(
        train_mode=False,
        in_dir=eval_dir,
        label_dict=label_dict,
        data_augmentation=validation_transform,
    )
    eval_dataloader = DataLoader(
        eval_pairwise, batch_size=1, shuffle=False, num_workers=os.cpu_count() // 2  # type: ignore
    )

    # perform inference
    model.to(device)
    model.eval()

    y_preds = []
    with torch.no_grad():
        for batch, (I1, I2, y1, y2) in enumerate(eval_dataloader):
            I1, I2 = I1.to(device, dtype=torch.float32), I2.to(
                device, dtype=torch.float32
            )
            y1, y2 = y1.to(device), y2.to(device)

            y1 = y1.float()
            y2 = y2.float()
            y_pred = model(I1, I2).squeeze(dim=-1)

            y_preds.append(y_pred)

    return y_preds
