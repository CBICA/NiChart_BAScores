import os
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from BAScores.loader import SingleSubjectDataloader, validation_transform
from BAScores.utils import load_single_model_weights


def inference(
    model: torch.nn.Module,
    model_weights: str,
    eval_dir: str,
    label_dict_csv: str,
    target: str,
    device: str,
) -> Any:

    # TODO: implement load_single_model_weights(waiting for the first single subject model to train)
    load_single_model_weights(model, model_weights, device)

    labels = pd.read_csv(label_dict_csv)
    label_dict = {row["MRID"]: row[target] for _, row in labels.iterrows()}

    eval_pairwise = SingleSubjectDataloader(
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
        for batch, (X, y) in enumerate(eval_dataloader):
            X = X.to(device)
            y = y.to(device)
            X = X.float()
            y = y.float()

            y_pred = model(X).squeeze()
            y_preds.append(y_pred)

    return y_preds
