import os

import pandas as pd
import torch
import torchio as tio

from BAScores.loader import validation_transform
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

    model.to(device)
    model.eval()

    y_preds = []
    indexes = []

    images = []
    img_indices = []

    with torch.no_grad():
        for idx, img in enumerate(os.listdir(in_dir)):
            img_path = os.path.join(in_dir, img)
            img = tio.ScalarImage(img_path)
            img = validation_transform(img)
            img_tensor = img.data  # type: ignore
            img_tensor = img_tensor.unsqueeze(0).float()

            images.append(img_tensor)
            img_indices.append(idx)

            if len(images) == batch_size or idx == len(os.listdir(in_dir)) - 1:
                batch = torch.cat(images, dim=0).to(device)
                batch_preds = model(batch).squeeze(dim=-1)

                y_preds.extend(batch_preds.tolist())
                indexes.extend(img_indices)

                images = []
                img_indices = []

    inference_res = pd.DataFrame({"Index": indexes, "Prediction": y_preds})
    out_path = os.path.join(out_dir, csv_name)
    inference_res.to_csv(out_path, index=False)
