# import os

# import pandas as pd

# import numpy as np
import torch

# from BAScores.loader import PairwiseDataloader, validation_transform
from BAScores.utils import load_pairwise_model_weights

# from torch.utils.data import DataLoader


# TODO: Complete the implementation, i have to create a loader that will push the images in pairs, it's not as simple as the single models
def inference(
    model: torch.nn.Module,
    model_weights: str,
    in_dir: str,
    out_dir: str,
    csv_name: str,
    device: str,
) -> None:

    load_pairwise_model_weights(model, model_weights, device)

    model.to(device)
    model.eval()

    # y_preds = []
    # indexes = []
    # with torch.no_grad():
    # for idx, (I1, I2, y1, y2) in enumerate(eval_dataloader):
    # I1, I2 = I1.to(device, dtype=torch.float32), I2.to(
    # device, dtype=torch.float32
    # )
    # y1, y2 = y1.to(device), y2.to(device)


#
# y1 = y1.float()
# y2 = y2.float()
# y_pred = model(I1, I2).squeeze(dim=-1)
#
# y_preds.append(y_pred.item())
# indexes.append(idx)
#
# inference_res = pd.DataFrame({"Index": indexes, "Prediction": y_preds})
# out_path = os.path.join(out_dir, csv_name)
# inference_res.to_csv(out_path, index=False)
