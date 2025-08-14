import os

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from typing_extensions import Literal, Optional

from BAScores.loader import PairwiseDataloader
from BAScores.utils import load_pairwise_model_weights, plot_tsne_clusters


def inference(
    model: torch.nn.Module,
    mode: str,
    model_weights: str,
    in_dir: str,
    out_dir: str,
    in_csv: str,
    csv: str,
    device: Literal["cuda", "mps", "cpu"] = "cuda",
    plot_path: Optional[str] = None,
) -> None:

    load_pairwise_model_weights(model=model, model_weights=model_weights, device=device)

    in_csv = pd.read_csv(in_csv)
    pairwise_loader = PairwiseDataloader(
        mode="inference",
        in_dir=in_dir,
        in_csv=in_csv,
    )

    dataloader = DataLoader(
        pairwise_loader,
        shuffle=False,
        batch_size=1,
        num_workers=os.cpu_count() // 2,  # type: ignore
    )

    model.to(device)
    model.eval()

    y_preds = []
    predicted_features = []
    with torch.no_grad():
        for idx, (I1, I2, mrid1, mrid2) in enumerate(dataloader):
            I1, I2 = I1.to(device), I2.to(device)
            I1, I2 = I1.float(), I2.float()

            y_pred, y_features = model(I1, I2, return_features=True)
            y_pred = y_pred.squeeze(dim=-1)
            y_features = np.array(y_features.squeeze().cpu())
            if y_features.ndim != 1:
                y_features = y_features.flatten()
            predicted_features.append(y_features)
            y_pred = y_pred.cpu().item()
            y_preds.append((y_pred, mrid1, mrid2))

    inference_res = pd.DataFrame(
        {
            "MRID1": [y_pred[1][0] for y_pred in y_preds],
            "MRID2": [y_pred[2][0] for y_pred in y_preds],
            "Prediction": [y_pred[0] for y_pred in y_preds],
        }
    )
    output_dir = os.path.dirname(csv)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    inference_res.to_csv(csv, index=False)

    if plot_path is not None:
        PCA_features = PCA(n_components=30).fit_transform(predicted_features)
        TSNE_features = TSNE(
            n_components=2, learning_rate="auto", init="random", perplexity=30
        ).fit_transform(PCA_features)
        plot_tsne_clusters(TSNE_features, plot_path)
