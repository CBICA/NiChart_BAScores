import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from medcam import medcam
from torch import nn
from torch.utils.data import DataLoader
from typing_extensions import Literal

from BAScores.loader import SingleSubjectDataloader
from BAScores.models.guided_back_propagation import GuidedBackPropagation
from BAScores.utils import load_single_model_weights, save_attention_as_niftii


def inference(
    model: torch.nn.Module,
    mode: str,
    model_weights: str,
    in_dir: str,
    out_dir: str,
    csv: str,
    return_attention: bool = False,
    device: Literal["cuda", "mps", "cpu"] = "cuda",
) -> None:
    load_single_model_weights(model, model_weights, device)

    if return_attention:
        model = medcam.inject(
            model,
            output_dir=out_dir,
            backend="gbp",
            save_maps=False,
            return_attention=True,
        )

    single_loader = SingleSubjectDataloader(
        mode="inference",
        in_dir=in_dir,
    )
    dataloader = DataLoader(
        single_loader,
        batch_size=1,
        shuffle=False,
        num_workers=os.cpu_count() // 2,  # type: ignore
        pin_memory=True,
    )

    model.to(device)
    model.eval()

    y_preds = []
    with torch.no_grad():
        for idx, (imgs, mrids) in enumerate(dataloader):
            affine = nib.load(f"{in_dir}/{mrids[0]}_T1_LPS_dlicv_aligned.nii.gz").affine
            imgs = imgs.to(device)
            imgs = imgs.float()

            y_pred, attention = model(imgs)
            y_pred = y_pred.view(-1)
            attention = np.squeeze(attention)

            y_pred = y_pred.cpu().tolist()
            attention = attention.cpu().numpy()
            save_attention_as_niftii(
                attention,
                affine,
                f"../{mrids[0]}_T1_LPS_dlicv_aligned.nii.gz",
            )

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
