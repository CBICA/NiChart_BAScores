import os

import nibabel as nib
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing_extensions import Literal

from BAScores.loader import SingleSubjectDataloader
from BAScores.models.guided_back_propagation import GuidedBackPropagation
from BAScores.utils import load_single_model_weights


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

        @GuidedBackPropagation(output_dir=out_dir, device=device, mode=mode)
        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = model

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.net(x)

        model = _Model()

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
        collate_fn=lambda x: torch.utils.data.dataloader.default_collate(x),
    )

    model.to(device)
    model.eval()

    y_preds = []
    for _, (imgs, mrids) in enumerate(dataloader):
        preprocessed_img = nib.load(f"{in_dir}/{mrids[0]}_T1_LPS_dlicv_aligned.nii.gz")
        imgs = imgs.to(device)
        imgs = imgs.float()
        imgs.requires_grad_()

        if return_attention:
            y_pred = model(
                imgs,
                niftii_header=preprocessed_img.header.copy(),
                out_name=f"{mrids[0]}_T1_LPS_dlicv_aligned.nii.gz",
            ).squeeze(dim=-1)
        else:
            y_pred = model(imgs).squeeze(dim=-1)

        if mode == "binary":
            y_pred = (torch.sigmoid(y_pred) > 0.5).float()
        elif mode == "multiclass":
            y_pred = torch.argmax(y_pred, dim=1)

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
    out_path = os.path.join(out_dir, csv)
    output_dir = os.path.dirname(out_path)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    inference_res.to_csv(out_path, index=False)
