import itertools
import os
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torchio as tio
from torch.utils.data import DataLoader, Dataset
from typing_extensions import Literal

from BAScores.utils import get_prefix


class SingleSubjectDataloader(Dataset):
    """
    Class for the SingleSubject Dataloader. Returns the input tensor with it's label(as a torch.tensor)

    :param mode: train, evaluate or inference
    :type mode: str
    :param in_dir: The input directory that contains train/ test/ and eval/ folders
    :type in_dir: str
    :param label_dict: A dictionary that maps MRID to target value
    :type label_dict: Optional[dict]
    :param data_augmentation: True if you want data augmentation to be applied, False otherwise
    :type data_augmentation: bool
    """

    def __init__(
        self,
        mode: Literal["train", "evaluate", "inference"],
        in_dir: str,
        label_dict: Optional[dict] = None,
        data_augmentation: bool = False,
    ) -> None:
        super().__init__()

        assert mode in ["train", "evaluate", "inference"]
        if label_dict is not None:
            assert (
                mode != "inference"
            ), "Can't perform inference with a passed label dict."
        else:
            assert (
                mode == "inference"
            ), "Can't perform training or evaluation without a label dict."

        self.in_dir = Path(in_dir)
        self.data_augmentation = data_augmentation
        self.mode = mode
        self.label_dict = label_dict
        self.images: list = []
        self.img_label: list = []
        self.mrids: list = []
        self._fetch_images()
        if mode != "inference":
            self._fetch_labels()

    def _fetch_images(self) -> None:
        for img_name in os.listdir(self.in_dir):
            if img_name.endswith(".nii.gz"):
                image_path = self.in_dir / img_name
                self.images.append(image_path)
                self.mrids.append(get_prefix(img_name))

    def _fetch_labels(self) -> None:
        for img in self.images:
            if self.label_dict is not None:
                label = self.label_dict.get(get_prefix(img.name), 0)
            self.img_label.append((img, label))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Any:
        if self.mode != "inference":
            img_path, label = self.img_label[index]
        else:
            img_path = self.images[index]
            mrid = self.mrids[index]

        img = tio.ScalarImage(img_path)
        resize = tio.Resize((128, 128, 128))
        img = resize(img)

        transforms = []
        if self.data_augmentation:
            if self.mode == "train":
                if random.randint(0, 2):
                    transforms.append(tio.RandomBlur(2))
                if random.randint(0, 2):
                    transforms.append(tio.RandomGamma(0.3))
                if random.randint(0, 2):
                    transforms.append(tio.RandomFlip(axes=("LR",)))
                if random.randint(0, 2):
                    tio.RandomNoise(mean=0, std=2),
                if random.randint(0, 2):
                    transforms.append(
                        tio.Affine(
                            scales=(1, 1, 1),
                            degrees=tuple(np.random.uniform(low=-40, high=40, size=3)),
                            translation=tuple(
                                np.random.uniform(low=-10, high=10, size=3)
                            ),
                            image_interpolation="linear",
                            default_pad_value="minimum",
                        )
                    )
        if len(transforms) > 0:
            single_transforms = tio.Compose(transforms)
            img = single_transforms(img)

        img_tensor = torch.tensor(img.data, dtype=torch.float32)

        if self.mode != "inference":
            return img_tensor, torch.tensor(label, dtype=torch.float32)
        else:
            return img_tensor, mrid


class PairwiseDataloader(Dataset):
    """
    Class for the Pairwise Dataloader. Returns the input tensors with their labels(as a torch.tensor)

    :param mode: train, evaluate or inference
    :type mode: str
    :param in_dir: The input directory that contains train/ test/ and eval/ folders
    :type in_dir: str
    :param label_dict: A dictionary that maps MRID to target value
    :type label_dict: Optional[dict]
    :param data_augmentation: True if you want data augmentation to be applied, False otherwise
    :type data_augmentation: bool
    """

    def __init__(
        self,
        mode: Literal["train", "evaluate", "inference"],
        in_dir: str,
        in_csv: pd.DataFrame,
        label_dict: Optional[dict] = None,
        data_augmentation: bool = False,
    ) -> None:
        super().__init__()

        assert mode in ["train", "evaluate", "inference"]

        self.in_dir = Path(in_dir)
        self.data_augmentation = data_augmentation
        self.mode = mode
        self.label_dict = label_dict
        self.in_csv = in_csv
        self.img_suffix = "_T1_LPS_dlicv_aligned.nii.gz"
        self.mrids: list = []
        self.pairs = self._create_pairs()

    def _create_pairs(self) -> list:
        pairs: list = []

        ptid_groups = self.in_csv.sort_values(by=["PTID", "Age"]).groupby("PTID")

        for ptid, group in ptid_groups:
            mrids = group["MRID"].tolist()
            subject_pairs = list(itertools.combinations(mrids, 2))
            for mrid1, mrid2 in subject_pairs:
                img1_path = self.in_dir / (mrid1 + self.img_suffix)
                img2_path = self.in_dir / (mrid2 + self.img_suffix)

                if not img1_path.exists() or not img2_path.exists():
                    continue

                if self.mode == "inference":
                    self.mrids.append((mrid1, mrid2))
                    pairs.append((img1_path, img2_path))
                else:
                    if self.label_dict is not None:
                        label1 = self.label_dict[ptid][mrid1]
                        label2 = self.label_dict[ptid][mrid2]
                        # assert label2 >= label1
                        pairs.append((img1_path, img2_path, label1, label2))

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Any:
        if self.mode == "inference":
            img1_path, img2_path = self.pairs[index]
        else:
            img1_path, img2_path, label1, label2 = self.pairs[index]
        img1 = tio.ScalarImage(img1_path)
        img2 = tio.ScalarImage(img2_path)

        resize = tio.Resize((128, 128, 128))

        img1 = resize(img1)
        img2 = resize(img2)

        transforms = []
        if self.data_augmentation:
            if self.mode == "train":
                if random.randint(0, 2):
                    transforms.append(tio.RandomBlur(2))
                if random.randint(0, 2):
                    transforms.append(tio.RandomGamma(0.3))
                if random.randint(0, 2):
                    transforms.append(tio.RandomFlip(axes=("LR",)))
                if random.randint(0, 2):
                    tio.RandomNoise(mean=0, std=2),
                if random.randint(0, 2):
                    transforms.append(
                        tio.Affine(
                            scales=(1, 1, 1),
                            degrees=tuple(np.random.uniform(low=-40, high=40, size=3)),
                            translation=tuple(
                                np.random.uniform(low=-10, high=10, size=3)
                            ),
                            image_interpolation="linear",
                            default_pad_value="minimum",
                        )
                    )
        if len(transforms) > 0:
            pairwise_transforms = tio.Compose(transforms)
            img1 = pairwise_transforms(img1)
            img2 = pairwise_transforms(img2)

        img1_tensor = torch.tensor(img1.data, dtype=torch.float32)
        img2_tensor = torch.tensor(img2.data, dtype=torch.float32)

        if self.mode != "inference":
            return (
                img1_tensor,
                img2_tensor,
                torch.tensor(label1, dtype=torch.float32),
                torch.tensor(label2, dtype=torch.float32),
            )

        return img1_tensor, img2_tensor, self.mrids[index][0], self.mrids[index][1]


def create_dataloaders(
    in_dir: str,
    batch_size: int,
    label_dict_csv: str,
    target: str,
    pairwise: bool = False,
) -> Any:
    """
    Creates the train, test and eval dataloader

    :param in_dir: The input directory that contains train/ test/ and eval/ folders
    :type in_dir: str
    :param batch_size: The passed batch_size that the data will be splitted into
    :type batch_size: int
    :param label_dict_csv: The complete path to the .csv that contains the MRID and the target columns
    :type label_dict: str
    :param pairwise: True if you want Pairwise Dataloaders, False otherwise
    :type pairwise: bool
    """

    train_dir = f"{in_dir}/train"
    test_dir = f"{in_dir}/test"
    eval_dir = f"{in_dir}/eval"

    in_csv = pd.read_csv(label_dict_csv)

    if pairwise:
        label_dict = {}  # type: ignore
        for _, row in in_csv.iterrows():
            if row["PTID"] not in label_dict:
                label_dict[row["PTID"]] = {}
            label_dict[row["PTID"]][row["MRID"]] = row[target]
    else:
        label_dict = {row["MRID"]: row[target] for _, row in in_csv.iterrows()}

    if pairwise:
        if os.path.isdir(train_dir):
            train_loader = PairwiseDataloader(
                mode="train",
                in_dir=train_dir,
                in_csv=in_csv,
                label_dict=label_dict,
                data_augmentation=True,
            )
        else:
            train_loader = None

        if os.path.isdir(test_dir):
            test_loader = PairwiseDataloader(
                mode="evaluate",
                in_dir=test_dir,
                in_csv=in_csv,
                label_dict=label_dict,
                data_augmentation=False,
            )
        else:
            test_loader = None

        if os.path.isdir(eval_dir):
            eval_loader = PairwiseDataloader(
                mode="evaluate",
                in_dir=eval_dir,
                in_csv=in_csv,
                label_dict=label_dict,
                data_augmentation=False,
            )
        else:
            eval_loader = None

    else:
        if os.path.isdir(train_dir):
            train_loader = SingleSubjectDataloader(
                mode="train",
                in_dir=train_dir,
                label_dict=label_dict,
                data_augmentation=True,
            )
        else:
            train_loader = None

        if os.path.isdir(test_dir):
            test_loader = SingleSubjectDataloader(
                mode="evaluate",
                in_dir=test_dir,
                label_dict=label_dict,
                data_augmentation=False,
            )
        else:
            test_loader = None

        if os.path.isdir(eval_dir):
            eval_loader = SingleSubjectDataloader(
                mode="evaluate",
                in_dir=eval_dir,
                label_dict=label_dict,
                data_augmentation=False,
            )
        else:
            eval_loader = None

    if train_loader is not None:
        train_dataloader = DataLoader(
            train_loader,
            batch_size=batch_size,
            shuffle=True,
            num_workers=os.cpu_count() // 4,  # type: ignore
            collate_fn=lambda x: torch.utils.data.dataloader.default_collate(x),
        )
    else:
        train_dataloader = None

    if test_loader is not None:
        test_dataloader = DataLoader(
            test_loader,
            batch_size=batch_size,
            num_workers=os.cpu_count() // 4,  # type: ignore
            collate_fn=lambda x: torch.utils.data.dataloader.default_collate(x),
        )
    else:
        test_dataloader = None

    if eval_loader is not None:
        eval_dataloader = DataLoader(
            eval_loader,
            batch_size=1,
            num_workers=os.cpu_count() // 4,  # type: ignore
            collate_fn=lambda x: torch.utils.data.dataloader.default_collate(x),
        )
    else:
        eval_dataloader = None

    # print(
    # f"Generated {len(train_dataloader)}, {len(test_dataloader)}, {len(eval_dataloader)}"
    # )
    return train_dataloader, test_dataloader, eval_dataloader
