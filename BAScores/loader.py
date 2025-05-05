import itertools
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchio as tio
from torch.utils.data import DataLoader, Dataset

from BAScores.utils import get_prefix, get_prefix_oasis

validation_transform = tio.Compose(
    [
        tio.Resize((128, 128, 128)),
    ]
)


class SingleSubjectDataloader(Dataset):
    """
    Class for the SingleSubject Dataloader. Returns the input tensor with it's label(as a torch.tensor)

    :param train_mode: True for train dataloader and False otherwise
    :type train_mode: bool
    :param in_dir: The input directory that contains train/ test/ and eval/ folders
    :type in_dir: str
    :param label_dict: A dictionary that maps MRID to target value
    :type label_dict: dict
    :param data_augmentation: True if you want data augmentation to be applied, False otherwise
    :type data_augmentation: bool
    """

    def __init__(
        self,
        train_mode: bool,
        in_dir: str,
        label_dict: dict,
        data_augmentation: bool = False,
    ) -> None:
        super().__init__()

        self.in_dir = Path(in_dir)
        self.data_augmentation = data_augmentation
        self.train_mode = train_mode
        self.label_dict = label_dict
        self.images: list = []
        self.img_label: list = []
        self._fetch_images()
        self._fetch_labels()

    def _fetch_images(self) -> None:
        for img_name in os.listdir(self.in_dir):
            if img_name.endswith(".nii.gz"):
                # subject_prefix = get_prefix_oasis(img_name)
                image_path = self.in_dir / img_name
                self.images.append(image_path)

    def _fetch_labels(self) -> None:
        for img in self.images:
            label = self.label_dict.get(get_prefix(img.name), 0)
            self.img_label.append((img, label))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Any:
        img_path, label = self.img_label[index]
        img = tio.ScalarImage(img_path)
        resize = tio.Resize((128, 128, 128))
        img = resize(img)

        transforms = []
        if self.data_augmentation:
            if self.train_mode:
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
            img = pairwise_transforms(img)

        img_tensor = torch.tensor(img.data, dtype=torch.float32)

        return img_tensor, torch.tensor(label, dtype=torch.float32)


class PairwiseDataloader(Dataset):
    """
    Class for the Pairwise Dataloader. Returns the input tensors with theri labels(as a torch.tensor)

    :param train_mode: True for train dataloader and False otherwise
    :type train_mode: bool
    :param in_dir: The input directory that contains train/ test/ and eval/ folders
    :type in_dir: str
    :param label_dict: A dictionary that maps MRID to target value
    :type label_dict: dict
    :param data_augmentation: True if you want data augmentation to be applied, False otherwise
    :type data_augmentation: bool
    """

    def __init__(
        self,
        train_mode: bool,
        in_dir: str,
        label_dict: dict,
        data_augmentation: bool = False,
    ) -> None:
        super().__init__()
        self.in_dir = Path(in_dir)
        self.data_augmentation = data_augmentation
        self.train_mode = train_mode
        self.label_dict = label_dict
        self.image_groups = self._group_images()
        self.pairs = self._create_pairs()

    def _group_images(self) -> dict:
        image_groups: dict = {}
        for img_name in os.listdir(self.in_dir):
            if img_name.endswith(".nii.gz"):
                subject_prefix = get_prefix_oasis(img_name)
                image_path = self.in_dir / img_name
                if subject_prefix not in image_groups:
                    image_groups[subject_prefix] = []
                image_groups[subject_prefix].append(image_path)

        for key in image_groups:
            image_groups[key].sort()

        return image_groups

    def _create_pairs(self) -> list:
        pairs: list = []
        for subject_images in self.image_groups.values():
            if len(subject_images) > 1:
                subject_pairs = list(itertools.combinations(subject_images, 2))
                for img1, img2 in subject_pairs:
                    if (
                        get_prefix(img1.name) in self.label_dict
                        and get_prefix(img2.name) in self.label_dict
                    ):
                        label1 = self.label_dict[get_prefix(img1.name)]
                        label2 = self.label_dict[get_prefix(img2.name)]
                        pairs.append((img1, img2, label1, label2))

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Any:
        img1_path, img2_path, label1, label2 = self.pairs[index]
        img1 = tio.ScalarImage(img1_path)
        img2 = tio.ScalarImage(img2_path)

        resize = tio.Resize((128, 128, 128))

        img1 = resize(img1)
        img2 = resize(img2)

        transforms = []
        if self.data_augmentation:
            if self.train_mode:
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

        return (
            img1_tensor,
            img2_tensor,
            torch.tensor(label1, dtype=torch.float32),
            torch.tensor(label2, dtype=torch.float32),
        )


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

    labels = pd.read_csv(label_dict_csv)

    label_dict = {row["MRID"]: row[target] for _, row in labels.iterrows()}

    if pairwise:
        if os.path.isdir(train_dir):
            train_loader = PairwiseDataloader(
                train_mode=True,
                in_dir=train_dir,
                label_dict=label_dict,
                data_augmentation=True,
            )
        else:
            train_loader = None

        if os.path.isdir(test_dir):
            test_loader = PairwiseDataloader(
                train_mode=False,
                in_dir=test_dir,
                label_dict=label_dict,
                data_augmentation=False,
            )
        else:
            test_loader = None

        if os.path.isdir(eval_dir):
            eval_loader = PairwiseDataloader(
                train_mode=False,
                in_dir=eval_dir,
                label_dict=label_dict,
                data_augmentation=False,
            )
        else:
            eval_loader = None

    else:
        if os.path.isdir(train_dir):
            train_loader = SingleSubjectDataloader(
                train_mode=True,
                in_dir=train_dir,
                label_dict=label_dict,
                data_augmentation=True,
            )
        else:
            train_loader = None

        if os.path.isdir(test_dir):
            test_loader = SingleSubjectDataloader(
                train_mode=False,
                in_dir=test_dir,
                label_dict=label_dict,
                data_augmentation=False,
            )
        else:
            test_loader = None

        if os.path.isdir(eval_dir):
            eval_loader = SingleSubjectDataloader(
                train_mode=False,
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
            num_workers=os.cpu_count() // 2,  # type: ignore
            collate_fn=lambda x: torch.utils.data.dataloader.default_collate(x),
        )
    else:
        train_dataloader = None

    if test_loader is not None:
        test_dataloader = DataLoader(
            test_loader,
            batch_size=batch_size,
            num_workers=os.cpu_count() // 2,  # type: ignore
            collate_fn=lambda x: torch.utils.data.dataloader.default_collate(x),
        )
    else:
        test_dataloader = None

    if eval_loader is not None:
        eval_dataloader = DataLoader(
            eval_loader,
            batch_size=1,
            num_workers=os.cpu_count() // 2,  # type: ignore
            collate_fn=lambda x: torch.utils.data.dataloader.default_collate(x),
        )
    else:
        eval_dataloader = None

    return train_dataloader, test_dataloader, eval_dataloader
