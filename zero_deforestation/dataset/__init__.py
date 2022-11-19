import os
import pathlib

import numpy as np
import cv2
import pandas as pd
from torch.utils.data import Dataset
import torch

from zero_deforestation import data


class ZeroDeforestationDataset(Dataset):
    """Deforestation dataset."""

    def __init__(
        self,
        csv_file,
        image_size=(332, 332),
        transform=None,
        return_label=True,
        debug=False,
    ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.image_size = tuple(image_size)
        self.transform = transform
        self.return_label = return_label
        self.debug = debug

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = pathlib.Path(data.__file__).parent / self.df.loc[idx, "example_path"]
        image = cv2.imread(str(img_name))
        image = cv2.resize(image, self.image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sample = {"image": image}
        if self.transform:
            sample["image"] = self.transform(sample["image"])

        if self.return_label:
            label = self.df.loc[idx, "label"]
            sample["target"] = int(label)
            return sample
        else:
            return sample
