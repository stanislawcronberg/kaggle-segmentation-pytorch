from pathlib import Path
from typing import Callable

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ClothesDataset(Dataset):
    def __init__(self, index_path, transforms: A.Compose = None, preprocessing: Callable = None):
        super().__init__()
        self.index = pd.read_csv(Path(index_path))
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.image_mask_paths: list[tuple] = list(zip(self.index["image_path"], self.index["mask_path"]))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index: int):
        image_path, mask_path = self.image_mask_paths[index]

        # Read image and mask
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (416, 288))
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.clip(mask, 0, 58)
        mask = cv2.resize(mask, (416, 288))

        # Add channels so that mask has 3 dimensions (1, H, W)
        mask = np.expand_dims(mask, axis=-1)

        # Preprocess image
        if self.preprocessing:
            image = self.preprocessing(image)

        # Apply transforms
        # if self.transforms:
        #     augmented = self.transforms(image=image, mask=mask)
        #     image = augmented["image"]
        #     mask = augmented["mask"]
        image = torch.tensor(image).float()
        mask = torch.tensor(mask).long()

        return image, mask


if __name__ == "__main__":
    model = ClothesDataset(index_path="data/index/train.csv")
    print(model[0][0].shape, model[0][1].shape)
