from pathlib import Path

import albumentations as A
import cv2
import pandas as pd

from torch.utils.data import Dataset

from segmentation_models_pytorch.encoders import get_preprocessing_fn


class ClothesDataset(Dataset):
    def __init__(self, index_path, transforms: A.Compose = None, preprocessing=None):
        super().__init__()
        self.index = pd.read_csv(Path(index_path))
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.image_mask_paths: list[tuple] = list(zip(self.index["image_path"], self.index["mask_path"]))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        image_path, mask_path = self.image_mask_paths[index]

        # Read image and mask
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Apply pretrained model specific preprocessing
        # TODO: Test if we can do this before augs
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask


if __name__ == "__main__":
    model = ClothesDataset(index_path="data/index/train.csv")
    print(model[0][:10])
