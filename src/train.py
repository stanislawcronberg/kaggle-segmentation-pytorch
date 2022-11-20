import albumentations as A
import hydra
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from datasets import ClothesDataset
from models.clothes_model import ClothesSegmentationModel


@hydra.main(config_path="../conf", config_name="config", version_base="1.2.0")
def train(cfg):

    transforms = A.Compose([A.Resize(*cfg.data.image_size)])

    train_dataset = ClothesDataset(
        index_path=cfg.data.index.train,
        transforms=transforms,
        preprocessing=smp.encoders.get_preprocessing_fn(cfg.model_info.encoder_name, cfg.model_info.encoder_weights),
    )
    val_dataset = ClothesDataset(
        index_path=cfg.data.index.val,
        transforms=transforms,
        preprocessing=smp.encoders.get_preprocessing_fn(cfg.model_info.encoder_name, cfg.model_info.encoder_weights),
    )

    train_loader = DataLoader(train_dataset, **cfg.training.dataloader_kwargs)
    val_loader = DataLoader(val_dataset, **cfg.eval.dataloader_kwargs)

    model = ClothesSegmentationModel(cfg)

    trainer = pl.Trainer(**cfg.training.trainer_kwargs)

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()
    