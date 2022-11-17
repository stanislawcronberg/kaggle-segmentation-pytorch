import torch

import segmentation_models_pytorch as smp
import pytorch_lightning as pl


class ClothesSegmentationModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.model = self.__initialize_model()
        self.criterion = smp.utils.losses.DiceLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.training.learning_rate)
        return optimizer

    def __initialize_model(self):
        """Initialize model with pretrained weights with config parameters."""
        model = smp.Unet(**self.cfg.model.model_kwargs)

        return model
