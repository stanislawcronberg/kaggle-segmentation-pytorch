import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch


class ClothesSegmentationModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.model = self.__initialize_model()
        self.criterion = smp.losses.DiceLoss(mode="multiclass")
        preprocessing_params = smp.encoders.get_preprocessing_params(
            self.cfg.model.encoder_backbone, pretrained=self.cfg.model.encoder_weights
        )
        self.register_buffer("std", torch.tensor(preprocessing_params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(preprocessing_params["mean"]).view(1, 3, 1, 1))

        self.__log_params = dict(on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def forward(self, x):
        # TODO: Fix this hack here and use get
        image = (x - self.mean) / self.std
        mask = self.model(image)
        return mask

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, **self.__log_params)
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.training.learning_rate)
        return optimizer

    def _shared_eval_step(self, batch, batch_idx, stage):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log(f"{stage}_loss", loss, **self.__log_params)

    def __initialize_model(self):
        """Initialize model with pretrained weights with config parameters."""
        # model = smp.Unet(**self.cfg.model.model_kwargs)

        model = smp.Unet(
            encoder_name="mobilenet_v2",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=59,  # model output channels (number of classes in your dataset)
        )

        return model


if __name__ == "__main__":
    model = ClothesSegmentationModel()
