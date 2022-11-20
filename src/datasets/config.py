from dataclasses import dataclass
from typing import Any


@dataclass
class Data:
    index: dict[str, str]
    image_size: tuple[int, int]


@dataclass
class ModelInfo:
    encoder_name: str
    encoder_weights: str
    in_channels: int
    classes: int  # Number of classes


@dataclass
class Training:
    learning_rate: float
    trainer_kwargs: dict[str, Any]
    dataloader_kwargs: dict[str, Any]


@dataclass
class Eval:
    checkpoint_path: str


@dataclass
class SegmentationConfig:
    data: Data
    model: ModelInfo
    training: Training
    eval: Eval
