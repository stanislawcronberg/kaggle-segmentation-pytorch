import hydra

from datasets import ClothesDataset


@hydra.main(config_path="conf", config_name="config", version_base="1.2.0")
def train(cfg):

    dataset = ClothesDataset(index_path=cfg.dataset.index_path)
