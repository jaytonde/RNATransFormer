import yaml
import torch
import logging
from rna_model import RNAModel

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries
    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Loading model configurations")
    config = load_config_from_yaml("configs.yaml")
    logger.info("Model configurations loaded successfully")

    logger.info("Loading model...")
    model  = RNAModel(logger, config)
    logger.info("Model loaded successfully")

    logger.info("Inference the model for sample input")
    x         = torch.ones(1,6).long()
    mask      = torch.ones(1,6).long()
    output    = model(x,src_mask=mask)
    logger.info(f"Model response : {output}")
    logger.info("Model Inference completed.")