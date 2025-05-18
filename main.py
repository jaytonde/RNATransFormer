import yaml
import logging
from   RNAModel import RNAModel

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
    x         = torch.ones(4,128).long().cuda()
    mask      = torch.ones(4,128).long().cuda()
    output    = model(x,src_mask=mask)
    logger.info(f"Model response : {output}")
    logger.info("Model Inference completed.")