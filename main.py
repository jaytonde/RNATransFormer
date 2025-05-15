import yaml
import logging
form RNAModel import RNAModel

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Loading model configurations")
    config = load_config_from_yaml("configs.yaml")
    logger.info("Model configurations loaded successfully")

    logger.info("Loading model...")
    model  = RNAModel(logger, config)
    logger.info("Model loaded successfully")
