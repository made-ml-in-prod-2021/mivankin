"""
ML in production homework 1, main function
Author: MADE DS-22 Ivankin M.
"""
import os
import hydra
from omegaconf import DictConfig
from utils import loggers
from models import callback_build, callback_predict


@hydra.main(config_name="train_config")
def main(loaded: DictConfig):
    """Main function in  Heart Disease UCI classification model utility
    """

    os.chdir(hydra.utils.get_original_cwd())

    loggers.setup_logging()

    if loaded.eval is False:
        callback_build(loaded)
    else:
        callback_predict(loaded)


if __name__ == "__main__":
    #Entry point in Heart Disease UCI classification model utility
    main()
