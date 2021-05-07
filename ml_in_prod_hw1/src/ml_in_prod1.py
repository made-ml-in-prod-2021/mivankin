import io
import os
import pandas as pd
import pickle

from utils import loggers

from models import callback_build, callback_predict

from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf

@dataclass()
class ConfigParams:
    eval: bool
    dataset_path: str
    dump_model_path: str
    load_model_path: str
    predict_path: str
    save_path: str
    solver: str
    reg: str
    max_iter: int
    seed: int
    sample_size: int


@hydra.main(config_name="train_config")
def main(loaded : DictConfig):
    """Main function in  Heart Disease UCI classification model utility
    """

    os.chdir(hydra.utils.get_original_cwd())

    loggers.setup_logging()

    if loaded.eval == False:
        callback_build(loaded)
    else:
        callback_predict(loaded)


if __name__ == "__main__":
    #Entry point in Heart Disease UCI classification model utility
    main()