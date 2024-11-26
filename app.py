from omegaconf import OmegaConf

import src.data as data_module
from src.data.DataInterface import DataInterface
from src.model import Model


def app():
    cfg = OmegaConf.load("./config/config.yaml")

    model_type: str = cfg["model"]["type"]
    recbole_cfg_path = f"./config/model_configs/{model_type.lower()}.yaml"
    model_cfg = OmegaConf.load(recbole_cfg_path)
    ## Data Pre-Processing & Export data(for recbole)
    print("---------Start Data Pre-Processing---------")
    data_type: str = cfg["data"]["type"]
    data_path = cfg["data"]["base_path"]
    _: DataInterface = getattr(data_module, data_type)(model_cfg["dataset"], data_path, model_cfg["data_path"])

    ## (Train or Train/Valid) & Inference
    mode = cfg["model"]["mode"]
    print(f"-------Start Train{mode}------------")
    Model(cfg)

    return


if __name__ == "__main__":
    app()
