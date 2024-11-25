from omegaconf import OmegaConf

import src.data as data_module
from src.model import Model


def app():
    cfg = OmegaConf.load("./config/config.yaml")
    ## Data Pre-Processing & Export data(for recbole)
    print("---------Start Data Pre-Processing---------")
    data_type: str = cfg["data"]["type"]
    data_path = cfg["data"]["path"]
    data_class = getattr(data_module, data_type)(data_path)

    ## (Train or Train/Valid) & Inference
    mode = cfg["model"]["mode"]
    print(f"-------Start Train{mode}------------")
    model_type: str = cfg["model"]["type"]
    recbole_cfg_path = f"./config/model_configs/{model_type.lower()}.yaml"
    model_save_path = cfg["model"]["save_path"]
    model = Model(recbole_cfg_path, model_save_path, mode)
    return


if __name__ == "__main__":
    app()
