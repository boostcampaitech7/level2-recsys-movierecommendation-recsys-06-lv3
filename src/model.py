import sys
from logging import getLogger

import torch.distributed as dist
from omegaconf import OmegaConf
from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
)
from recbole.data.transform import construct_transform
from recbole.model.abstract_recommender import AbstractRecommender
from recbole.trainer import Trainer
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)


class Model(object):
    def __init__(self, cfg, model=None, saved=True):
        model_type: str = cfg["model"]["type"]
        recbole_cfg_path = f"./config/model_configs/{model_type.lower()}.yaml"
        save_path = cfg["model"]["save_path"]
        mode = cfg["model"]["mode"]
        self.cfg = cfg
        self.rc_cfg = None
        self.rc_cfg_path = recbole_cfg_path
        self.save_path = save_path
        self.result = {}
        self.trainer: Trainer | None = None
        self.saved = saved
        self.model: AbstractRecommender | None = model
        self.dataset = None
        self.config_dict = None
        self.queue = None
        self._set_config()
        self._wandb_init()
        #
        self._train()
        if mode.lower() == "valid":
            self._valid()
        self._inference()
        self._saved_model()
        self._wandb_finish()

    def _set_config(self):
        self.rc_cfg = Config(
            model=self.model,
            dataset=self.dataset,
            config_file_list=[self.rc_cfg_path],
            config_dict=self.config_dict,
        )

    def _wandb_init(self):
        if self.cfg.wandb.use:
            import wandb

            w_cfg = self.cfg.wandb
            wandb.login(key=self.cfg.wandb.api_key)
            # wandb.require("core")
            # https://docs.wandb.ai/ref/python/init 참고
            wandb.init(
                project=w_cfg.project,
                config=OmegaConf.to_container(OmegaConf.load(self.rc_cfg_path), resolve=True),
                name=w_cfg.run_name,
                notes=w_cfg.memo,
                tags=[self.cfg.model.type],
                resume="allow",
            )
            self.cfg.run_href = wandb.run.get_url()

            wandb.run.log_code(
                "./src"
            )  # src 내의 모든 파일을 업로드. Artifacts에서 확인 가능

    def _wandb_finish(self):
        if self.cfg.wandb.use:
            import wandb
            wandb.finish()

    def _saved_model(self):
        pass

    def _train(self):

        init_seed(self.rc_cfg["seed"], self.rc_cfg["reproducibility"])
        # logger initialization
        init_logger(self.rc_cfg)
        logger = getLogger()
        logger.info(sys.argv)
        logger.info(self.rc_cfg)

        # dataset filtering
        dataset = create_dataset(self.rc_cfg)
        logger.info(dataset)

        # dataset splitting
        train_data, valid_data, test_data = data_preparation(self.rc_cfg, dataset)

        # model loading and initialization
        init_seed(self.rc_cfg["seed"] + self.rc_cfg["local_rank"], self.rc_cfg["reproducibility"])
        self.model = get_model(self.rc_cfg["model"])(self.rc_cfg, train_data._dataset).to(self.rc_cfg["device"])
        logger.info(self.model)

        transform = construct_transform(self.rc_cfg)
        flops = get_flops(self.model, dataset, self.rc_cfg["device"], logger, transform)
        logger.info(set_color("FLOPs", "blue") + f": {flops}")

        # trainer loading and initialization
        trainer: Trainer = get_trainer(self.rc_cfg["MODEL_TYPE"], self.rc_cfg["model"])(self.rc_cfg, self.model)

        # model training
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=self.saved, show_progress=self.rc_cfg["show_progress"]
        )
        self.Trainer = trainer
        # model evaluation
        test_result = trainer.evaluate(
            test_data, load_best_model=self.saved, show_progress=self.rc_cfg["show_progress"]
        )

        environment_tb = get_environment(self.rc_cfg)
        logger.info(
            "The running environment of this training is as follows:\n"
            + environment_tb.draw()
        )

        logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
        logger.info(set_color("test result", "yellow") + f": {test_result}")

        result = {
            "best_valid_score": best_valid_score,
            "valid_score_bigger": self.rc_cfg["valid_metric_bigger"],
            "best_valid_result": best_valid_result,
            "test_result": test_result,
        }

        if not self.rc_cfg["single_spec"]:
            dist.destroy_process_group()

        if self.rc_cfg["local_rank"] == 0 and self.queue is not None:
            self.queue.put(result)  # for multiprocessing, e.g., mp.spawn

        # result = run_recbole(config_file_list=[self.rc_cfg_path])
        self.result.update(result)

        # print(result)

    def _inference(self):
        pass

    def _valid(self):
        pass
