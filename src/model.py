import datetime
import sys
from logging import getLogger

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation, Interaction,
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
from tqdm import tqdm


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
        recall_n = self.rc_cfg["topk"]
        if type(recall_n) == list:
            recall_n = recall_n[0]
        dataset = create_dataset(self.rc_cfg)

        # device 설정
        device = self.rc_cfg['device']

        # user, item id -> token 변환 array
        user_id = self.rc_cfg['USER_ID_FIELD']
        item_id = self.rc_cfg['ITEM_ID_FIELD']
        user_id2token = dataset.field2id_token[user_id]
        item_id2token = dataset.field2id_token[item_id]

        # user id list
        all_user_list = torch.arange(1, len(user_id2token)).view(-1, 128)

        # user, item 길이
        # user_len = len(user_id2token)
        item_len = len(item_id2token)

        # user-item sparse matrix
        matrix = dataset.inter_matrix(form='csr')

        # user id, predict item id 저장 변수
        pred_list = None
        user_list = None

        # model 평가모드 전환
        self.model.eval()

        # progress bar 설정
        tbar = tqdm(all_user_list, desc=set_color(f"Inference", 'pink'))

        for data in tbar:
            # interaction 생성
            interaction = dict()
            interaction = Interaction(interaction)
            interaction[user_id] = data
            interaction = interaction.to(device)

            # user item별 score 예측
            score = self.model.full_sort_predict(interaction)
            score = score.view(-1, item_len)

            rating_pred = score.cpu().data.numpy().copy()

            user_index = data.numpy()

            idx = matrix[user_index].toarray() > 0

            rating_pred[idx] = -np.inf
            rating_pred[:, 0] = -np.inf
            ind = np.argpartition(rating_pred, -recall_n)[:, -recall_n:]

            arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

            arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

            batch_pred_list = ind[
                np.arange(len(rating_pred))[:, None], arr_ind_argsort
            ]

            if pred_list is None:
                pred_list = batch_pred_list
                user_list = user_index
            else:
                pred_list = np.append(pred_list, batch_pred_list, axis=0)
                user_list = np.append(
                    user_list, user_index, axis=0
                )

        result = []
        for user, pred in zip(user_list, pred_list):
            for item in pred:
                result.append((int(user_id2token[user]), int(item_id2token[item])))

        # 데이터 저장
        dataframe = pd.DataFrame(result, columns=["user", "item"])
        dataframe.to_csv(
            f"./saved/submission-{self.rc_cfg['model']}-{datetime.datetime.now().timestamp()}.csv", index=False
        )
        print('inference done!')

    def _valid(self):
        pass
