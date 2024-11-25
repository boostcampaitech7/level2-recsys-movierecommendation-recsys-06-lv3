from recbole.quick_start import run_recbole


class Model(object):
    def __init__(self, recbole_cfg_path, save_path: str, mode="train"):
        self.cfg_path = recbole_cfg_path
        self.save_path = save_path
        self._train()
        if mode.lower() == "valid":
            self._valid()
        self._inference()
        self._saved_model()

    def _saved_model(self):
        pass

    def _train(self):
        run_recbole(config_file_list=[self.cfg_path])

    def _inference(self):
        pass

    def _valid(self):
        pass
