import os
from abc import ABC, abstractmethod

import pandas as pd


class DataInterface(ABC):
    def __init__(self, dataset, data_path: str, export_path: str):
        assert dataset is not None or dataset != ""
        self.dataset = dataset
        self.data_path = data_path
        self.export_path = export_path
        self.export_dfs: dict[str, pd.DataFrame] = {}
        self.data: dict[str, pd.DataFrame] = {}
        self._load_data()
        self.pre_processing()
        self.save_data()

    def _load_data(self) -> None:
        """
        기본 .tsv & .csv 파일 로딩
        -> self.data:dict[str,pd.DataFrame] 적재
        :return: None
        """
        filelist = os.listdir(self.data_path)
        for file in filelist:
            if file.endswith(".csv"):
                print("Load File:" + file)
                df = pd.read_csv(os.path.join(self.data_path, file), sep=",")
                self.data.update({file.replace(".csv", ""): df})
            elif file.endswith(".tsv"):
                print("Load File:" + file)
                df = pd.read_csv(os.path.join(self.data_path, file), sep="\t")
                self.data.update({file.replace(".tsv", ""): df})

    @abstractmethod
    def pre_processing(self) -> None:
        raise NotImplemented("Not Implemented")

    @abstractmethod
    def save_data(self) -> None:
        raise NotImplemented("Not Implemented")
