import os

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from src.data.DataInterface import DataInterface


class SasRecFData(DataInterface):
    def __init__(self, dataset: str, data_path, export_path):
        super(SasRecFData, self).__init__(dataset, data_path, export_path)

    def pre_processing(self) -> None:
        """
        export_dfs는 save_data에서 저장된 df 목록입니다.
        저장 형식은 아래와 같습니다.
        {dataset} = {model}.yaml에 지정된 dataset 명
        {dataset}.inter -> user와 item의 연관 데이터(timestamp, review, etc..)
        {dataset}.user -> user feature - 유저별 관련 정보들
        {dataset}.item -> item feature - 아이템별 관련 정보들
        {dataset}.kg -> 데이터 별 그래프 관계도(knowledge graph)
        {dataset}.link -> item-entity linkage data (entity, item_id)
        {dataset}.net -> social graph data (source, target)
        :return:None
        """
        item: pd.DataFrame = self.data["titles"]
        # Director & Writer
        director: pd.DataFrame = self.data["directors"]
        writer: pd.DataFrame = self.data["writers"]
        director["director_id"] = director["director"].apply(lambda x: int(x.replace("nm", "")))
        director = director.drop(columns=["director"])
        writer["writer_id"] = writer["writer"].apply(lambda x: int(x.replace("nm", "")))
        writer = writer.drop(columns=["writer"])
        #
        genre: pd.DataFrame = self.data["genres"]
        # genre_id_ls = self._generate_genre_id(genre)
        genre = self._multi_hot_encode_except_user_item(genre, "genre")
        year: pd.DataFrame = self.data["years"]
        # Merge All .item
        item = item.merge(director, how="left", on="item")
        item = item.merge(writer, how="left", on="item")
        item = item.merge(genre, how="left", on="item")
        item = item.merge(year, how="left", on="item")
        item = item.drop(columns=["title"])

        train_ratings: pd.DataFrame = self.data["train_ratings"]

        item = self.change_item_colname(item)
        inter = self.change_inter_colname(train_ratings)
        self.export_dfs.update({f"{self.dataset}.item": item, f"{self.dataset}.inter": inter})

    @staticmethod
    def change_inter_colname(inter: pd.DataFrame) -> pd.DataFrame:
        inter.columns = ["user_id:token", "item_id:token", "timestamp:float"]
        print(inter.columns)
        return inter

    @staticmethod
    def change_item_colname(item: pd.DataFrame) -> pd.DataFrame:
        cols = item.columns
        change_col_ls = []
        for col in cols:
            if col == "item":
                col = f"{col}_id:token"
            elif col not in ["year"]:
                col = f"{col}:token"
            else:
                col = f"{col}:float"
            change_col_ls.append(col)
        item.columns = change_col_ls
        return item

    @staticmethod
    def _multi_hot_encode_except_user_item(df, multi_hot_col: str):
        df = pd.DataFrame(df.groupby("item")[multi_hot_col].apply(list))
        mlb = MultiLabelBinarizer()
        mlb_fit = mlb.fit_transform(df[multi_hot_col])
        df[mlb.classes_] = mlb_fit
        df = df.drop(columns=[multi_hot_col])
        return df

    @staticmethod
    def _generate_genre_id(genre: pd.DataFrame) -> dict[str, int]:
        genre_ls = genre["genre"].unique().tolist()
        genre_dict = {}
        i = 0
        for name in genre_ls:
            genre_dict[name] = i
            i += 1
        return genre_dict

    def save_data(self) -> None:
        os.makedirs(self.export_path, exist_ok=True)
        os.makedirs(os.path.join(self.export_path, self.dataset), exist_ok=True)

        for key in self.export_dfs:
            print(f"{key} is export file in {self.export_path}")
            print(f"{key} column list is {self.export_dfs[key].columns} - shape({self.export_dfs[key].shape})")
            self.export_dfs[key].to_csv(os.path.join(self.export_path, self.dataset, key), index=False, sep="\t")
