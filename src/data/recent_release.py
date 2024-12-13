import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from src.data.DataInterface import DataInterface

class RecentRelease(DataInterface):
    RECENT_RELEASE_THRESHOLD = 90
    YEAR_DIFF_THRESHOLD = 1

    def __init__(self, dataset: str, data_path, export_path):
        super(RecentRelease, self).__init__(dataset, data_path, export_path)

    def pre_processing(self) -> None:
        """
        sasrecf.py에서 수행되는 작업에 새로운 피처를 추가합니다.
        업데이트 내용은 다음과 같습니다.
        {dataset} = {model}.yaml에 지정된 dataset 명
        {dataset}.inter -> user와 item의 연관 데이터(timestamp, review, recent_release 등)
        """

        # 데이터 로드
        director: pd.DataFrame = self.data["directors"]
        writer: pd.DataFrame = self.data["writers"]
        genre: pd.DataFrame = self.data["genres"]
        year: pd.DataFrame = self.data["years"]
        
        # 데이터 처리
        director = self._process_director_or_writer(director, "director")
        writer = self._process_director_or_writer(writer, "writer")
        genre = self._multi_hot_encode_except_user_item(genre, "genre")

        # 아이템 데이터 병합
        items = self._merge_item_data(self.data["items"], [director, writer, genre, year])
        items = self._rename_item_columns(items)
        inter = self._rename_inter_columns(self.data["train_ratings"])
        
        # timestamp 처리 및 recent_release 피처 추가
        inter, items = self._process_interactions(inter, items)
        self.export_dfs.update({
            f"{self.dataset}.item": items,
            f"{self.dataset}.inter": inter
            })

    def _process_director_or_writer(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        director 또는 wirter 데이터의 원본 칼럼에서 id를 추출하여 새로운 칼럼을 생성합니다.
        """
        df[f"{column}_id"] = df[column].apply(lambda x: int(x.replace("nm", "")))
        return df.drop(columns=[column])
    
    def _merge_item_data(self, base_df: pd.DataFrame, dfs: list) -> pd.DataFrame:
        """
        (director, writer, genre, year) 데이터를 item 데이터에 병합합니다.
        """
        for df in dfs:
            base_df = base_df.merge(df, how="left", on="item")
        return base_df.drop(columns=["title"])
    
    def _process_interactions(self, inter: pd.DataFrame, item: pd.DataFrame) -> tuple:
        """
        인터랙션 데이터의 timestamp를 처리하고 recent_release 피처를 추가합니다.
        """
        inter['timestamp:float'] = pd.to_datetime(inter['timestamp:float'], unit='s')
        item['year:float'] = pd.to_datetime(item['year:float'], format='%Y', errors='coerce')
        item['year_end'] = item['year:float'] + pd.offsets.YearEnd(0)

        items = self._calculate_first_rated(inter, item).drop(columns=['first_rated', 'year_end'])
        inter['recent_release:float'] = self._calculate_recent_release(inter, items)

        return inter, items
    
    def _calculate_first_rated(
            self, inter: pd.DataFrame, item: pd.DataFrame
    ) -> pd.DataFrame:
        """
        아이템 데이터에 첫 번째 리뷰와 두 번째 리뷰 정보를 추가
        첫 번째 리뷰가 개봉연도보다 이전인 경우, 두 번째 리뷰를 첫 번째 리뷰로 대체합니다.
        """
        first_rated = (
            inter.groupby('item_id:token')['timestamp:float'].min()
            .rename("first_rated")
        )

        second_rated = (
            inter[inter['timestamp:float'] > first_rated]
            .groupby('item_id:token')['timestamp:float'].min()
            .reset_index(name='second_rated')
        )

        item = item.merge(first_rated, on='item_id:token', how='left')
        item = item.merge(second_rated, on='item_id:token', how='left')
        
        item['year_diff'] = item['first_rated'].dt.year - item['year:float'].dt.year
        item.loc[item['year_diff'] < 0, 'first_rated'] = item.loc[
            item['year_diff'] < 0, 'second_rated'
        ]
        
        return item.drop(columns=['second_rated', 'year_diff'])

    def _calculate_recent_release(self, inter: pd.DataFrame, item: pd.DataFrame) -> pd.Series:
        """
        아이템이 recent release에 해당하는지 계산합니다.
        """
        inter['item_year'] = inter['item_id:token'].map(item.set_index('item_id:token')['year:float'])
        inter['item_year_end'] = inter['item_id:token'].map(item.set_index('item_id:token')['year_end'])
        inter['item_time'] = inter['item_id:token'].map(item.set_index('item_id:token')['first_rated'])

        within_one_year = (inter['item_year'].dt.year - inter['timestamp:float'].dt.year == self.YEAR_DIFF_THRESHOLD)
        within_same_year = (inter['item_year'].dt.year - inter['timestamp:float'].dt.year == 0)
        recent_release = (
            (within_one_year & (inter['timestamp:float'] - inter['item_year_end']).dt.days <= self.RECENT_RELEASE_THRESHOLD) |
            (within_same_year & (inter['timestamp:float'] - inter['item_time']).dt.days <= self.RECENT_RELEASE_THRESHOLD)
        )
        return recent_release.astype(int)

    def _rename_inter_columns(self, inter: pd.DataFrame) -> pd.DataFrame:
        """
        인터랙션 데이터의 칼럼 이름을 변경합니다.
        """
        inter.columns = ["user_id:token", "item_id:token", "timestamp:float"]
        return inter
    
    def _rename_item_columns(self, item: pd.DataFrame) -> pd.DataFrame:
        """
        아이템 데이터의 칼럼 이름을 변경합니다.
        """
        item.columns = [
            "item_id:token", "year:float", "genre:token_seq", 
            "director_id:token", "writer_id:token"
        ]
        return item

    def save_data(self) -> None:
        os.makedirs(self.export_path, exist_ok=True)
        os.makedirs(os.path.join(self.export_path, self.dataset), exist_ok=True)

        for key in self.export_dfs:
            print(f"{key} is export file in {self.export_path}")
            print(f"{key} column list is {self.export_dfs[key].columns} - shape({self.export_dfs[key].shape})")
            self.export_dfs[key].to_csv(os.path.join(self.export_path, self.dataset, key), index=False, sep="\t")
    
    @staticmethod
    def _multi_hot_encode_except_user_item(df, multi_hot_col: str):
        df = pd.DataFrame(df.groupby("item")[multi_hot_col].apply(list))
        mlb = MultiLabelBinarizer()
        mlb_fit = mlb.fit_transform(df[multi_hot_col])
        df[mlb.classes_] = mlb_fit
        df = df.drop(columns=[multi_hot_col])
        return df

