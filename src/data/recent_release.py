import os
import pandas as pd

from src.data.sasrecf import SasRecFData

class RecentRelease(SasRecFData):
    def __init__(self, dataset: str, data_path, export_path):
        super(RecentRelease, self).__init__(dataset, data_path, export_path)

    def pre_processing(self) -> None:
        """
        sasrecf.py 실행 결과 업데이트되는 export_dfs에서 inter feature를 업데이트합니다.
        업데이트 내용은 다음과 같습니다.
        {dataset} = {model}.yaml에 지정된 dataset 명
        {dataset}.inter -> user와 item의 연관 데이터(timestamp, review, recent_release 등)
        """
        super().pre_processing()
        
        inter_df = self.export_dfs[f"{self.dataset}.inter"]
        item_df = self.export_dfs[f"{self.dataset}.item"]

        inter_df['timestamp:float'] = pd.to_datetime(inter_df['timestamp:float'], unit='s')
        item_df['year:float'] = pd.to_datetime(item_df['year:float'], format='%Y', errors='coerce')
        item_df['year_end'] = item_df['year:float'] + pd.offsets.YearEnd(0)

        item_df = (
            item_df.merge(
                inter_df.groupby('item_id:token', as_index=False)['timestamp:float'].min()
                .rename(columns={'timestamp:float': 'first_rated'}),
                on='item_id:token', how='left'
            )
        )

        item_df['year_diff'] = item_df['first_rated'].dt.year - item_df['year:float'].dt.year

        second_rated = (
            inter_df[inter_df['timestamp:float'] > inter_df.groupby('item_id:token')['timestamp:float'].transform('min')]
            .groupby('item_id:token')['timestamp:float'].min()
            .reset_index(name='second_rated')
        )

        item_df = item_df.merge(second_rated, on='item_id:token', how='left')
        item_df.loc[item_df['year_diff'] < 0, 'first_rated'] = item_df.loc[item_df['year_diff'] < 0, 'second_rated']

        item_year_map = item_df.set_index('item_id:token')['year:float']
        item_year_end_map = item_df.set_index('item_id:token')['year_end']
        item_time_map = item_df.set_index('item_id:token')['first_rated']

        inter_df['item_year'] = inter_df['item_id:token'].map(item_year_map)
        inter_df['item_year_end'] = inter_df['item_id:token'].map(item_year_end_map)
        inter_df['item_time'] = inter_df['item_id:token'].map(item_time_map)

        inter_df['recent_release:float'] = (
            (
                (inter_df['item_year'].dt.year - inter_df['timestamp:float'].dt.year == 1) &
                (abs((inter_df['timestamp:float'] - inter_df['item_year_end']).dt.days) <= 90)
            ) |
            (
                (inter_df['item_year'].dt.year - inter_df['timestamp:float'].dt.year == 0) &
                (abs((inter_df['timestamp:float'] - inter_df['item_time']).dt.days) <= 90)
            )
        ).astype(int)

        inter = self.change_inter_colname(inter_df)
        self.export_dfs.update({f"{self.dataset}.inter": inter})

    @staticmethod
    def change_inter_colname(inter: pd.DataFrame) -> pd.DataFrame:
        inter.columns = ["user_id:token", "item_id:token", "timestamp:float", "recent_release:float"]
        print(inter.columns)
        return inter

    @staticmethod
    def change_user_colname(user: pd.DataFrame) -> pd.DataFrame:
        user.columns = ["item_id:token", "title:token_seq", "year:float", "genre:token_seq", "director:token_seq", "writer:token_seq", "num_reviews:float"]
        return user

    def save_data(self) -> None:
        os.makedirs(self.export_path, exist_ok=True)
        os.makedirs(os.path.join(self.export_path, self.dataset), exist_ok=True)

        for key in self.export_dfs:
            print(f"{key} is export file in {self.export_path}")
            print(f"{key} column list is {self.export_dfs[key].columns} - shape({self.export_dfs[key].shape})")
            self.export_dfs[key].to_csv(os.path.join(self.export_path, self.dataset, key), index=False, sep="\t")
