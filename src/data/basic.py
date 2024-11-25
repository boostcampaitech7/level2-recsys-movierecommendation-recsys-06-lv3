from src.data.DataInterface import DataInterface


class DataBasic(DataInterface):
    def __init__(self, data_path):
        super(DataBasic, self).__init__(data_path)
        return

    def pre_processing(self) -> None:
        raise NotImplemented("구현되지 않은 메서드입니다.")

    def save_data(self) -> None:
        raise NotImplemented("구현되지 않은 메서드입니다.")
