import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split


class DataProcessor:

    @staticmethod
    def parse_csv_by_attribute(file: str, attribute: str):
        data = pd.read_csv(file)

        x = data.drop(attribute, axis=1)
        y = data[attribute]

        return x, y

    @staticmethod
    def split_by_train_and_test_data(x: DataFrame, y: Series):
        return train_test_split(x, y, test_size=0.2, random_state=42)
