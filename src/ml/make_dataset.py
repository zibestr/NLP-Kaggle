from pickle import dump
from dataclasses import dataclass
from pandas import read_csv
from src.ml.transformer import make_transformer


@dataclass
class Dataset:
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y


def save_dataset(train: bool, name: str, train_ratio: float = 0.8) -> None:
    dataframe = read_csv('data/bbc_data.csv')[:]
    dataframe = dataframe.sample(frac=1)
    train_len = int(len(dataframe) * train_ratio)
    transformer = make_transformer()

    X, y = transformer.fit_transform(dataframe)
    if train:
        dataset = Dataset(X[:train_len], y[:train_len])
    else:
        dataset = Dataset(X[train_len:], y[train_len:])

    with open(f'data/{name}.data', 'wb') as file:
        dump(dataset, file)
