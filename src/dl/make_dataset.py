import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.dl.transformer import make_transformer


class TextData(Dataset):
    def __init__(self, dataframe: pd.DataFrame,
                 device: str = 'cpu',
                 train: bool = True):
        self._labels = {key: value for value, key in
                        enumerate(dataframe['labels'].unique())}
        self._classes = len(dataframe['labels'].unique())
        if train:
            self._df = dataframe[:int(len(dataframe) * 0.8)]
        else:
            self._df = dataframe[int(len(dataframe) * 0.8):]
        transformer = make_transformer()
        self._text = [transformer(text) for text in self._df['data'].to_list()]
        self._device = device

    def __len__(self):
        return len(self._df)

    def _label_binarize(self, emotion: str) -> torch.Tensor:
        label = np.zeros(self._classes)
        label[self._labels[emotion]] = 1
        return torch.tensor(label,
                            dtype=torch.float32, device=self._device)

    def __getitem__(self, ind: int) -> tuple[torch.Tensor,
                                             torch.Tensor]:
        return (self._text[ind].to(device=self._device),
                self._label_binarize(self._df['labels'].iloc[ind]))


def save_dataset(train: bool, device: str, name: str) -> None:
    dataframe = pd.read_csv('data/bbc_data.csv')
    dataframe['tokens_len'] = dataframe['data'].apply(
        lambda text: len(text.split()))

    max_len = 2969
    dataframe = dataframe.loc[dataframe['tokens_len'] <= max_len]
    dataframe = dataframe.drop(columns=['tokens_len'])

    dataset = TextData(dataframe, device, train)

    torch.save(dataset, f'data/{name}.data')
