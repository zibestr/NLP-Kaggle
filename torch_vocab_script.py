from collections import Counter, OrderedDict
from pickle import dump

import pandas as pd

from src.dl.transformer import TextNormalizer


class Vocab:
    '''
    ### Created by Grander78498
    '''
    tokens: list[str]

    def __init__(self, texts: pd.Series) -> None:
        normalizer = TextNormalizer()
        print('Loading tokens...')
        tokens_matrix = [normalizer(text) for text in texts.to_list()]

        flatten_tokens_matrix = []
        for tokens in tokens_matrix:
            flatten_tokens_matrix += tokens
        self.tokens = flatten_tokens_matrix

    def freq_indexing(self, min_freq: int = 1,
                      max_freq: int = -1) -> OrderedDict[str, int]:
        counter = Counter(self.tokens)
        print('Filtered tokens...')
        filtered_tokens = filter(lambda x: min_freq <= x[1] <= max_freq
                                 if max_freq != -1 else min_freq <= x[1],
                                 counter.items())
        print('Sorted tokens...')
        sorted_by_freq_tuples = sorted(filtered_tokens, key=lambda x: x[1],
                                       reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        print('Finish!')
        return ordered_dict

    def generate_dict(self, index_type: str | None = None,
                      min_freq: int = 1,
                      max_freq: int = -1) -> OrderedDict[str, int]:
        match index_type:
            case 'index':
                dict_ = OrderedDict([(key, i)
                                     for i, key in enumerate(self.tokens)])
            case 'freq' | _:
                dict_ = self.freq_indexing(min_freq, max_freq)
        return dict_


if __name__ == '__main__':
    series = pd.read_csv('data/bbc_data.csv')['data']
    vocab = Vocab(series)
    vocabulary = vocab.generate_dict(index_type='freq')

    with open('data/vocab.data', 'wb') as file:
        dump(vocabulary, file)
