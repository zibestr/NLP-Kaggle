from pickle import load
from string import punctuation

import torchtext.transforms as T
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from torch import int64, nn
from torchtext.vocab import vocab

download('punkt')
download('stopwords')


class TextNormalizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._tokenizer = word_tokenize
        self._stopwords = stopwords.words('english')
        self._punctuation = punctuation
        self._stemmer = SnowballStemmer('english')
        self._lemmatizer = WordNetLemmatizer()

    def forward(self, text: str) -> list[str]:
        text = ''.join([char for char in text.lower()
                        if char not in self._punctuation])
        tokens = [word for word in self._tokenizer(text)
                  if word not in self._punctuation]
        stemmed = [self._stemmer.stem(word) for word in tokens]
        lemmated = [self._lemmatizer.lemmatize(word) for word in stemmed]
        return lemmated


def make_transformer():
    max_len = 2969
    with open('data/vocab.data', 'rb') as file:
        vocabulary = vocab(load(file))
    print('Vocabulary length:', len(vocabulary))

    return T.Sequential(
        TextNormalizer(),
        T.VocabTransform(vocabulary),
        T.Truncate(max_len),
        T.ToTensor(dtype=int64),
        T.PadTransform(max_length=max_len, pad_value=0)
    )
