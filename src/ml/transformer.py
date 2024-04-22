from string import punctuation

from nltk import download
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from numpy import ndarray
from pandas import Series
from scipy.sparse import spmatrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

download('punkt')
download('stopwords')


class TextNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        self._tokenizer = word_tokenize
        self._stopwords = stopwords.words('english')
        self._punctuation = punctuation
        self._stemmer = SnowballStemmer('english')

        self.__fitted_tokens: list[str] = []

    def _normalize(self, text: str) -> str:
        text = ''.join([char for char in text.lower()
                        if char not in self._punctuation])
        tokens = [word for word in self._tokenizer(text)
                  if word not in self._punctuation]
        # return ' '.join(tokens)
        return ' '.join([self._stemmer.stem(word) for word in tokens])

    def fit(self, data: Series) -> TransformerMixin:
        self.__fitted_tokens = [self._normalize(text)
                                for text in data.to_list()]
        return self

    def transform(self, text: str | None = None) -> list[str]:
        return self.__fitted_tokens


class FeaturesTransformer(ColumnTransformer):
    def __init__(self, transformers):
        super().__init__(transformers=transformers)

    def fit_transform(self, X, y=None, **params):
        _, X_transform, X_cols = self.transformers[0]
        _, y_transform, y_cols = self.transformers[1]
        X_transform.fit(X[X_cols])
        y_transform.fit(X[y_cols])
        X_t = X_transform.transform(X[X_cols])
        y_t = y_transform.transform(X[y_cols])

        return X_t, y_t


def make_transformer() -> ColumnTransformer:
    text_transformer = Pipeline([
        ('text_normalization', TextNormalizer()),
        ('vectorize', TfidfVectorizer(
            max_df=0.8,
            min_df=5
        ))
    ])

    column_transform = FeaturesTransformer(
        transformers=[
            ('text_preprocessing', text_transformer, 'data'),
            ('label_encoding', LabelBinarizer(), 'labels')
        ]
    )

    return column_transform
