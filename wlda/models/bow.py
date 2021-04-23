import re
import numbers
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable

import numpy as np

from utils.processor import Processor

class BaseEstimator:

    @classmethod
    def _get_param_names(cls):
        pass

    def get_params(self):
        pass

    def set_params(self):
        pass

    def __repr__(self):
        pass

    def __getstate__(self):
        pass

    def __setstate__(self):
        pass

    def _check_n_features(self):
        pass

    def _validate_data(self):
        pass


class CountVectorizer(BaseEstimator):
    """ Scikit-learn's feature_extraction.text.CountVectorizer class """

    def __init__(
        self, 
        preprocessor: Callable = None, 
        tokenizer: Callable = None, 
        ngram_range: Tuple[int] = (1, 1),
        lowercase: bool = True,
        stop_words: Optional[frozenset, List[str]] = None,
        max_df: float = 1.0,
        min_df: int = 1,
        max_features: Optional[int] = None,
    ):
        assert ngram_range[0] <= ngram_range[1],
            "Requires ngram_range=(min_n, max_n) such that min_n <= max_n."
        self.processor = Processor(
            preprocess=preprocess,
            tokenizer=tokenizer,
            ngram_range=ngram_range,
            lowercase=lowercase,
            stop_words=stop_words
        )
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.min_df = min_df
        if max_df < 0 or min_df < 0:
            raise ValueError("negative value for max_df or min_df")
        self.max_features = max_features
        if max_features is not None:
            if (not isinstance(max_features, numbers.Integral) or
                    max_features <= 0):
                raise ValueError(
                    "max_features=%r, neither a positive integer nor None"
                    % max_features)

    @property
    def stop_words(self):
        return self.processor.stop_words

    @property
    def preprocesser(self):
        return self.processor.preprocessor

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def ngram(self):
        return self.processor.ngram

    def _sort_features(self):
        pass

    def _limit_features(self):
        pass

    def _count_vocab(self):
        pass

    def fit(self):
        pass

    def fit_transform(self):
        pass

    def transform(self):
        pass

    def get_feature_names(self):
        pass


vectorizer = CountVectorizer(
    input='content', analyzer='word', stop_words='english',
    max_df=0.8, min_df=3, max_features=20000)

print(vectorizer.__dict__)
