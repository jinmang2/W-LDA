import re
from nltk.stem import WordNetLemmatizer
from typing import Callable, Optional, List, Tuple
from _stop_words import ENGLISH_STOP_WORDS


class LemmaTokenizer:

    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.token_pattern = re.compile(r"(?u)\b\w\w+\b")

    def __call__(self, doc):
        return [
            self.wnl.lemmatize(t) for t in doc.split()
            if (len(t) >= 2 and
                re.match("[a-z].*", t) and
                re.match(self.token_pattern, t))
        ]


class Processor:
    """
    Pipeline(
        preprocessor  : Chain together an optional series of text preprocessing
                        steps to apply to a document/sentence
        tokenizer     : Tokenize document/sentence to tokens(char or word)
                        % Does not support huggingface/tokenizers yet...
        ngram         : Make <=N-Gram features
    )
    """

    def __init__(
        self,
        preprocessor: Callable = None,
        tokenizer: Callable = None,
        ngram: Callable = None,
        stop_words: Optional[frozenset, List[str]] = None,
        ngram_range: Tuple[int] = (1,1),
        lowercase: bool = True,
    ):
        self.preprocessor = preprocessor
        if tokenizer is None:
            self.tokenizer = LemmaTokenizer()
        self.ngram = ngram if ngram is not None else self._word_ngrams
        if stop_words is None:
            self.stop_words = ENGLISH_STOP_WORDS
        else:
            self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.lowercase = lowercase

    def __call__(self, doc: str):
        if self.lowercase:
            doc = doc.lower()
        if self.preprocessor is not None:
            doc = self.preprocessor(doc)
        tokens = self.tokenizer(doc)        
        if self.stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]
        tokens = self._word_ngrams(tokens)
        return tokens

    def _word_ngrams(self, tokens):
        """Turn tokens into a sequence of n-grams after stop words filtering"""
        # handle token n-grams
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # no need to do any slicing for unigrams
                # just iterate through the original tokens
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append
            space_join = " ".join

            for n in range(min_n,
                        min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i: i + n]))

        return tokens
