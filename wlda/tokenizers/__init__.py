from .tokenizer import Tokenizer, Tokens
from .lemma_tokenizer import LemmaTokenizer


def get_class(name):
    if name == "lemma":
        return LemmaTokenizer

    raise RuntimeError('Invalid tokenizer: %s' % name)
