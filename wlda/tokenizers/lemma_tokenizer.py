import re
from nltk.stem import WordNetLemmatizer
from .tokenizer import Tokens, Tokenizer


class LemmaTokenizer(Tokenizer):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.annotators = set()

    def __call__(self, doc):
        return [
            self.wnl.lemmatize(t) for t in doc.split() 
            if (len(t) >= 2 and 
                re.match("[a-z].*", t) and 
                re.match(re.compile(r"(?u)\b\w\w+\b"), t))
        ]
    
    def tokenize(self, text):
        data = self(text)
        return Tokens(data, self.annotators)