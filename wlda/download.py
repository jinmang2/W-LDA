import os
import shutil
import re
import time
import logging

import nltk
from nltk.stem import WordNetLemmatizer

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import scipy.sparse as sparse


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self._token_pattern = re.compile(r"(?u)\b\w\w+\b")

    @property
    def token_pattern(self):
        return self._token_pattern

    @token_pattern.setter
    def token_pattern(self, s):
        if isinstance(s, str):
            self._token_pattern = re.compile(s)
        elif isinstance(s, type(re.compile(""))):
            self._token_pattern = s
        else:
            raise AttributeError

    def __call__(self, doc):
        return [
            self.wnl.lemmatize(t) for t in doc.split() 
            if (len(t) >= 2 and 
                re.match("[a-z].*", t) and 
                re.match(self.token_pattern, t))
        ]


class DataManager:

    def __init__(self, dataset, data_dir="data", rmtree=False):
        self.cwd = os.getcwd()
        self.dataset = dataset
        self.data_dir = os.path.join(self.cwd, data_dir)
        self.rmtree = rmtree

    def download(self, website, saveto=None):
        if saveto is None:
            saveto = os.getcwd()
        os.system(f"curl -O {website}")
        out_file = website.split("/")[-1]
        if os.name == "nt":
            os.system("powershell.exe Expand-Archive "
                      f"-LiteralPath {out_file} "
                      f"-DestinationPath {saveto}")
        else:
            os.system(f"unzip {out_file} -d {saveto}")
         
    def __enter__(self):
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        data_dir = os.path.join(self.data_dir, self.dataset)
        if self.rmtree and os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        os.chdir(data_dir)
        return self

    def __exit__(self, type, value, traceback):
        os.chdir(self.cwd)


class Wikitext103DataManager(DataManager):

    def script_to_docs(self, input_dir, token_file):
        res = []
        filename = os.path.join(input_dir, token_file)
        with open(filename, mode="r", encoding="utf-8") as f:
            for l in f:
                line = l.strip()
                if self.is_document_start(line):
                    res.append(line)
                elif line:
                    res[-1] = res[-1] + " " + line
        return res

    @staticmethod
    def is_document_start(line):
        if len(line) < 4:
            return False
        if line[0] is '=' and line[-1] is '=':
            if line[2] is not '=':
                return True
            else:
                return False
        else:
            return False


class ScikitProcessor:

    def __init__(self, vectorizer, **kwargs):
        super().__init__()
        self.vectorizer = vectorizer

    def get_vocab(self):
        return self.vectorizer.get_feature_names()

    def save_vocab(self, save_directory="", filename="vocab.txt"):
        vocab_filename = os.path.join(save_directory, filename)
        with open(vocab_filename, 'w', encoding='utf-8') as f:
            for item in self.get_vocab():
                f.write(item+'\n')

    def save_npz(self, filename, vectors):
        sparse.save_npz(filename, vectors)

    def vectorize(self, x, mode="train"):
        if mode == "train":
            func = self.vectorizer.fit_transform
        else:
            func = self.vectorizer.transform
        return func(x)

    def shuffle(self, vectors):
        idx = np.arange(vectors.shape[0])
        np.random.shuffle(idx)
        vectors = vectors[idx]
        vectos = sparse.csr_matrix(vectors, dtype=np.float32)
        return vectors



if __name__ == "__main__":

    # Log Settting
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    nltk.download('wordnet')

    is_download = False
    rmtree = False
    dataset = "wikitext-103"
    website = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"

    processor = ScikitProcessor(
        vectorizer=CountVectorizer(
            input='content', analyzer='word', stop_words='english',
            tokenizer=LemmaTokenizer(),
            max_df=0.8, min_df=3, max_features=20000)
    )

    with Wikitext103DataManager(
        data_dir="data", dataset=dataset, rmtree=rmtree
    ) as dl_manager:
        # Enter to ./data/{dataset}

        if is_download:
            dl_manager.download(website)
        
        # Move to ./data/{dataset}/{dataset}
        os.chdir(os.path.join(os.getcwd(), dataset))

        logging.info(
            "Lemmatizing and counting, this may take a few minutes...")

        for mode in ["train", "valid", "test"]:
            # Script to collection of documents
            docs = dl_manager.script_to_docs(os.getcwd(), f"wiki.{mode}.tokens")
            # Lemmitize and Count
            vectors = processor.vectorize(docs, mode=mode)
            logging.info(f"{mode}_vec.shape: {vectors.shape}")
            # Save vocab
            processor.save_vocab()
            # Shuffle the vectors
            vectors = processor.shuffle(vectors)
            # Save vectors
            processor.save_npz(f"{dataset}_{mode}.csr.npz", vectors)
        logging.info("Done!")
