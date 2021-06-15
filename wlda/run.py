import os
import sys
import shutil
import pickle
import logging

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset

from .args import (
    MyArgumentParser,
    ModelArguments,
    AdvTrainingArguments
)
from .record import TrainRecorder
from .compute_op import Unsupervised


def main():
    """
    5 Steps
    (1) Read arguments from json or shell script
    (2) Check and Prepare arguments for experiments
    (3) Download or Caching Data for experiments
        If Statistical Methodology, cache the features
        else, tokenize input text(sentences or documents)
    (4) Define `Trainer`, `Model` for experiments
    (5) Run experiments or Execute test/inference script
    """

    # ==================================================================
    # Step (1), Read arguments from json or shell script
    # ==================================================================
    parser = MyArgumentParser((ModelArguments, AdvTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # read args from json file
        model_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        # read args from shell script or real arguments
        model_args, training_args = parser.parse_args_into_dataclasses()
    
    # ==================================================================
    # Step (2), Check and Prepare arguments for experiments
    # ==================================================================
    assert model_args.latent_noise >= 0 and model_args.latent_noise <= 1

    # ==================================================================
    # Step (3), Download or Caching Data for experiment
    #           If statistical methodology,
    #              (such as Count or TF-IDF Vectorize)
    #           then cache the features
    # ==================================================================

    # @TODO ag_news / dbpedia / lda_synthetic / nyt / 20 news / yelp polarity
    data = load_dataset(
        path="wikitext.py", 
        name="wikitext-103-v1",
        cache_dir="data/wikitext/"
    )

    return data
    # Specify the file locations
    train_path = './data/wikitext/features/wikitext-103_tra.csr.npz'
    test_path = './data/wikitext/features/wikitext-103_test.csr.npz'
    vocab_path = './data/wikitext/features/vocab.txt'

    # Load train
    train_csr = sp.load_npz(train_path)
    train = np.array(train_csr.todense()).astype("float32")

    # Load Test
    test_csr = sp.load_npz(test_path)
    test = np.array(test_csr.todense()).astype("float32")

    # Load vocab
    ENCODING = "ISO-8859-1"
    # ENCODING = "utf-8"
    with open(vocab_path, encoding=ENCODING) as f:
        vocab_list = [line.strip("\n") for line in f]

    # Construct maps
    vocab2vec = dict(zip(vocab_list, range(len(vocab_list))))
    vec2vocab = {v: vocab for vocab, v in vocab2vec.items()}

    # data   : "train" "valid" "test" "train_with_labels" "valid_with_labels" "test_with_labels"
    #           train   None    test   None                None                None
    # labels : "train_label" "valid_label" "test_label"
    #           None          None          None
    # maps   : "vocab2dim" "dim2vocab" "topic2dim" "dim2topic"
    #           vocab2dim   dim2vocab   None        None

    mean_length = np.mean(np.sum(train, axis=1))
    vocab_size = train.shape[1]
    
    # Multiplier of the reconstruction loss when combined with mmd loss
    if training_args.recon_alpha < 0:
        training_args.recon_alpha = 1.0 / (mean_length + np.log(vocab_size))

    # Hybridize? Declaritive / Imperative?
    from .models.dirichlet import WassersteinAutoEncoder
    wae = WassersteinAutoEncoder(model_args)
    
    N_train = train.shape[0]

    epochs = training_args.num_train_epochs
    verbose = training_args.verbose
    batch_size = training_args.train_batch_size

    train_record = TrainRecoder()
    eval_record = EvalRecorder()

    total_iterations_train = N_train // batch_size

    trainer = UnsupervisedTrainer(trainig_args)
    print(trainer)


if __name__ == "__main__":
    main()
