# get_topic_words_decoder_weights
# calc_topic_uniqueness
# request_pmi
# print_topics

import torch
import torch.nn as nn


NON_LINEARITY = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "hardswish": nn.Hardswish,
    "elu": nn.ELU,
    "celu": nn.CELU,
    "selu": nn.SELU,
    "glu": nn.GLU,
    "gelu": nn.GELU,
    "hardshrink": nn.Hardshrink,
    "leakyrelu": nn.LeakyReLU,
    "logsigmoid": nn.LogSigmoid,
    "softplus": nn.Softplus,
    "softshrink": nn.Softshrink,
    "prelu": nn.PReLU,
    "softsign": nn.Softsign,
    "tanhshrink": nn.Tanhshrink,
    "softmin": nn.Softmin,
    "softmax": nn.Softmax,
    "logsoftmax": nn.LogSoftmax,
}


def get_topic_words_decoder_weights(D, data, ctx, k=10, decoder_weights=False):
    if decoder_weights:
        params = D.weight.data # topic X number of vocabs
    else:
        y = D.y_as_topics()
        params = D(y)
    top_word_ids = torch.argsort(params, dim=-1, descending=True)[:,:k]
    # @TODO! tokenizer가 필요함! 혹은 tokens to ids method
    top_word_strings = top_word_ids
    return top_word_strings


def calc_topic_uniqueness(top_words_idx_all_topics):
    """
    This function calculates topic uniqueness scores for a given list of topics.
    For each topic, the uniqueness is calculated as: (\sum_{i=1}^n 1/cnt(i)) / n,
    where n is the number of top words in the topic and cnt(i) is the counter for the number of times the word
    appears in the top words of all the topics.
    :param top_words_ids_all_topics: a list, each element is a list of top word indices for a topic
    :return: a dict, key is topic_id (starting from 0), value is topic_uniqueness score
    """
    n_topics = len(top_words_idx_all_topics)

    # build word_cnt_dict: number of times the word appears in top words
    word_cnt_dict = collections.Counter()
    for i in range(n_topics):
        word_cnt_dict.update(top_words_idx_all_topics[i])
        
    uniqueness_dict = dict()
    for i in range(n_topics):
        cnt_inv_sum = 0.0
        for ind in top_words_idx_all_topics[i]:
            cnt_inv_sum += 1.0 / word_cnt_dict[ind]
        uniqueness_dict[i] = cnt_inv_sum / len(top_words_idx_all_topics[i])

    return uniqueness_dict