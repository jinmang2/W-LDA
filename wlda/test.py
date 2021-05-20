# import os
# import sys
# from .hf_transformers.trainer import Trainer
# from .args import (
#     MyArgumentParser,
#     ModelArguments,
#     AdvTrainingArguments
# )
# from datasets import load_dataset
# from .models.dirichlet import WassersteinAutoEncoder


# parser = MyArgumentParser((ModelArguments, AdvTrainingArguments))
# if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
#     # read args from json file
#     model_args, training_args = parser.parse_json_file(
#         json_file=os.path.abspath(sys.argv[1]))
# else:
#     # read args from shell script or real arguments
#     model_args, training_args = parser.parse_args_into_dataclasses()



# wae = WassersteinAutoEncoder(model_args)
# # train_data = load_dataset(
# #     path="wikitext.py",
# #     name="wikitext-103-v1",
# #     cache_dir="data/wikitext/",
# # )
# trainer = Trainer(model=wae, train_dataset=None, args=training_args)
# print(trainer)

from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % \
            (f.__name__, te-ts))
        return result
    return wrap


import torch


@timing
def calc1(x):
    n = x.shape[0]
    sum_ = torch.zeros(1)
    for i in range(n):
        for j in range(i+1, n):
            sum_ = sum_ + torch.sum(torch.abs(x[i] - x[j]))
    return sum_


@timing
def calc2(x):
    n = x.shape[0]
    sum_ = torch.zeros(1)
    for i in range(1, n):
        sum_ = sum_ + torch.sum((x[:-i] - x[i:]).abs())
    return sum_


x = torch.randn(32, 50)
out1 = calc1(x) / (32 * 31)
out2 = calc2(x) / (32 * 31)
print(out1, out2)
