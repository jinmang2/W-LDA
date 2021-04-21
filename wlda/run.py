import os
import sys
import shutil
import pickle
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset

from .args import (
    MyArgumentParser,
    AdvModelArguments,
    AdvTrainingArguments
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stream_hander = logging.StreamHandler()
stream_hander.setFormatter(formatter)

logger.addHandler(stream_hander)


def prepare_experiments():
    pass


def main():
    """
    5 Steps
    (1) Read arguments from json or shell script
    (2) Check and Prepare arguments for experiments
    (3) Download or Caching Data for experiments
    (4) If Statistical Methodology, cache the features
        else, tokenize input text(sentences or documents)
    (5) Define `Trainer`, `Model` for experiments
    (6) Run experiments or Execute test/inference script
    (7) Visualize or Reports the results
    """

    parser = MyArgumentParser(
        (AdvModelArguments, AdvTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # read args from json file
        model_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        # read args from shell script or real arguments
        model_args, training_args = parser.parse_args_into_dataclasses()

    # (1) Check arguments
    assert model_args.latent_noise >= 0 and model_args.latent_noise <= 1
    # description은 domain + algorithm + model + (unsup or semisup or sup)

    # output_dir 구축하는 부분 비슷하게는 만들되, 따라하진 않기
    # Trainer 클래스에서 담당할 것임
    # data_args.saveto = saveto
    # with open(saveto+'args/args.txt', 'w') as file:
    #     with open("run.sh", "r") as fp:
    #         for line in fp:
    #             file.write(line)
    # pickle.dump(model_args, open(saveto+"args/model_args.p", "wb"))
    # pickle.dump(training_args, open(saveto+"args/training_args.p", "wb"))

    import datasets
    datasets.set_caching_enabled(False)
    data = datasets.load_dataset(
        path="wikitext.py", name="wikitext-103-v1", split="train", cache_dir="data")
    def f(example):
        return {"text": example["text"], "feature": 1.}
    data = data.map(f, cache_file_name="data")
    print(data._data_files[0]["filename"])
    print(os.path.split(data._data_files[0]["filename"]))



if __name__ == "__main__":
    main()
