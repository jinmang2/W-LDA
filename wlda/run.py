import os
import sys
import shutil
import pickle
import logging
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from .args import (
    MyArgumentParser,
    DataArguments,
    ModelArguments,
    TrainingArguments
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stream_hander = logging.StreamHandler()
stream_hander.setFormatter(formatter)

logger.addHandler(stream_hander)

from datasets import load_dataset


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
        (DataArguments, ModelArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # read args from json file
        data_args, model_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        # read args from shell script or real arguments
        data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    # (1) Check arguments
    assert model_args.latent_noise >= 0 and model_args.latent_noise <= 1
    # description은 domain + algorithm + model + (unsup or semisup or sup)

    if not data_args.saveto:
        data_args.saveto = "results/" + \
            training_args.description.replace("-", "/")
    
    # now_date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S/{}').format('')
    now_date = ""
    saveto = data_args.saveto + "/" + now_date

    if not os.path.exists(saveto):
        os.makedirs(saveto)
        os.makedirs(saveto + '/weights/encoder')
        os.makedirs(saveto + '/weights/decoder')
        os.makedirs(saveto + '/weights/discriminator_y')
        os.makedirs(saveto + "/args")
    
    data_args.saveto = saveto
    with open(saveto+'args/args.txt', 'w') as file:
        with open("run.sh", "r") as fp:
            for line in fp:
                file.write(line)

    if training_args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    training_args = device
        
    pickle.dump(data_args, open(saveto+"args/data_args.p", "wb"))
    pickle.dump(model_args, open(saveto+"args/model_args.p", "wb"))
    pickle.dump(training_args, open(saveto+"args/training_args.p", "wb"))


if __name__ == "__main__":
    main()
