import os
import sys
from .hf_transformers.trainer import Trainer
from .args import (
    MyArgumentParser,
    ModelArguments,
    AdvTrainingArguments
)
from datasets import load_dataset
from .models.dirichlet import WassersteinAutoEncoder


parser = MyArgumentParser((ModelArguments, AdvTrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # read args from json file
    model_args, training_args = parser.parse_json_file(
        json_file=os.path.abspath(sys.argv[1]))
else:
    # read args from shell script or real arguments
    model_args, training_args = parser.parse_args_into_dataclasses()



wae = WassersteinAutoEncoder(model_args)
# train_data = load_dataset(
#     path="wikitext.py",
#     name="wikitext-103-v1",
#     cache_dir="data/wikitext/",
# )
trainer = Trainer(model=wae, train_dataset=None, args=training_args)
print(trainer)
