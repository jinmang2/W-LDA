from args import (
    MyArgumentParser,
    DataArguments,
    ModelArguments,
    TrainingArguments
)
import sys
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stream_hander = logging.StreamHandler()
stream_hander.setFormatter(formatter)

logger.addHandler(stream_hander)

try:
    from datasets import load_dataset
    logger.info("import `load_dataset` from `huggingface.datasets`")
except ModuleNotFoundError as e:
    from .download import load_dataset
    logger.info("import `load_dataset` from `wlda.download`")


def main():
    parser = MyArgumentParser(
        (DataArguments, ModelArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args, model_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    data = load_dataset(
        path="wikitext.py",
        name="wikitext-103-v1",
        cache_dir="data",
    )
    print(data)


if __name__ == "__main__":
    main()
