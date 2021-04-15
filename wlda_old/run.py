import sys
from args import (
    MyArgumentParser, 
    DataArguments, 
    ModelArguments, 
    TrainingArguments
)


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
    
    # ==================================================
    # First Step
    # ==================================================
    assert training_args["latent_noise"] >= 0 and trainig_args["latent_noise"] <= 1





if __name__ == "__main__":
    main()
