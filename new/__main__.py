import os
# from .datasets.load import prepare_module
from .datasets import config


# prepare_module('wikitext.py')
# print(os.environ.get("USE_TORCH", "AUTO").upper())

# print(config.TORCH_AVAILABLE)
# print(config.TF_AVAILABLE)
# print(config.BEAM_AVAILABLE)

# from .datasets.utils.file_utils import get_datasets_user_agent

# print(get_datasets_user_agent())

from .datasets.load import prepare_module


prepare_module("wikitext.py")