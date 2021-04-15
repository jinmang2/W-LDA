""" huggingface.datasets.load.py """
import os
import shutil
import importlib
from typing import (
    Optional, Callable, List, Union, Any, NewType, Iterable, Dict
)

from utils.download_manager import GenerateMode
from utils.filelock import FileLock
from utils.file_utils import (
    url_or_path_parent, 
    url_or_path_join, 
)

DATASET_INFOS_DICT_FILE_NAME = "dataset_infos.json"
MODULE_NAME_FOR_DYNAMIC_MODULES = "datasets_modules"

MODULE_PATH = NewType("module_path", str)
FILE_PATH = NewType("file_path", str)


def prepare_module(
    path : str,
    name : str,
    cache_dir : str = "",
    use_hash : bool = False,
    force_redownload : bool = False,
) -> Tuple[MODULE_PATH, FILE_PATH]:
    module_type = "dataset"
    script_name = list(
        filter(lambda x: x, path.replace(os.sep, "/").split("/")))[-1]
    if not script_name.endswith(".py"):
        raise AttributeError("")
    short_name = script_name[:-3]

    dynamic_modules_path = os.path.join(
        os.path.abspath(cache_dir), MODULE_NAME_FOR_DYNAMIC_MODULES)

    module_name_for_dynamic_modules = os.path.basename(dynamic_modules_path)
    datasets_modules_path = os.path.join(dynamic_modules_path, "datasets")
    datasets_modules_name = module_name_for_dynamic_modules + ".datasets"

    main_folder_path = os.path.join(datasets_modules_path, short_name)

    file_path = path
    local_path = path

    base_path = url_or_path_parent(file_path)  # remove the filename
    dataset_infos = url_or_path_join(base_path, DATASET_INFOS_DICT_FILE_NAME)

    hash = files_to_hash([local_path]) if use_hash else ""
    hash_folder_path = os.path.join(main_folder_path, hash)

    local_file_path = os.path.join(hash_folder_path, name)
    dataset_infos_path = os.path.join(
        hash_folder_path, DATASET_INFOS_DICT_FILE_NAME)

    # Prevent parallel disk operations
    lock_path = local_path + ".lock"
    with FileLock(lock_path):
        # Create main dataset/metrics folder if needed
        if force_redownload and os.path.exists(main_folder_path):
            shutil.rmtree(main_folder_path)

        if not os.path.exists(main_folder_path):
            logger.info(
                f"Creating main folder for {module_type} {file_path} at {main_folder_path}")
            os.makedirs(main_folder_path, exist_ok=True)
        else:
            logger.info(
                f"Found main folder for {module_type} {file_path} at {main_folder_path}")

        # add an __init__ file to the main dataset folder if needed
        init_file_path = os.path.join(main_folder_path, "__init__.py")
        if not os.path.exists(init_file_path):
            with open(init_file_path, "w"):
                pass

        # Create hash dataset folder if needed
        if not os.path.exists(hash_folder_path):
            logger.info(
                f"Creating specific version folder for {module_type} {file_path} at {hash_folder_path}")
            os.makedirs(hash_folder_path)
        else:
            logger.info(
                f"Found specific version folder for {module_type} {file_path} at {hash_folder_path}")

        # add an __init__ file to the hash dataset folder if needed
        init_file_path = os.path.join(hash_folder_path, "__init__.py")
        if not os.path.exists(init_file_path):
            with open(init_file_path, "w"):
                pass

        # Copy dataset.py file in hash folder if needed
        if not os.path.exists(local_file_path):
            logger.info("Copying script file from %s to %s",
                        file_path, local_file_path)
            shutil.copyfile(local_path, local_file_path)
        else:
            logger.info("Found script file from %s to %s",
                        file_path, local_file_path)

        # Copy dataset infos file if needed
        if not os.path.exists(dataset_infos_path):
            if local_dataset_infos_path is not None:
                logger.info("Copying dataset infos file from %s to %s",
                            dataset_infos, dataset_infos_path)
                shutil.copyfile(local_dataset_infos_path, dataset_infos_path)
            else:
                logger.info(
                    "Couldn't find dataset infos file at %s", dataset_infos)
        else:
            if local_dataset_infos_path is not None and not filecmp.cmp(local_dataset_infos_path, dataset_infos_path):
                logger.info("Updating dataset infos file from %s to %s",
                            dataset_infos, dataset_infos_path)
                shutil.copyfile(local_dataset_infos_path, dataset_infos_path)
            else:
                logger.info("Found dataset infos file from %s to %s",
                            dataset_infos, dataset_infos_path)

        # Record metadata associating original dataset path with local unique folder
        meta_path = local_file_path.split(".py")[0] + ".json"
        if not os.path.exists(meta_path):
            logger.info(
                f"Creating metadata file for {module_type} {file_path} at {meta_path}")
            meta = {"original file path": file_path,
                    "local file path": local_file_path}
            # the filename is *.py in our case, so better rename to filenam.json instead of filename.py.json
            with open(meta_path, "w", encoding="utf-8") as meta_file:
                json.dump(meta, meta_file)
        else:
            logger.info(
                f"Found metadata file for {module_type} {file_path} at {meta_path}")

        # Copy all the additional imports
        for import_name, import_path in local_imports:
            if os.path.isfile(import_path):
                full_path_local_import = os.path.join(
                    hash_folder_path, import_name + ".py")
                if not os.path.exists(full_path_local_import):
                    logger.info("Copying local import file from %s at %s",
                                import_path, full_path_local_import)
                    shutil.copyfile(import_path, full_path_local_import)
                else:
                    logger.info("Found local import file from %s at %s",
                                import_path, full_path_local_import)
            elif os.path.isdir(import_path):
                full_path_local_import = os.path.join(
                    hash_folder_path, import_name)
                if not os.path.exists(full_path_local_import):
                    logger.info("Copying local import directory from %s at %s",
                                import_path, full_path_local_import)
                    shutil.copytree(import_path, full_path_local_import)
                else:
                    logger.info("Found local import directory from %s at %s",
                                import_path, full_path_local_import)
            else:
                raise OSError(f"Error with local import at {import_path}")
