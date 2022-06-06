#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/2/17 16:27 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/17 16:27   wangfc      1.0         None
"""

import copy
import glob
import hashlib
import logging
import os
import shutil
from subprocess import CalledProcessError, DEVNULL, check_output  # skipcq:BAN-B404
import tempfile
import typing
from pathlib import Path
from typing import Any, Text, Tuple, Union, Optional, List, Dict, NamedTuple

# from rasa.utils.common import TempDirectoryPath
from utils.common import TempDirectoryPath
from utils.constants import DEFAULT_MODELS_PATH
from utils.exceptions import ModelNotFound

import logging

logger = logging.getLogger(__name__)


# from rasa.model import get_model

def get_model(model_path: Text = DEFAULT_MODELS_PATH,use_compressed_model:bool =True) -> TempDirectoryPath:
    """Gets a model and unpacks it.

    Args:
        model_path: Path to the zipped model. If it's a directory, the latest
                    trained model is returned.
        use_compressed_model: 增加 use_compressed_model 控制 是否从压缩文件中读取
    Returns:
        Path to the unpacked model.

    Raises:
        ModelNotFound Exception: When no model could be found at the provided path.

    """
    model_path = get_local_model(model_path,use_compressed_model=use_compressed_model)

    try:
        model_relative_path = os.path.relpath(model_path)
    except ValueError:
        model_relative_path = model_path

    # logger.info(f"Loading model {model_relative_path}...")
    if use_compressed_model:
        return unpack_model(model_path)
    else:
        return model_path


def get_local_model(model_path: Text = DEFAULT_MODELS_PATH, use_compressed_model:bool =True) -> Text:
    """Returns verified path to local model archive.

    Args:
        model_path: Path to the zipped model. If it's a directory, the latest
                    trained model is returned.
        use_compressed_model: 增加 use_compressed_model 控制 是否从压缩文件中读取
    Returns:
        Path to the zipped model. If it's a directory, the latest
                    trained model is returned.

    Raises:
        ModelNotFound Exception: When no model could be found at the provided path.

    """
    if not model_path:
        raise ModelNotFound("No path specified.")
    elif not os.path.exists(model_path):
        raise ModelNotFound(f"No file or directory at '{model_path}'.")

    if os.path.isdir(model_path):
        model_path = get_latest_model(model_path,use_compressed_model=use_compressed_model)
        if not model_path:
            raise ModelNotFound(
                f"Could not find any Rasa model files in '{model_path}'."
            )
    elif not model_path.endswith(".tar.gz"):
        raise ModelNotFound(f"Path '{model_path}' does not point to a Rasa model file.")

    return model_path


def get_latest_model(model_path: Text = DEFAULT_MODELS_PATH, use_compressed_model:bool =True) -> Optional[Text]:
    """Get the latest model from a path.

    Args:
        model_path: Path to a directory containing zipped models.
        use_compressed_model: 增加 use_compressed_model 控制 是否从压缩文件中读取

    Returns:
        Path to latest model in the given directory.

    """
    if not os.path.exists(model_path) or os.path.isfile(model_path):
        model_path = os.path.dirname(model_path)

    if use_compressed_model:
        list_of_files = glob.glob(os.path.join(model_path, "*.tar.gz"))
    else:
        from utils.io import list_subdirectory
        list_of_files = list_subdirectory(model_path)


    if len(list_of_files) == 0:
        return None

    return max(list_of_files, key=os.path.getctime)


def unpack_model(
        model_file: Text, working_directory: Optional[Union[Path, Text]] = None
) -> TempDirectoryPath:
    """Unpack a zipped Rasa model.

    Args:
        model_file: Path to zipped model.
        working_directory: Location where the model should be unpacked to.
                           If `None` a temporary directory will be created.

    Returns:
        Path to unpacked Rasa model.

    """
    import tarfile

    if working_directory is None:
        working_directory = tempfile.mkdtemp()

    # All files are in a subdirectory.
    try:
        with tarfile.open(model_file, mode="r:gz") as tar:
            tar.extractall(working_directory)
            logger.debug(f"Extracted model to '{working_directory}'.")
    except Exception as e:
        logger.error(f"Failed to extract model at {model_file}. Error: {e}")
        raise

    return TempDirectoryPath(working_directory)


def get_model_subdirectories(
        unpacked_model_path: Text,
) -> Tuple[Optional[Text], Optional[Text]]:
    """Return paths for Core and NLU model directories, if they exist.
    If neither directories exist, a `ModelNotFound` exception is raised.

    Args:
        unpacked_model_path: Path to unpacked Rasa model.

    Returns:
        Tuple (path to Core subdirectory if it exists or `None` otherwise,
               path to NLU subdirectory if it exists or `None` otherwise).

    """
    # from rasa.shared.constants import DEFAULT_CORE_SUBDIRECTORY_NAME
    # from rasa.shared.constants import DEFAULT_NLU_SUBDIRECTORY_NAME

    from utils.constants import DEFAULT_CORE_SUBDIRECTORY_NAME, DEFAULT_NLU_SUBDIRECTORY_NAME
    core_path = os.path.join(unpacked_model_path, DEFAULT_CORE_SUBDIRECTORY_NAME)

    nlu_path = os.path.join(unpacked_model_path, DEFAULT_NLU_SUBDIRECTORY_NAME)

    if not os.path.isdir(core_path):
        core_path = None

    if not os.path.isdir(nlu_path):
        nlu_path = None

    if not core_path and not nlu_path:
        raise ModelNotFound(
            "No NLU or Core data for unpacked model at: '{}'.".format(
                unpacked_model_path
            )
        )

    return core_path, nlu_path



